#!/usr/bin/env python3
"""Retarget five CC0 Vicon Plug-in Gait strides to the viewer's shared rigid skeleton.

Requires c3d, numpy, scipy and opensim. By default only the five compressed C3D
members (5.4 MB) are fetched. Alternatively pass the extracted source directory.
"""

import json
import struct
import zlib
from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import Request, urlopen

import c3d
import numpy as np
import opensim
from prepare_whole_body import JOINT_FRAMES, MODEL_PATH, RAW, SEGMENTS, simtk_values
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
STUDY = "https://doi.org/10.1038/s41597-023-02767-y"
DATASET = "https://doi.org/10.6084/m9.figshare.24192480.v1"
ARCHIVE = "https://ndownloader.figshare.com/files/42450060"


@dataclass(frozen=True)
class Trial:
    subject: int
    trial: int
    stride: int  # zero-based consecutive right-strike interval
    start_byte: int
    byte_count: int
    digest: str

    @property
    def filename(self):
        return f"SUBJ{self.subject} ({self.trial}).c3d"

    @property
    def participant(self):
        return f"SUBJ{self.subject:02d}"


TRIALS = (
    Trial(1, 1, 0, 225006, 1039185, "0c3e446ea03ed8fc50de2b0f7e8a7fbadad1597ef6dabf5e374d60ec3c0b1e97"),
    Trial(2, 3, 0, 6372700, 1372367, "8c1a35834aec9b4812fb657dfa2e91f0f27bb904aeac8d0f018591faa8d4c6b2"),
    Trial(3, 1, 0, 7979736, 907606, "970d0bec8a3ef7d0475010e4cdaede0c01a118c55178a929626117028d108c90"),
    Trial(4, 3, 1, 13074344, 926065, "9a44a31368141099d3541134ed0ae78adc4a91ab7a4e0e870e0c42f3c5bd0440"),
    Trial(130, 1, 0, 148965716, 1142082, "3ce2df1c671e8ed4d6e53f49b53dcee1635529a0e284cf500e267d7625b93b9d"),
)


# From Supplementary Table 1, identified using the C3D directory's numeric subject ID.
SUPPLEMENT = (
    "https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41597-023-02767-y/"
    "MediaObjects/41597_2023_2767_MOESM1_ESM.pdf"
)
DEMOGRAPHICS = {
    1: (86, "Male", 1.580, 64),
    2: (85, "Female", 1.500, 78),
    3: (85, "Female", 1.510, 69),
    4: (84, "Male", 1.625, 70),
    130: (21, "Female", 1.590, 65),
}


def trial_bytes(trial, directory):
    if directory:
        (path,) = directory.rglob(trial.filename)
        content = path.read_bytes()
    else:
        end = trial.start_byte + trial.byte_count - 1
        request = Request(
            f"{ARCHIVE}?member={trial.participant}", headers={"Range": f"bytes={trial.start_byte}-{end}"}
        )
        with urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise ValueError("Figshare did not return a partial archive; use --source-directory")
            content = zlib.decompress(response.read(trial.byte_count + 1), -15)
    if sha256(content).hexdigest() != trial.digest:
        raise ValueError(f"Unexpected source file for {trial.participant}")
    return content


def unit(v):
    length = np.linalg.norm(v)
    if length < 1e-7:
        raise ValueError("Degenerate segment axis")
    return v / length


def basis(anterior, proximal):
    """A right-handed frame with proximal axis and projected anterior axis."""
    y = unit(proximal)
    z = unit(np.cross(anterior, y))
    return np.column_stack((np.cross(y, z), y, z))


def read_stride(trial, directory):
    reader = c3d.Reader(BytesIO(trial_bytes(trial, directory)))
    names = [name.strip().split(":")[-1] for name in reader.point_labels]
    events = {}
    for side, label, (minute, seconds) in zip(
        reader.get("EVENT:CONTEXTS").string_array,
        reader.get("EVENT:LABELS").string_array,
        reader.get("EVENT:TIMES").float_array,
        strict=True,
    ):
        events.setdefault((side.strip(), label.strip()), []).append(
            round(60 * float(minute) + float(seconds), 5)
        )
    strikes = sorted(events["Right", "Foot Strike"])
    start, end = strikes[trial.stride : trial.stride + 2]
    (off,) = [t for t in events["Right", "Foot Off"] if start < t < end]
    if reader.get("POINT:UNITS").string_value.strip().lower() != "mm":
        raise ValueError("Expected marker coordinates in mm")
    rate = float(reader.point_rate)
    # Nexus frame 1 is t=0. Absolute point frame numbers are retained on cropping.
    first, last = (round(t * rate) + 1 for t in (start, end))
    rows = [points for index, points, _ in reader.read_frames() if first <= index <= last]
    if len(rows) != last - first + 1:
        raise ValueError("Missing frames between right strikes")
    points = np.stack(rows)
    needed = {"PELO", "PELA", "PELP", "TRXO", "TRXA", "TRXP"}
    for side in "RL":
        needed.update(side + name for name in ("KneeAngles", "WRA", "WRB", "FIN", "CLO", "HNO", "HEE", "TOE"))
        for seg in ("FE", "TI", "FO", "HU", "RA"):
            needed.update(side + seg + axis for axis in "OAP")
    phase = np.linspace(0, len(points) - 1, 101)
    samples = {}
    for name in sorted(needed):
        data = points[:, names.index(name)]
        if np.any(data[:, 3] < 0) or not np.isfinite(data[:, :3]).all():
            raise ValueError(f"{trial.filename} has gaps in {name}")
        samples[name] = np.stack(
            [np.interp(phase, np.arange(len(data)), col) for col in data[:, :3].T], axis=1
        )

    pelvis = samples["PELO"] / 1000
    travel = pelvis[-1] - pelvis[0]
    travel[2] = 0
    forward = unit(travel)
    up = np.array([0.0, 0.0, 1.0])
    lab_to_view = np.stack((forward, up, np.cross(forward, up)))
    for name, values in samples.items():
        if not name.endswith("Angles"):
            samples[name] = values @ lab_to_view.T / 1000
    # Swing clearance comes from the lower of the recorded heel and toe markers.
    # Remove marker mounting height and set the observed strike/off boundaries to zero.
    time = np.linspace(start, end, 101)
    left_off = min(t for t in events["Left", "Foot Off"] if start < t < end)
    left_strike = min(t for t in events["Left", "Foot Strike"] if left_off < t < end)
    for side, lift, land in (("R", off, end), ("L", left_off, left_strike)):
        heights = np.stack([samples[side + mark][:, 1] for mark in ("HEE", "TOE")])
        low = (heights - np.quantile(heights, 0.1, axis=1, keepdims=True)).min(axis=0)
        boundary = np.interp([lift, land], time, low)
        excursion = low - np.interp(time, [lift, land], boundary)
        samples[side + "Clearance"] = np.where((time > lift) & (time < land), np.maximum(0, excursion), 0)
    return samples, start, off, end, rate, first, last


def reference_rig(model_path):
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    rest = {}
    for name in SEGMENTS:
        transform = model.getBodySet().get(name).getTransformInGround(state)
        quaternion = simtk_values(transform.R().convertRotationToQuaternion(), 4)[[1, 2, 3, 0]]
        rest[name] = (simtk_values(transform.p(), 3), Rotation.from_quat(quaternion).as_matrix())
    joints = {}
    for name, (joint, side) in JOINT_FRAMES.items():
        link = model.getJointSet().get(joint)
        frame = link.getParentFrame() if side == "parent" else link.getChildFrame()
        joints[name] = simtk_values(frame.getTransformInGround(state).p(), 3)
    return rest, joints


def mesh_points(glb_path):
    raw = glb_path.read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20 : 20 + length])
    binary_start = 28 + length
    meshes = {}
    for node in gltf["nodes"]:
        if "mesh" not in node:
            continue
        (primitive,) = gltf["meshes"][node["mesh"]]["primitives"]
        accessor = gltf["accessors"][primitive["attributes"]["POSITION"]]
        view = gltf["bufferViews"][accessor["bufferView"]]
        assert accessor["componentType"] == 5126 and "byteStride" not in view
        meshes[node["name"]] = np.frombuffer(
            raw,
            dtype="<f4",
            count=3 * accessor["count"],
            offset=binary_start + view["byteOffset"] + accessor.get("byteOffset", 0),
        ).reshape(-1, 3)
    return meshes


def frame_pose(p, rest, joints, meshes, travel_scale, fit=None):
    """Orient native segments to the observations and connect them at the model pivots."""
    frames = {}
    x = np.array([1.0, 0.0, 0.0])

    def axes(seg, forward_sign=1, up_sign=1):
        return basis(forward_sign * (p[seg + "A"] - p[seg + "O"]), up_sign * (p[seg + "P"] - p[seg + "O"]))

    def put(segment, pivot, anchor, rotation):
        rest_origin, rest_rotation = rest[segment]
        frames[segment] = (anchor + rotation @ (rest_origin - pivot), rotation @ rest_rotation)

    def joint_position(segment, point):
        origin, rot = frames[segment]
        rest_origin, rest_rot = rest[segment]
        return origin + rot @ rest_rot.T @ (joints[point] - rest_origin)

    zero = rest["pelvis"][0]
    put("pelvis", zero, p["PELO"], axes("PEL"))
    put("torso", joints["lumbar"], joint_position("pelvis", "lumbar"), axes("TRX", up_sign=-1))

    for side, word, suffix in (("R", "right", ""), ("L", "left", "_l")):
        hip, knee, ankle = (word + k for k in ("Hip", "Knee", "Ankle"))
        shoulder, elbow, wrist = (word + k for k in ("Shoulder", "Elbow", "Wrist"))
        upper, ulna, radius, hand = (k + "_" + side.lower() for k in ("humerus", "ulna", "radius", "hand"))
        femur, tibia = "femur" + suffix, "tibia" + suffix
        thigh_source = axes(side + "FE")
        thigh = thigh_source @ basis(x, joints[hip] - joints[knee]).T
        shin = axes(side + "TI") @ basis(x, joints[knee] - joints[ankle]).T
        put(femur, joints[hip], joint_position("pelvis", hip), thigh)
        knee_pos = joint_position(femur, knee)
        put(tibia, joints[knee], knee_pos, shin)
        put("patella" + suffix, joints[knee], knee_pos, thigh)
        foot = side + "FO"
        foot_rotation = basis(p[foot + "O"] - p[foot + "P"], p[foot + "A"] - p[foot + "O"])
        ankle_pos = joint_position(tibia, ankle)
        for seg in ("talus", "calcaneus", "toes"):
            put(seg + suffix, joints[ankle], ankle_pos, foot_rotation)

        source_shoulder, source_elbow, source_wrist = (p[side + k] for k in ("CLO", "HUO", "RAO"))
        arm = basis(p[side + "HUO"] - p[side + "HUA"], source_shoulder - source_elbow)
        arm = arm @ basis(x, joints[shoulder] - joints[elbow]).T
        put(upper, joints[shoulder], joint_position("torso", shoulder), arm)
        # Wrist A and B are the radial and ulnar markers respectively. Index-to-little
        # metacarpal separation supplies the matching thumbward axis on the reference rig.
        radial_axis = p[side + "WRA"] - p[side + "WRB"]
        label = word.capitalize()
        target_radial = rest[hand][1] @ (
            meshes[label + " index proximal"].mean(axis=0) - meshes[label + " little proximal"].mean(axis=0)
        )
        forearm = (
            basis(radial_axis, source_elbow - source_wrist)
            @ basis(target_radial, joints[elbow] - joints[wrist]).T
        )
        elbow_pos = joint_position(upper, elbow)
        for seg in (ulna, radius):
            put(seg, joints[elbow], elbow_pos, forearm)
        hand_axis = source_wrist - p[side + "HNO"]
        target_axis = -rest[hand][1] @ meshes[label + " metacarpal3"].mean(axis=0)
        hand_rot = basis(radial_axis, hand_axis) @ basis(target_radial, target_axis).T
        put(hand, joints[wrist], joint_position(radius, wrist), hand_rot)

    # Match observed ankle travel and sole clearance despite differing leg proportions.
    # A temporally coupled fit balances ankle targets, observed knee bend and pelvis travel.
    # Solve each fixed-length knee in its observed flexion plane.
    leg_targets = []
    for side, word, suffix in (("R", "right", ""), ("L", "left", "_l")):
        hip = joint_position("pelvis", word + "Hip")
        knee = joint_position("femur" + suffix, word + "Knee")
        ankle = joint_position("tibia" + suffix, word + "Ankle")
        sole_y = min(
            np.min(meshes[word.capitalize() + " " + mesh] @ frames[seg + suffix][1][1])
            + frames[seg + suffix][0][1]
            for mesh, seg in (("hindfoot and midfoot", "calcaneus"), ("forefoot", "toes"))
        )
        target = hip + (p[side + "TIO"] - p[side + "FEP"]) * travel_scale
        # Both ankles share the recorded pelvis origin in the frontal plane.
        # Measuring each from its own model hip would subtract the difference
        # between the participant's hip width and the display skeleton's.
        target[2] = p["PELO"][2] + (p[side + "TIO"][2] - p["PELO"][2]) * travel_scale
        target[1] = ankle[1] - sole_y + p[side + "Clearance"] * travel_scale
        upper_length = np.linalg.norm(hip - knee)
        lower_length = np.linalg.norm(knee - ankle)
        leg_targets.append((side, word, suffix, hip, knee, ankle, target, upper_length, lower_length))
    source_height = p["PELO"][1] * travel_scale
    desired_shift = source_height - np.mean([leg[3][1] for leg in leg_targets])
    if fit is None:
        return (
            np.stack([leg[3] - leg[6] for leg in leg_targets]),
            np.array([np.linalg.norm(leg[3] - leg[5]) for leg in leg_targets]),
            desired_shift,
        )
    dx, dy, right_dx, left_dx = fit
    for leg, delta in zip(leg_targets, (right_dx, left_dx), strict=True):
        leg[6][0] += delta
        hip, target, upper_length, lower_length = leg[3], leg[6], leg[7], leg[8]
        horizontal = np.sum((target[[0, 2]] - hip[[0, 2]] - np.array([dx, 0])) ** 2)
        ceiling = np.sqrt((upper_length + lower_length - 0.00001) ** 2 - horizontal)
        dy = min(dy, target[1] + ceiling - hip[1])
    shift = np.array([dx, dy, 0.0])
    frames = {name: (origin + shift, rot) for name, (origin, rot) in frames.items()}

    def reorient(old, new):
        old, new = unit(old), unit(new)
        axis = np.cross(old, new)
        sine = np.linalg.norm(axis)
        if sine < 1e-10:
            return np.eye(3)
        return Rotation.from_rotvec(axis * (np.arctan2(sine, old @ new) / sine)).as_matrix()

    for side, word, suffix, hip, knee, ankle, target, upper_length, lower_length in leg_targets:
        up = unit(hip + shift - target)
        reach = np.linalg.norm(hip + shift - target)
        along = (upper_length**2 - lower_length**2 + reach**2) / (2 * reach)
        offset = np.sqrt(max(0.0, upper_length**2 - along**2))
        circle_center = hip + shift - along * up
        # The measured knee determines where to bend around the hip-to-ankle
        # line. A thigh frame's anterior axis alone does not encode this point.
        source_knee = p[side + "FEO"]
        source_up = unit(p[side + "FEP"] - p[side + "TIO"])
        source_shin = source_knee - p[side + "TIO"]
        bend = (source_shin - source_up * (source_up @ source_shin)) * travel_scale
        bend[2] = p["PELO"][2] + (source_knee[2] - p["PELO"][2]) * travel_scale - circle_center[2]
        forward = unit(bend - up * (up @ bend))
        new_knee = circle_center + offset * forward
        old_thigh = frames["femur" + suffix][1] @ rest["femur" + suffix][1].T
        thigh = reorient(knee - hip, new_knee - hip - shift) @ old_thigh
        old_shin = frames["tibia" + suffix][1] @ rest["tibia" + suffix][1].T
        shin = reorient(ankle - knee, target - new_knee) @ old_shin
        put("femur" + suffix, joints[word + "Hip"], hip + shift, thigh)
        put("patella" + suffix, joints[word + "Knee"], new_knee, thigh)
        put("tibia" + suffix, joints[word + "Knee"], new_knee, shin)
        for seg in ("talus", "calcaneus", "toes"):
            name = seg + suffix
            origin, rot = frames[name]
            frames[name] = (origin + target - ankle - shift, rot)
    return frames


def fit_cycle(targets):
    """Fit fixed-length legs across the whole capture, regularizing corrections in time."""
    hip_minus_ankle = np.stack([row[0] for row in targets])
    desired_reach = np.stack([row[1] for row in targets])
    desired_y = np.array([row[2] for row in targets])
    n = len(targets)
    # Cross-strike continuity applies to retarget corrections. Original source
    # differences at consecutive heel strikes remain in the recorded poses.
    identity = sparse.eye(n, format="csr")
    d1 = identity - identity[np.roll(np.arange(n), 1)]
    d2 = d1 @ d1
    pattern = sparse.vstack(
        [
            sparse.kron(identity, np.ones((6, 4))),
            sparse.kron(abs(d1), np.eye(4)),
            sparse.kron(abs(d2), np.eye(4)),
        ]
    ).tocsr()

    def residual(parameters):
        values = parameters.reshape(n, 4)
        difference = hip_minus_ankle.copy()
        difference[:, :, 0] += values[:, 0, None] - values[:, 2:4]
        difference[:, :, 1] += values[:, 1, None]
        reach = np.linalg.norm(difference, axis=2)
        prior = values.copy()
        prior[:, 1] -= desired_y
        pose_error = np.column_stack((prior / [0.012, 0.018, 0.025, 0.025], (reach - desired_reach) / 0.0015))
        velocity = d1 @ values
        acceleration = d2 @ values
        return np.concatenate(
            (
                pose_error.ravel(),
                (velocity / [0.0014, 0.002, 0.003, 0.003]).ravel(),
                (acceleration / 0.00028).ravel(),
            )
        )

    initial = np.zeros((n, 4))
    initial[:, 1] = desired_y
    fit = least_squares(residual, initial.ravel(), jac_sparsity=pattern, ftol=1e-6, xtol=1e-6, max_nfev=100)
    if not fit.success:
        raise ValueError(f"Stride fit did not converge: {fit.message}")
    return fit.x.reshape(n, 4)


def clip(trial, directory, rest, joints, meshes):
    samples, start, off, end, rate, first, last = read_stride(trial, directory)
    pelvis = samples["PELO"]
    speed = (pelvis[-1, 0] - pelvis[0, 0]) / (end - start)
    # FEP is the proximal tip of the Plug-in Gait thigh frame (hip center);
    # FEO and TIO are the knee and ankle joint centers.
    target_leg = sum(
        np.linalg.norm(joints[a] - joints[b])
        for a, b in (("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"))
    )
    leg_lengths = [
        np.linalg.norm(samples[side + "FEP"] - samples[side + "FEO"], axis=1)
        + np.linalg.norm(samples[side + "FEO"] - samples[side + "TIO"], axis=1)
        for side in "RL"
    ]
    travel_scale = target_leg / np.median(leg_lengths)
    observed = [{key: series[i] for key, series in samples.items()} for i in range(101)]
    targets = [frame_pose(p, rest, joints, meshes, travel_scale) for p in observed]
    corrections = fit_cycle(targets)
    frames, previous = [], {}
    for i, (p, correction) in enumerate(zip(observed, corrections, strict=True)):
        pose = frame_pose(p, rest, joints, meshes, travel_scale, correction)
        row = []
        for name in SEGMENTS:
            origin, rot = pose[name]
            origin[[0, 2]] += (pelvis[i, [0, 2]] - pelvis[0, [0, 2]]) * (travel_scale - 1) - pelvis[0, [0, 2]]
            quat = Rotation.from_matrix(rot).as_quat()
            if name in previous and quat @ previous[name] < 0:
                quat = -quat
            previous[name] = quat
            row.extend(origin.round(5).tolist() + quat.round(6).tolist())
        frames.append(row)
    packed = np.array(frames).reshape(len(frames), len(SEGMENTS), 7)
    pelvis_pose = packed[:, SEGMENTS.index("pelvis"), :3]
    detrended = pelvis_pose - np.linspace(pelvis_pose[0], pelvis_pose[-1], len(frames))
    if np.max(np.linalg.norm(np.diff(detrended, axis=0), axis=1)) > 0.02:
        raise ValueError(f"{trial.participant}: discontinuous pelvis motion")
    for side, word in (("R", "Right"), ("L", "Left")):
        for joint, segment, source in (("Knee", "femur", "FEO"), ("Ankle", "tibia", "TIO")):
            name = segment + ("_l" if side == "L" else "")
            local = rest[name][1].T @ (joints[word.lower() + joint] - rest[name][0])
            pose = packed[:, SEGMENTS.index(name)]
            center = pose[:, :3] + Rotation.from_quat(pose[:, 3:]).apply(local)
            expected_z = (samples[side + source][:, 2] - pelvis[:, 2]) * travel_scale
            error = center[:, 2] - pelvis_pose[:, 2] - expected_z
            # Grounding the fixed-length leg can move the knee off its marker.
            # Ankle lateral targets are exactly attainable up to archive rounding.
            rms_limit = 0.012 if joint == "Knee" else 0.00002
            if np.sqrt(np.mean(error**2)) > rms_limit:
                raise ValueError(f"{trial.participant}: {word} {joint} lateral track differs from recording")
        hand = packed[:, SEGMENTS.index("hand_" + side.lower())]
        wrist_curve = hand[:, 0] - pelvis_pose[:, 0]
        source_wrist_curve = samples[side + "RAO"][:, 0] - pelvis[:, 0]
        if np.corrcoef(wrist_curve, source_wrist_curve)[0, 1] < 0.96:
            raise ValueError(f"{trial.participant}: {word} wrist no longer tracks its recording")
        thumbward = meshes[word + " index proximal"].mean(0) - meshes[word + " little proximal"].mean(0)
        palm_normal = unit(np.cross(meshes[word + " metacarpal3"].mean(0), thumbward))
        curl = meshes[word + " middle distal"].mean(0) - meshes[word + " middle proximal"].mean(0)
        palm_normal *= np.sign(curl @ palm_normal)
        palm_world = Rotation.from_quat(hand[:, 3:]).apply(palm_normal)
        wrist_dorsum = (samples[side + "WRA"] + samples[side + "WRB"]) / 2 - samples[side + "RAO"]
        wrist_dorsum /= np.linalg.norm(wrist_dorsum, axis=1)[:, None]
        if np.max(np.sum(palm_world * wrist_dorsum, axis=1)) > -0.8:
            raise ValueError(f"{trial.participant}: {word} palm faces the dorsal wrist bar")
    signals = {
        "rightKneeFlexionDeg": samples["RKneeAngles"][:, 0].round(3).tolist(),
        "leftKneeFlexionDeg": samples["LKneeAngles"][:, 0].round(3).tolist(),
        "pelvisVerticalDisplacementM": (pelvis[:, 1] - pelvis[0, 1]).round(5).tolist(),
    }
    wrist_excursions = {}
    for word, side in (("right", "R"), ("left", "L")):
        fore_aft = samples[side + "RAO"][:, 0] - pelvis[:, 0]
        wrist_excursions[word] = round(float(np.ptp(fore_aft)), 4)
    age, sex, height, mass = DEMOGRAPHICS[trial.subject]
    return {
        "ageYears": age,
        "sex": sex,
        "heightM": height,
        "massKg": mass,
        "cohort": "Able-bodied",
        "id": f"vicon-{trial.participant.lower()}-trial-{trial.trial}",
        "participant": trial.participant,
        "recording": trial.filename,
        "condition": "Barefoot overground walking at preferred speed",
        "phase": "stride",
        "heading": 1,
        "durationSeconds": round(end - start, 4),
        "stanceEndPct": round(100 * (off - start) / (end - start), 3),
        "walkingSpeedMps": round(float(speed), 4),
        "wristExcursionM": wrist_excursions,
        "sourceFrames": [first, last],
        "sourceFrameRateHz": rate,
        "sourceSha256": trial.digest,
        "rightFootEventSeconds": [start, off, end],
        "frames": frames,
        "signals": signals,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-directory", type=Path, help="Directory containing the original extracted C3D files"
    )
    parser.add_argument("--model", type=Path, help="Pinned Harba display model")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/reference-walks.json")
    args = parser.parse_args()
    meshes = mesh_points(PROJECT / "assets/body.glb")
    with TemporaryDirectory() as tmp:
        model = args.model
        if model is None:
            model = Path(tmp) / "reference.osim"
            with urlopen(f"{RAW}/{quote(MODEL_PATH)}", timeout=60) as response:
                model.write_bytes(response.read())
        rest, joints = reference_rig(model)
    data = {
        "source": {
            "title": "Van Criekinge et al. · Full-body optical gait recordings",
            "authors": "Tamaya Van Criekinge, Wim Saeys, Steven Truijen, Luc Vereeck, Lizeth H. Sloot, Ann Hallemans",
            "url": DATASET,
            "publication": STUDY,
            "demographicsUrl": SUPPLEMENT,
            "license": "CC0 1.0",
            "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
            "acquisition": "Barefoot overground walking at self-selected speed, recorded using 100 Hz Vicon Plug-in Gait.",
            "processing": "Each stride spans consecutive source-labeled right foot strikes and includes one right toe-off. Marker and fitted segment trajectories are sampled at 101 phase points. Source knee flexion, pelvis displacement, pelvis travel speed and pelvis-relative wrist excursions are reported in original subject units.",
            "retargeting": "Recorded optical segment axes, joint-center directions, radial/ulnar wrist landmarks and Vicon hand centers derived from the third metacarpal markers orient the bones of the shared display model. Arm and hand motion is specific to each recorded side. Bone lengths are fixed and horizontal travel is scaled to model leg length. Both ankle targets follow their measured lateral tracks relative to a common pelvis origin. Measured knee centers guide knee bend around each hip-to-ankle line. A temporally regularized two-link leg fit balances tracked ankle positions and knee flexion while grounding the bony sole during stance. Swing foot height follows the lower heel/toe marker relative to its contact baseline. Toes and rearfoot share a rigid segment, the patella follows the femur, and skull and fingers follow torso and hand respectively.",
        },
        "segmentOrder": list(SEGMENTS),
        "clips": [clip(trial, args.source_directory, rest, joints, meshes) for trial in TRIALS],
    }
    args.output.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Wrote {len(data['clips'])} participant strides to {args.output}")


if __name__ == "__main__":
    main()
