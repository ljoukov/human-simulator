"""Replay published Falisse gait solutions using the corresponding OpenSim model.

Requires numpy, scipy, opensim 4.6. Example:
    python predsim_motion.py /tmp/falisse --fetch --output-dir /tmp/gait-clips
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.io import loadmat

REV = "5fa3929fa9ba29d5c057c16b18671a017d8f15c1"
REPO = "https://github.com/antoinefalisse/3dpredictsim"
FILES = {
    "Results/PredSim_all/Results_all.mat": "8469220733184270ae20e55f9014baa0b428951d82597a93ca70494d826ab2a1",
    "OpenSimModel/subject1/subject1.osim": "3b223f733293fc16f5c20cf7e74091d78ec61f3d451c91655498c71d5118e6e9",
    "Results/PredSim_all_v2/Results_all_v2.mat": "a637b860297a9ac445bff2c90ccd00f5795d6fa0f4f11691affe809fe9bc56d6",
    "OpenSimModel/subject1/subject1_v2.osim": "a5e089523014f7353cd1f2bc8b92c3785e2dcc2d63c3aea746798535d6d0c8a6",
}
SPEED = 1.33
# getSettings_predSim_all.m. Same task, initial guess, contact and objective
# for the four original archived hip weakness solves.
COMMON_CASE = (
    "Speed_133/W_MetabolicEnergyRate_500/W_MuscleActivity_2000/"
    "W_JointAcceleration_50000/W_PassiveTorque_1000/W_ArmExcitation_1000000/"
    "Power_MetabolicEnergyRate_2/InitialGuessType_1/InitialGuessMode_0/InitialGuessCase_0/"
    "WeaknessHipActuators_{weakness}/WeaknessAnklePlantarflexors_0/MetabolicEnergyModel_0/"
    "ContactModel_1/Number_MeshIntervals_50/MaximumContractionVelocity_0/CoContraction_0"
)


def source_files(directory, fetch=False):
    for key, checksum in FILES.items():
        path = directory / Path(key).name
        if fetch and not path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            raw = f"https://raw.githubusercontent.com/antoinefalisse/3dpredictsim/{REV}/{key}"
            with urlopen(raw, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(f"Checksum mismatch downloading {raw}")
            path.write_bytes(content)
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Unexpected source version: {path}")


def xyz(vector):
    return np.array([vector.get(i) for i in range(3)])


def replay(case, labels, model_path, clip_id, label):
    import opensim as osim

    osim.Logger.removeFileSink()
    osim.Logger.setLevelString("error")
    model = osim.Model(str(model_path))
    state = model.initSystem()
    coords = [model.getCoordinateSet().get(name) for name in labels["joints"]]
    # Joint rotations are exported in degrees, translations in metres.
    factors = np.array(
        [
            1.0 if name in ("pelvis_tx", "pelvis_ty", "pelvis_tz") else np.pi / 180
            for name in labels["joints"]
        ]
    )
    indices = {name: idx for idx, name in enumerate(labels["joints"])}
    qs = case["Qs_opt"]
    count = len(qs)
    # The upstream optimizer imposes periodic bilateral symmetry. Sample N/2
    # of the reconstructed cycle repeats the opposite leg one exact step later.
    stride = 2 * (qs[count // 2, indices["pelvis_tx"]] - qs[0, indices["pelvis_tx"]])
    duration = stride / SPEED
    if qs.shape != (100, 29) or not 0.7 < duration < 1.6:
        raise ValueError("Unexpected source gait dimensions or duration")

    joints = model.getJointSet()
    bodies = model.getBodySet()
    torso = bodies.get("torso")

    def station(body, offset):
        return xyz(body.findStationLocationInGround(state, offset))

    def joint(name, side):
        # Child frames include the coupled translations of the flexing knee.
        frame = joints.get(f"{name}_{side}").getChildFrame()
        return xyz(frame.getPositionInGround(state))

    force_col = list(labels["GRF"]).index("vertical_r")
    muscle_cols = [list(labels["muscles"]).index(f"glut_med{i}_r") for i in (1, 2, 3)]
    frames = []
    for i, row in enumerate(qs):
        phase = i / count
        for coord, value in zip(coords, row * factors):
            coord.setValue(state, float(value), False)
        model.realizePosition(state)
        sides = []
        for suffix in ("l", "r"):
            hand = bodies.get(f"hand_{suffix}")
            toes = bodies.get(f"toes_{suffix}")
            heel = bodies.get(f"calcn_{suffix}")
            shoulder_name = (
                "shoulder" if joints.contains(f"shoulder_{suffix}") else "acromial"
            )
            sides.append(
                [
                    joint(shoulder_name, suffix),
                    joint("elbow", suffix),
                    joint("radius_hand", suffix),
                    station(hand, hand.getMassCenter()),
                    joint("hip", suffix),
                    joint("knee", suffix),
                    joint("ankle", suffix),
                    xyz(heel.getPositionInGround(state)),
                    station(toes, toes.getMassCenter()),
                ]
            )
        pelvis = 0.5 * (sides[0][4] + sides[1][4])
        shoulder = 0.5 * (sides[0][0] + sides[1][0])
        # The published body treats torso, neck and head as one rigid body.
        origin = xyz(torso.getPositionInGround(state))
        torso_up = station(torso, osim.Vec3(0, 1, 0)) - origin
        points = np.array(
            [
                pelvis,
                station(torso, torso.getMassCenter()),
                shoulder + 0.045 * torso_up,
                shoulder + 0.181 * torso_up,
                *sides[0],
                *sides[1],
            ]
        )
        points[:, 0] -= qs[0, indices["pelvis_tx"]] + phase * stride
        if not np.isfinite(points).all() or points[17, 2] < points[8, 2]:
            raise ValueError(f"Unexpected coordinates in {clip_id} frame {i}")
        frames.append(
            {
                "phase": phase,
                "points": points.round(5).tolist(),
                "hipFlexionDeg": round(float(row[indices["hip_flexion_r"]]), 4),
                "hipAdductionDeg": round(float(row[indices["hip_adduction_r"]]), 4),
                "hipRotationDeg": round(float(row[indices["hip_rotation_r"]]), 4),
                "verticalGrfBw": round(float(case["GRFs_opt"][i, force_col]) / 100, 5),
                "gluteusMediusActivation": round(
                    float(case["Acts_opt"][i, muscle_cols].mean()), 5
                ),
            }
        )
    # The archive omits the duplicate endpoint by design.
    frames.append(dict(frames[0], phase=1.0))
    return {
        "id": clip_id,
        "label": label,
        "kind": "simulated",
        "source": "Falisse et al. · 3dpredictsim",
        "sourceUrl": f"{REPO}/tree/{REV}/Results",
        "paper": "https://doi.org/10.1098/rsif.2019.0402",
        "phaseLabel": "Right heel strike to right heel strike",
        "side": "right",
        "durationS": round(float(duration), 6),
        "speedMps": SPEED,
        "frames": frames,
        "landmarkMethod": "OpenSim forward kinematics of 29 published coordinates. Joint centers use child frames. Pelvis is the hip midpoint. Chest, hands and toes use segment mass centers. Heels use calcaneus origins. Neck and head are torso-fixed display landmarks at 4.5 and 18.1 cm above the shoulder midpoint.",
        "replayMethod": "Right heel strike aligned, symmetric periodic gait. Constant forward progression removed. Ground reaction and equal-weight mean activation of the three right gluteus medius bundles are published simulation outputs.",
    }


def make_clip(path):
    """Export the v2 Case 22 example. Place subject1_v2.osim next to the .mat."""
    path = Path(path)
    source_files(path.parent)
    results = loadmat(path, simplify_cells=True)["Results_all"]
    case = results["Case_22"]
    if not case["stats"]["success"]:
        raise ValueError("Case 22 did not converge")
    return replay(
        case,
        results["colheaders"],
        path.with_name("subject1_v2.osim"),
        "predsim-case22",
        "Reference gait · Case 22",
    )


def matched_clips(directory):
    results = loadmat(directory / "Results_all.mat", simplify_cells=True)["Results_all"]
    for weakness in (0, 50, 75, 90):
        case = results
        for key in COMMON_CASE.format(weakness=weakness).split("/"):
            case = case[key]
        if case["OptSol"] != 1:
            raise ValueError(f"Unconverged hip weakness solution: {weakness}")
        clip = replay(
            case,
            results["colheaders"],
            directory / "subject1.osim",
            f"predsim-hip-{weakness}",
            f"Hip capacity · {100 - weakness}%",
        )
        clip["hipCapacityFraction"] = 1 - weakness / 100
        clip["scenario"] = (
            "Bilateral peak isometric force capacity varied for muscles acting "
            "on hip flexion–extension. The remaining task and objective settings are shared."
        )
        yield clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    source_files(args.source_dir, args.fetch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips = [
        *matched_clips(args.source_dir),
        make_clip(args.source_dir / "Results_all_v2.mat"),
    ]
    for clip in clips:
        out = args.output_dir / (clip["id"] + ".json")
        out.write_text(json.dumps(clip, separators=(",", ":"), allow_nan=False) + "\n")
        print(clip["id"], len(clip["frames"]), clip["durationS"])
