#!/usr/bin/env python3
"""Replay the released Harba et al. OpenSim IK as phase-aligned world poses."""

from argparse import ArgumentParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import opensim

from prepare_data import (
    COMMIT, CONTACT_THRESHOLD_N, GRAVITY, MASS_KG, RAW, RECORDINGS, REPOSITORY, SOURCE_PATH,
    STANCE_POINTS, contact_intervals, sample, source_table,
)

MODEL_PATH = (
    "MuscleTendonLengthParameterizationReduction/"
    "JW All Body Model-scaled_forvisualization_leftright_v42.osim"
)
SEGMENTS = (
    "pelvis", "torso",
    "femur", "tibia", "patella", "talus", "calcaneus", "toes",
    "femur_l", "tibia_l", "patella_l", "talus_l", "calcaneus_l", "toes_l",
    "humerus_r", "ulna_r", "radius_r", "hand_r",
    "humerus_l", "ulna_l", "radius_l", "hand_l",
)
JOINT_FRAMES = {
    "lumbar": ("back", "parent"),
    "rightHip": ("hip", "parent"),
    "leftHip": ("hip_l", "parent"),
    # Both landmarks follow the femoral frame. The implanted right knee
    # uses a separately welded femoral component at that location.
    "rightKnee": ("femoral_component_weld", "parent"),
    "leftKnee": ("knee_l", "child"),
    "rightAnkle": ("ankle", "parent"),
    "leftAnkle": ("ankle_l", "parent"),
    "rightShoulder": ("acromial_r", "parent"),
    "leftShoulder": ("acromial_l", "parent"),
    "rightElbow": ("elbow_r", "parent"),
    "leftElbow": ("elbow_l", "parent"),
    "rightWrist": ("radius_hand_r", "parent"),
    "leftWrist": ("radius_hand_l", "parent"),
}


def simtk_values(vector, size):
    return np.array([vector.get(index) for index in range(size)])


def vector(values, decimals=6):
    return [round(float(value), decimals) for value in values]


def landmarks(model):
    result = {}
    for name, (joint, side) in JOINT_FRAMES.items():
        target = model.getJointSet().get(joint)
        frame = target.getParentFrame() if side == "parent" else target.getChildFrame()
        result[name] = {
            "segment": frame.findBaseFrame().getName(),
            "offsetM": vector(simtk_values(frame.findTransformInBaseFrame().p(), 3)),
        }
    return result


def right_contacts(grf, plates):
    return sorted(
        (start, end, plate)
        for plate in plates
        for start, end in contact_intervals(grf["time"], grf[f"grforce{plate}_fy"])
    )


def ground_reactions(grf, time, right_plates, origin):
    def read_plate(plate):
        columns = ["px", "py", "pz", "fx", "fy", "fz"]
        data = np.column_stack([
            sample(grf["time"], grf[f"grforce{plate}_{key}"], time, f"force plate {plate} {key}")
            for key in columns
        ])
        data[:, :3] -= origin
        return data

    right = [read_plate(plate) for plate in right_plates]
    left = read_plate(1)
    weight_n = MASS_KG * GRAVITY

    def packed(row):
        if row[4] < CONTACT_THRESHOLD_N:
            return None
        return vector(row[:3], 5) + vector(row[3:] / weight_n, 5)

    return [
        {
            "right": packed(max((plate[index] for plate in right), key=lambda row: row[4])),
            "left": packed(left[index]),
        }
        for index in range(len(time))
    ]


def sample_coordinates(kinematics, time, coordinates):
    sampled = np.column_stack([
        sample(kinematics["time"], kinematics[coord.getName()], time, coord.getName())
        for coord in coordinates
    ])
    angular = [i for i, coord in enumerate(coordinates) if coord.getMotionType() == opensim.Coordinate.Rotational]
    sampled[:, angular] = np.radians(sampled[:, angular])
    return sampled


def replay(model, state, coordinates, segments, samples, origin):
    poses = []
    previous = [None] * len(segments)
    for values in samples:
        for coordinate, value in zip(coordinates, values, strict=True):
            coordinate.setValue(state, float(value), False)
        model.realizePosition(state)

        frame = []
        for index, segment in enumerate(segments):
            transform = segment.getTransformInGround(state)
            position = simtk_values(transform.p(), 3) - origin
            quaternion = simtk_values(transform.R().convertRotationToQuaternion(), 4)[[1, 2, 3, 0]]
            if previous[index] is not None and previous[index] @ quaternion < 0:
                quaternion *= -1
            previous[index] = quaternion
            frame.extend(vector(position, 5))
            frame.extend(vector(quaternion))
        poses.append(frame)
    return poses


def make_clips(model, state, directory, canonical):
    coordinates = [model.getCoordinateSet().get(i) for i in range(model.getCoordinateSet().getSize())]
    segments = [model.getBodySet().get(name) for name in SEGMENTS]
    for coordinate in coordinates:
        if coordinate.getLocked(state):
            coordinate.setLocked(state, False)

    clips = []
    for recording, condition, plates, selection in RECORDINGS:
        kinematics = source_table(directory, "IK", recording, "Qsall")
        grf = source_table(directory, "GRF", recording, "GRFall")
        contacts = right_contacts(grf, plates)
        for index, (start, stance_end, plate) in enumerate(contacts):
            if selection and not selection[0] <= start < stance_end <= selection[1]:
                continue

            stance_index = plates.index(plate) + 1
            clip_id = f"{recording}-stance-{stance_index}"
            reference = canonical[clip_id]
            if (
                reference["condition"] != condition or reference["forcePlate"] != plate
                or abs(reference["stanceStartSeconds"] - start) > 1e-5
                or abs(reference["stanceSeconds"] - (stance_end - start)) > 1e-5
            ):
                raise ValueError(f"{clip_id} does not align with the recorded stance dataset")

            following_strike = contacts[index + 1][0] if index + 1 < len(contacts) else None
            complete_stride = following_strike is not None and following_strike <= kinematics["time"][-1]
            end = following_strike if complete_stride else stance_end
            time = np.linspace(start, end, STANCE_POINTS)
            origin = np.array([
                sample(kinematics["time"], kinematics["pelvis_tx"], time[:1], "pelvis x")[0], 0.0,
                sample(kinematics["time"], kinematics["pelvis_tz"], time[:1], "pelvis z")[0],
            ])
            samples = sample_coordinates(kinematics, time, coordinates)
            pelvis_vertical = sample(kinematics["time"], kinematics["pelvis_ty"], time, "pelvis height")
            poses = replay(model, state, coordinates, segments, samples, origin)
            _, qy, qz, _ = poses[0][3:7]
            pelvis_forward_x = 1 - 2 * (qy * qy + qz * qz)
            clips.append({
                "id": clip_id,
                "recording": recording,
                "condition": condition,
                "phase": "stride" if complete_stride else "stance",
                "heading": 1 if pelvis_forward_x >= 0 else -1,
                "durationSeconds": round(end - start, 6),
                "stanceEndPct": round(100 * (stance_end - start) / (end - start), 6),
                "frames": poses,
                "ground": ground_reactions(grf, time, plates, origin),
                "signals": {
                    "rightKneeFlexionDeg": vector(-sample(kinematics["time"], kinematics["knee_flexion"], time, "right knee flexion")),
                    "leftKneeFlexionDeg": vector(-sample(kinematics["time"], kinematics["knee_flexion_l"], time, "left knee flexion")),
                    "pelvisVerticalDisplacementM": vector(pelvis_vertical - pelvis_vertical[0], 5),
                },
            })

    if set(canonical) != {clip["id"] for clip in clips}:
        raise ValueError("Full-body clips and force-study stances do not cover the same IDs")
    return clips


def build_dataset(model_path, source_directory, gait):
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    return {
        "source": {
            "title": "Harba, Badia & Serrancolí · Grand Challenge walking recordings",
            "url": f"{REPOSITORY}/tree/{COMMIT}/{SOURCE_PATH}",
            "model": f"{REPOSITORY}/blob/{COMMIT}/{quote(MODEL_PATH)}",
            "license": "MIT · see HARBA-LICENSE.txt",
            "participant": "GC4-JW",
            "coordinateConvention": "OpenSim laboratory X/Y/Z with +Y up. Overground playback faces +X; the treadmill participant faces −X. The clip heading is the sign of the first pelvis forward axis projected on laboratory X. Each clip subtracts its own initial pelvis X/Z position from the motions and center of pressure. Quaternions are [x, y, z, w].",
            "frameConvention": "Flat 7 values per segment in segmentOrder, equally spaced in phase. A stride spans consecutive right-foot 20 N crossings and a stance-only clip ends at right toe-off. Force study waveforms cover the interval from initial contact to stanceEndPct.",
            "groundConvention": "A known plate/belt contact returns [CoP x, y, z in meters, force x, y, z in body weights] in world axes. Right uses overground plates 2/3 or treadmill belt 2; left uses plate/belt 1. Null means no measured contact above 20 N, including parts of overground recordings outside the plates.",
            "notes": [
                "Positions and orientations come from the released whole-body OpenSim fit to recorded movement. The released model includes body geometry for a rigid torso and hands, including reference skull and fingers where rendered. Head and individual finger movement have no independent recorded channels.",
                "The source knee joint has secondary fitted or constrained coordinates. Knee angles follow the native model here; the detailed tissue inset uses its own atlas rig and sagittal angle only.",
                "Overground second stances lack a following measured right initial contact. Those clips preserve observed stance only. Whole cycle clips start from the first right stance available for each gait condition.",
            ],
        },
        "segmentOrder": list(SEGMENTS),
        "landmarks": landmarks(model),
        "clips": make_clips(model, state, source_directory, {trial["id"]: trial for trial in gait["trials"]}),
    }


def main():
    setup = ArgumentParser(description=__doc__)
    data_dir = Path(__file__).resolve().parents[1] / "data"
    setup.add_argument("--source-directory", type=Path, help=f"Local copy of {SOURCE_PATH}")
    setup.add_argument("--model", type=Path, help="Local copy of the pinned Harba OpenSim visualization model")
    setup.add_argument("--gait", type=Path, default=data_dir / "gait.json")
    setup.add_argument("--output", type=Path, default=data_dir / "whole-body-motion.json")
    args = setup.parse_args()

    gait = json.loads(args.gait.read_text(encoding="utf-8"))
    with TemporaryDirectory() as temp:
        model_path = args.model
        if model_path is None:
            model_path = Path(temp) / "whole-body.osim"
            with urlopen(f"{RAW}/{quote(MODEL_PATH)}", timeout=30) as response:
                model_path.write_bytes(response.read())
        dataset = build_dataset(model_path, args.source_directory, gait)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(dataset['clips'])} recorded motion clips to {args.output}")


if __name__ == "__main__":
    main()
