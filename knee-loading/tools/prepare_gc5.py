"""Prepare a normal-walking sensitivity cohort from the Modenese et al. archive.

Run with Python 3 and numpy, scipy, and OpenSim 4.6 installed:
    python3 prepare_gc5.py PATH_TO_EXTRACTED_ARCHIVE OUTPUT_JSON
"""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import opensim
from scipy.io import loadmat


MASS_KG = 75.0
HEIGHT_M = 1.80
BODY_WEIGHT_N = MASS_KG * 9.80665
ARCHIVE_COMMIT = "ff2699b6dc5fbcfecd95d6115438f08daf3cc06d"
SOURCE_URL = f"https://github.com/modenaxe/femoral-anteversion-paper/tree/{ARCHIVE_COMMIT}"
MOTION_INPUTS = Path("4_run_simulations/Common_inputs")
LOAD_ASSIGNMENTS = Path("6_muscle_length_variations/ScaledMuscleParams_sims/GC5_femAntev40Deg_scaled/External Loads")


def read_storage(path):
    with path.open() as source:
        for line in source:
            if line.strip().lower() == "endheader":
                return np.genfromtxt(source, names=True)
    raise ValueError(f"No OpenSim header terminator in {path}")


def resample(table, labels, times):
    return np.column_stack([np.interp(times, table["time"], table[label]) for label in labels])


def vector(value):
    return np.array([value.get(i) for i in range(3)])


def knee_wrenches(model, state, motion, times, forces, points, torques):
    labels = [name for name in motion.dtype.names if name != "time"]
    coordinates = [model.getCoordinateSet().get(name) for name in labels]
    positions = resample(motion, labels, times)
    angle_columns = [index for index, name in enumerate(labels) if name not in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
    positions[:, angle_columns] = np.deg2rad(positions[:, angle_columns])
    knee_center = model.getJointSet().get("walker_knee_l").getChildFrame()
    tibia = model.getBodySet().get("tibia_l")
    anterior = opensim.Vec3(1, 0, 0)
    rightward = opensim.Vec3(0, 0, 1)
    moments = []
    for row, force, point, torque in zip(positions, forces, points, torques, strict=True):
        for coordinate, position in zip(coordinates, row, strict=True):
            coordinate.setValue(state, float(position), False)
        model.realizePosition(state)
        moment = np.cross(point - vector(knee_center.getPositionInGround(state)), force) + torque
        # In the left shank frame, a medial vertical load and a posterior
        # vertical load both have negative projected moments (about X and Z respectively).
        adduction = -moment @ vector(tibia.expressVectorInGround(state, anterior))
        flexion = -moment @ vector(tibia.expressVectorInGround(state, rightward))
        moments.append((adduction / MASS_KG, flexion / MASS_KG))
    return moments


def prepare_trial(root, model, state, trial_id, implant, role):
    motion = read_storage(root / MOTION_INPUTS / "IK" / f"{trial_id}_kinematics.mot")
    plates = read_storage(root / MOTION_INPUTS / "External Loads" / f"{trial_id}_kinetics.mot")
    assignments = ElementTree.parse(root / LOAD_ASSIGNMENTS / f"{trial_id}_External_Forces.xml")
    left_foot = next(force for force in assignments.findall(".//ExternalForce")
                     if force.findtext("applied_to_body") == "calcn_l")
    force_label = left_foot.findtext("force_identifier")
    point_label = left_foot.findtext("point_identifier")
    torque_label = left_foot.findtext("torque_identifier")
    contact = np.flatnonzero(plates[force_label + "y"] > 20)
    if not len(contact) or not np.all(np.diff(contact) == 1):
        raise ValueError(f"Expected one continuous left stance in {trial_id}")
    start, end = plates["time"][contact[[0, -1]]]
    if start < motion["time"][0] or end > motion["time"][-1]:
        raise ValueError(f"Missing motion during stance in {trial_id}")
    times = np.linspace(start, end, 101)
    forces = resample(plates, [force_label + axis for axis in "xyz"], times)
    points = resample(plates, [point_label + axis for axis in "xyz"], times)
    torques = resample(plates, [torque_label + "_" + axis for axis in "xyz"], times)
    moments = knee_wrenches(model, state, motion, times, forces, points, torques)
    knee_angles = np.interp(times, motion["time"], motion["knee_angle_l"])
    # The supplement uses a full gait cycle, while the pipeline scores stance.
    cycle_times = np.linspace(motion["time"][0], motion["time"][-1], len(implant))
    medial, lateral = [np.interp(times, cycle_times, channel) / BODY_WEIGHT_N for channel in implant.T]

    def sample(value):
        return round(float(value), 6)

    frames = [
        {
            "stancePct": pct,
            "kamNmKg": sample(kam),
            "kfmNmKg": sample(kfm),
            "grfBw": sample(forces[pct, 1] / BODY_WEIGHT_N),
            "kneeFlexionDeg": sample(knee_angles[pct]),
            "medialBw": sample(medial[pct]) if medial[pct] >= 0 else None,
            "lateralBw": sample(lateral[pct]) if lateral[pct] >= 0 else None,
            "totalBw": sample(medial[pct] + lateral[pct]),
        }
        for pct, (kam, kfm) in enumerate(moments)
    ]
    return {
        "id": trial_id,
        "subject": "GC5-PS",
        "condition": "normal",
        "role": role,
        "massKg": MASS_KG,
        "heightM": HEIGHT_M,
        "stanceSeconds": sample(end - start),
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Extracted Modenese et al. reproducibility archive")
    parser.add_argument("output", type=Path, help="Output JSON path")
    args = parser.parse_args()

    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(args.archive / "2_make_knee_weaker/GC5.osim"))
    state = model.initSystem()
    implant = loadmat(args.archive / "5_analyze_results/eTibia_data.mat", simplify_cells=True)
    walking = implant["eTibia_data"]["GC5"]["ngait_og"]
    columns = [list(walking["colheaders"]).index(name) for name in ("F_medial", "F_lateral")]
    order = sorted(range(len(walking["trial_names"])), key=lambda index: int(
        Path(walking["trial_names"][index]).stem.rsplit("ss", 1)[1]))
    trials = [
        prepare_trial(args.archive, model, state, Path(walking["trial_names"][index]).stem,
                      walking["data"][:, columns, index], "calibration" if rank < 3 else "evaluation")
        for rank, index in enumerate(order)
    ]
    dataset = {
        "source": {
            "title": "Grand Challenge fifth competition, five normal walking trials",
            "url": SOURCE_URL,
            "citation": "Modenese, Barzan & Carty (2021), doi:10.1016/j.gaitpost.2021.06.014; "
                        "Grand Challenge to Predict In Vivo Knee Loads, Fregly et al. (2012), doi:10.1002/jor.22023.",
            "license": "GPL-3.0 derived dataset from the Modenese et al. reproducibility repository. "
                       "See GC5-LICENSE.txt for attribution and original data origin.",
            "kind": "trial",
            "recordedSide": "left",
            "notes": [
                "One left eTibia recipient, five repetitions of normal walking only.",
                "Calibration uses the three lowest-numbered trials (ss1, ss3, ss8). "
                "The other two trials (ss9 and ss11) are held out.",
                "Inputs are published force-plate data and OpenSim fitted poses. "
                "Knee moments include the ground reaction wrench about the fitted left knee projected on tibial axes, "
                "without segment inertia or gravity. Vertical ground reaction approximates external axial compression.",
                "The supplement has whole-cycle implant observations without sample timestamps. "
                "Aligned by normalised phase within the matching motion trial, then cropped at 20 N vertical force. "
                "The resulting phase match is approximate.",
                "Medial and lateral are published implant-derived compartment forces. "
                "Negative compartment estimates are excluded from compartment scoring. "
                "Total axial force is the sum of both estimates before exclusion, distinct from the published 3-D magnitude.",
            ],
        },
        "trials": trials,
    }
    args.output.write_text(json.dumps(dataset, allow_nan=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
