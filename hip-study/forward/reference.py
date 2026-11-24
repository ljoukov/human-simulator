"""Extract the public gait target and independently published forward solutions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REV = "5fa3929fa9ba29d5c057c16b18671a017d8f15c1"
ROOT = Path(__file__).resolve().parent
SOURCE = f"https://github.com/antoinefalisse/3dpredictsim/tree/{REV}"


def strings(items):
    return [str(item).strip() for item in items]


def extract(experimental, solutions, model):
    import xml.etree.ElementTree as ET

    mass_kg = sum(
        float(node.findtext("mass", default="0"))
        for node in ET.parse(model).iter("Body")
    )
    data = loadmat(experimental, simplify_cells=True)["ExperimentalData"]
    poses = data["Q"]["subject1"]["Qs"]
    position = poses["mean"]
    torques = data["Torques"]["subject1"]
    names = strings(poses["colheaders"])
    torque_names = strings(torques["colheaders"])
    grf = data["GRFs"]["subject1"]["mean"][:, 1] / 100
    # Start at right heel strike (archive origin), end immediately before the first
    # low-force frame following midstance. Retain the source's native time scale in phase.
    end = next(i for i in range(int(len(grf) * 0.45), len(grf)) if grf[i] < 0.03)
    block = slice(0, end)
    target = np.column_stack(
        (
            np.linspace(0, 1, end),
            position[block, names.index("hip_flexion")],
            position[block, names.index("hip_adduction")],
            data["TrunkSway"]["subject1"]["mean"][block],
            np.maximum(0, grf[block]),
            torques["mean"][block, torque_names.index("hip_flexion")]
            / (mass_kg * 9.80665),
            -torques["mean"][block, torque_names.index("hip_adduction")]
            / (mass_kg * 9.80665),
        )
    )
    results = loadmat(solutions, simplify_cells=True)["Results_all"]
    names = strings(results["colheaders"]["joints"])
    cases = results["Speed_133"]
    common_keys = [
        "W_MetabolicEnergyRate_500",
        "W_MuscleActivity_2000",
        "W_JointAcceleration_50000",
        "W_PassiveTorque_1000",
        "W_ArmExcitation_1000000",
        "Power_MetabolicEnergyRate_2",
        "InitialGuessType_1",
        "InitialGuessMode_0",
        "InitialGuessCase_0",
    ]
    for key in common_keys:
        cases = cases[key]
    archived = []
    for pct in [0, 50, 75, 90]:
        case = cases[f"WeaknessHipActuators_{pct}"]
        for key in [
            "WeaknessAnklePlantarflexors_0",
            "MetabolicEnergyModel_0",
            "ContactModel_1",
            "Number_MeshIntervals_50",
            "MaximumContractionVelocity_0",
            "CoContraction_0",
        ]:
            case = case[key]
        q = case["Qs_opt"]
        archived.append(
            {
                "hipWeaknessPercent": pct,
                "solverStatus": int(case["OptSol"]),
                "hipAdductionExcursionDeg": round(
                    float(np.ptp(q[:, names.index("hip_adduction_r")])), 3
                ),
                "pelvicListExcursionDeg": round(
                    float(np.ptp(q[:, names.index("pelvis_list")])), 3
                ),
                "hipFlexionExcursionDeg": round(
                    float(np.ptp(q[:, names.index("hip_flexion_r")])), 3
                ),
            }
        )
    return {
        "source": SOURCE,
        "paper": "https://doi.org/10.1098/rsif.2019.0402",
        "fileSha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [experimental, solutions, model]
        },
        "massKg": mass_kg,
        "referenceTrials": 11,
        "referencePeople": 1,
        "columns": [
            "phase",
            "hipFlexionDeg",
            "hipAdductionDeg",
            "trunkLeanDeg",
            "verticalGrfBw",
            "flexionTorqueBwM",
            "abductionTorqueBwM",
        ],
        "frames": target.round(6).tolist(),
        "publishedForwardSensitivity": archived,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experimental", type=Path)
    p.add_argument("solutions", type=Path)
    p.add_argument("model", type=Path)
    args = p.parse_args()
    record = extract(args.experimental, args.solutions, args.model)
    (ROOT / "reference.json").write_text(json.dumps(record, indent=1) + "\n")
