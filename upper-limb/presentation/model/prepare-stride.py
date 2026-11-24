"""Extract the periodic walking reference from Falisse et al.'s published recording.

Run python prepare-stride.py ExperimentalData.mat with NumPy and SciPy.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parents[2] / "hip-study/forward/reference.json"
data_path = Path(sys.argv[1])
reference = json.loads(REFERENCE.read_text())
digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
if digest != reference["fileSha256"]["ExperimentalData.mat"]:
    raise ValueError("Use the pinned ExperimentalData.mat from Falisse et al.")
data = loadmat(data_path, simplify_cells=True)["ExperimentalData"]
poses = data["Q"]["subject1"]["Qs"]
torques = data["Torques"]["subject1"]


def mean_channel(record, name):
    return record["mean"][:, list(record["colheaders"]).index(name)]


fields = {
    "pelvisDropDeg": -mean_channel(poses, "pelvis_list"),
    "pelvisTiltDeg": mean_channel(poses, "pelvis_tilt"),
    "trunkLeanDeg": data["TrunkSway"]["subject1"]["mean"],
    "hipAdductionDeg": mean_channel(poses, "hip_adduction"),
    "hipFlexionDeg": mean_channel(poses, "hip_flexion"),
    "kneeFlexionDeg": -mean_channel(poses, "knee_angle"),
    "ankleDorsiflexionDeg": mean_channel(poses, "ankle_angle"),
    "shoulderFlexionDeg": mean_channel(poses, "arm_flex"),
    "shoulderAdductionDeg": mean_channel(poses, "arm_add"),
    "elbowFlexionDeg": mean_channel(poses, "elbow_flex"),
    "groundForceBw": np.maximum(data["GRFs"]["subject1"]["mean"][:, 1] / 100, 0),
    "hipMomentBwM": -mean_channel(torques, "hip_adduction") / (reference["massKg"] * 9.80665),
}
source = {
    "url": reference["source"],
    "sha256": digest,
    "subject": "subject1, eleven trials",
    "paper": reference["paper"],
    "referenceMassKg": round(reference["massKg"], 6),
    "strideLengthM": float(data["StrideLength"]["subject1"]["mean"]),
    "referenceSpeedMps": 1.33,
    "phasing": (
        "Positive flexion, adduction, rightward trunk lean and left pelvis drop. The left limb "
        "is the right template shifted half a cycle. All 100 source samples form a periodic motion "
        "template, with linear interpolation back to the first sample over the final 1% of stride."
    ),
    "channels": {name: values.round(6).tolist() for name, values in fields.items()},
}
(HERE / "stride.json").write_text(json.dumps(source, separators=(",", ":")) + "\n")
