"""Generate ../data/arm26.json from the pinned Arm26 XML using OpenSim 4.6.

Install opensim==4.6 in the Python environment and download MODEL_URL below.
Run: python sample_arm26.py /path/to/arm26.osim
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import opensim


COMMIT = "d9b05d470b1a481c222372c85b75772faf8f7792"
MODEL_URL = (
    "https://raw.githubusercontent.com/opensim-org/opensim-models/"
    f"{COMMIT}/Models/Arm26/arm26.osim"
)
MODEL_SHA256 = "e2224d0044eb393b05d64926c3fa1682c451a9adc7f510e5517ef9958d3d41b9"
MUSCLES = ("BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed")
DERIVATIVE_STEP_RAD = 1e-5
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "arm26.json"


def sample(model_path):
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError(f"Unexpected model digest. Use the pinned file from {MODEL_URL}")
    if opensim.GetVersion().split("-", 1)[0] != "4.6":
        raise ValueError(f"Expected OpenSim 4.6, found {opensim.GetVersion()}")

    # This sweep needs muscle paths and model wrap surfaces only. Display meshes
    # are optional and absent from the single-file model.
    opensim.Logger.removeFileSink()
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    model.getCoordinateSet().get("r_shoulder_elev").setValue(state, 0.0, False)
    elbow = model.getCoordinateSet().get("r_elbow_flex")
    muscles = {name: model.getMuscles().get(name) for name in MUSCLES}

    def lengths_at(angle_rad):
        elbow.setValue(state, angle_rad, False)
        model.realizePosition(state)
        return {name: muscle.getLength(state) for name, muscle in muscles.items()}

    max_error = {name: 0.0 for name in muscles}
    samples = []
    for angle_deg in range(126):
        angle = math.radians(angle_deg)
        low = max(elbow.getRangeMin(), angle - DERIVATIVE_STEP_RAD)
        high = angle + DERIVATIVE_STEP_RAD
        before = lengths_at(low)
        after = lengths_at(high)
        lengths = lengths_at(angle)
        arms = {}
        for name, muscle in muscles.items():
            arm = muscle.computeMomentArm(state, elbow)
            finite_diff = -(after[name] - before[name]) / (high - low)
            if not all(math.isfinite(x) for x in (lengths[name], finite_diff, arm)):
                raise ValueError(f"Non-finite geometry for {name} at {angle_deg} degrees")
            if lengths[name] <= 0 or arm * finite_diff <= 0:
                raise ValueError(f"Invalid length or moment arm sign for {name} at {angle_deg} degrees")
            max_error[name] = max(max_error[name], abs(arm - finite_diff))
            arms[name] = round(arm, 9)
        samples.append({
            "angleDeg": angle_deg,
            "lengthM": {name: round(length, 9) for name, length in lengths.items()},
            "momentArmM": arms,
        })

    return {
        "metadata": {
            "model": "OpenSim Arm26",
            "opensimVersion": opensim.GetVersion(),
            "source": MODEL_URL,
            "sourceCommit": COMMIT,
            "sourceSha256": MODEL_SHA256,
            "license": "CC BY 3.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/3.0/",
            "attribution": "The OpenSim Development Team (Jeff Reinbolt, Ajay Seth, Ayman Habib and Sam Hamner), adapted from Kate Holzbaur's upper-extremity model.",
            "derivation": "Sampled the supplied model at each integer degree of right elbow flexion with right shoulder elevation fixed at 0 degrees.",
            "scope": "Arm26 is a simplified, two-coordinate right arm with six muscle actuators and a fixed forearm rotation. Values describe generic model geometry.",
            "lengthDefinition": "Whole musculotendon path length, in meters.",
            "momentArmDefinition": "OpenSim signed elbow moment arm in meters, positive for flexion and negative for extension.",
            "validation": {
                "method": "Compared computeMomentArm with the negative finite difference of path length with respect to elbow angle in radians. Used a forward difference at 0 degrees and central differences elsewhere.",
                "stepRad": DERIVATIVE_STEP_RAD,
                "maxAbsDifferenceM": {name: round(value, 9) for name, value in max_error.items()},
            },
        },
        "muscles": list(MUSCLES),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = sample(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
