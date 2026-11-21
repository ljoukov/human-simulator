#!/usr/bin/env python3
"""Sample right knee muscle-tendon geometry from the pinned Rajagopal model.

Download MODEL_URL and pass the local .osim file to this script with OpenSim 4.6
installed. The OpenSim body visualization meshes are optional for this calculation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import opensim


MODEL_COMMIT = "d9b05d470b1a481c222372c85b75772faf8f7792"
MODEL_URL = (
    f"https://raw.githubusercontent.com/opensim-org/opensim-models/{MODEL_COMMIT}"
    "/Models/Rajagopal/Rajagopal2016.osim"
)
MODEL_SHA256 = "3f5c5f23e486073f2ad2aa4a4967ffe2fcdd582b1e355512bc54f70c36376bf4"
MUSCLES = {
    "recfem_r": ("Rectus femoris", -1),
    "vasmed_r": ("Vastus medialis", -1),
    "vasint_r": ("Vastus intermedius", -1),
    "vaslat_r": ("Vastus lateralis", -1),
    "bflh_r": ("Biceps femoris, long head", 1),
    "semimem_r": ("Semimembranosus", 1),
    "semiten_r": ("Semitendinosus", 1),
}
MAX_ANGLE_DEG = 120
DERIVATIVE_STEP_RAD = 1e-5
MAX_DERIVATIVE_ERROR_M = 5e-5


def sample_geometry(model_path):
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError(f"Expected the pinned OpenSim model at {MODEL_URL}")
    opensim.Logger.removeFileSink()
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    knee = coordinates.get("knee_angle_r")
    patella = coordinates.get("knee_angle_r_beta")
    muscles = {
        name: model.getMuscles().get(name) for name in MUSCLES
    }

    def pose_knee(angle_rad):
        # The pinned model's patellofemoral coordinate coupler is the identity.
        # Setting both coordinates directly avoids global assembly changing the
        # requested independent angle at full extension.
        knee.setValue(state, angle_rad, False)
        patella.setValue(state, angle_rad, False)
        model.realizePosition(state)

    def lengths():
        return {name: muscle.getLength(state) for name, muscle in muscles.items()}

    samples = []
    max_error = dict.fromkeys(MUSCLES, 0.0)
    max_angle = math.radians(MAX_ANGLE_DEG)
    for angle_deg in range(MAX_ANGLE_DEG + 1):
        angle = math.radians(angle_deg)
        pose_knee(angle)
        length = lengths()
        moment_arm = {
            name: muscle.computeMomentArm(state, knee) for name, muscle in muscles.items()
        }
        # Use one-sided differences at the endpoints of the model's range.
        low = max(0.0, angle - DERIVATIVE_STEP_RAD)
        high = min(max_angle, angle + DERIVATIVE_STEP_RAD)
        pose_knee(low)
        length_low = lengths()
        pose_knee(high)
        length_high = lengths()
        for name, (_, expected_sign) in MUSCLES.items():
            numerical_moment_arm = -(length_high[name] - length_low[name]) / (high - low)
            values = (length[name], moment_arm[name], numerical_moment_arm)
            if not all(map(math.isfinite, values)) or length[name] <= 0:
                raise ValueError(f"Invalid geometry for {name} at {angle_deg} degrees")
            if moment_arm[name] * expected_sign <= 0:
                raise ValueError(f"Unexpected moment arm sign for {name} at {angle_deg} degrees")
            error = abs(moment_arm[name] - numerical_moment_arm)
            if error > MAX_DERIVATIVE_ERROR_M:
                raise ValueError(
                    f"Moment arm derivative mismatch for {name} at {angle_deg} degrees: {error} m"
                )
            max_error[name] = max(max_error[name], error)
        samples.append({
            "angleDeg": angle_deg,
            "lengthM": {name: round(value, 8) for name, value in length.items()},
            "momentArmM": {name: round(value, 8) for name, value in moment_arm.items()},
        })

    fixed_coordinates = (
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
        "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r",
    )
    return {
        "metadata": {
            "model": "Rajagopal2016",
            "source": MODEL_URL,
            "sourceCommit": MODEL_COMMIT,
            "sourceSha256": MODEL_SHA256,
            "reference": "https://doi.org/10.1109/TBME.2016.2586891",
            "openSimVersion": opensim.GetVersion(),
            "coordinate": "knee_angle_r",
            "coupledCoordinate": "knee_angle_r_beta",
            "fixedCoordinatesDeg": {
                name: math.degrees(coordinates.get(name).getValue(state))
                for name in fixed_coordinates
            },
            "otherCoordinates": "All other coordinates held at model defaults.",
            "momentArmConvention": "Positive is flexion, negative is extension. Equals minus the muscle-tendon path length derivative with respect to knee flexion in radians.",
            "scope": "Reference-model muscle-tendon path geometry for a fixed hip and ankle.",
            "validation": {
                "derivativeStepRad": DERIVATIVE_STEP_RAD,
                "maxAllowedDerivativeErrorM": MAX_DERIVATIVE_ERROR_M,
                "maxObservedDerivativeErrorM": {
                    name: round(error, 10) for name, error in max_error.items()
                },
            },
        },
        "muscles": {name: label for name, (label, _) in MUSCLES.items()},
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Pinned Rajagopal2016.osim")
    default_output = Path(__file__).resolve().parents[1] / "data" / "leg-mechanics.json"
    parser.add_argument("output", type=Path, nargs="?", default=default_output)
    args = parser.parse_args()
    data = sample_geometry(args.model)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
