"""Rebuild the presentation's effective ITB path from the pinned Davis model.

python prepare_itb_geometry.py /path/to/Davis2026.osim
Requires OpenSim 4.6 and NumPy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import opensim


OUTPUT = Path(__file__).with_name("itb-geometry.json")
GEOMETRY = Path(__file__).resolve().parents[3] / "hip-study" / "data" / "geometry.json"
COORDINATES = ["hip_flexion_r", "hip_adduction_r", "knee_angle_r"]
RANGES = np.array([[-25, 50], [-18, 22], [0, 85]])
MID = RANGES.mean(axis=1)
SCALE = (RANGES[:, 1] - RANGES[:, 0]) / 2
HIP_POWERS = [(i, j, 0) for i in range(5) for j in range(5 - i)]
POWERS = HIP_POWERS + [(0, 0, k) for k in range(1, 5)]


def features(x):
    return np.column_stack([np.prod(x ** power, axis=1) for power in POWERS])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    source_geometry = json.loads(GEOMETRY.read_text())
    source = source_geometry["sources"]["Davis2026"]
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == source["sha256"]
    opensim.Logger.removeFileSink()
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(args.model))
    state = model.initSystem()
    coords = [model.getCoordinateSet().get(key) for key in COORDINATES]
    names = list(source_geometry["itb"]["tractWeights"])
    tracts = [model.getMuscles().get(name) for name in names]
    weights = np.array([t.getMaxIsometricForce() for t in tracts])
    weights /= weights.sum()

    def length(angles):
        for coordinate, angle in zip(coords, angles):
            coordinate.setValue(state, float(np.deg2rad(angle)), False)
        model.realizePosition(state)
        return float(weights @ [t.getLength(state) for t in tracts])

    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in (14, 11, 13)],
                                indexing="ij"), axis=-1).reshape(-1, 3)
    lengths = np.array([length(row) for row in grid * SCALE + MID])
    coefficients = np.linalg.lstsq(features(grid), lengths, rcond=None)[0]
    # Independent poses, with separate central differences along each physical coordinate.
    rng = np.random.default_rng(20261122)
    x = rng.uniform(-0.998, 0.998, (512, 3))
    poses = x * SCALE + MID
    errors_mm = 1000 * abs(features(x) @ coefficients - [length(row) for row in poses])
    gradient_errors_mm = []
    step = 0.03
    for axis in np.eye(3):
        physical = np.array([length(row + axis * step) - length(row - axis * step)
                             for row in poses])
        fitted = (features(x + axis * step / SCALE) - features(x - axis * step / SCALE)) @ coefficients
        gradient_errors_mm.append(abs(fitted - physical) * 1000 / (2 * np.deg2rad(step)))
    gradient_errors_mm = np.asarray(gradient_errors_mm)
    if errors_mm.max() > 0.1 or gradient_errors_mm.max() > 1.0:
        raise ValueError("Polynomial exceeds the path length or gradient error bound.")
    result = {
        "source": source, "reference": source_geometry["itb"]["reference"],
        "openSimVersion": opensim.GetVersionAndDate().split(",")[0],
        "definition": "Effective length is the maximum-isometric-force-weighted mean of three right muscle-tendon paths routed through the iliotibial band. Other joint coordinates are held at their model defaults.",
        "names": names, "weights": weights.tolist(), "coordinates": COORDINATES,
        "coordinateRangesDeg": RANGES.tolist(), "centersDeg": MID.tolist(), "halfRangesDeg": SCALE.tolist(),
        "polynomial": "Normalize each angle by subtracting its center and dividing by its half range. Sum hipCoefficientsM[i][j] times flexion^i times adduction^j, plus kneeCoefficientsM[k] times knee flexion^k.",
        "hipCoefficientsM": [[float(coefficients[POWERS.index((i,j,0))]) for j in range(5-i)] for i in range(5)],
        "kneeCoefficientsM": [0.0, *coefficients[len(HIP_POWERS):].tolist()],
        "directNeutralLengthM": length([0, 0, 0]),
        "validation": {"gridSamples": len(grid), "independentPoses": len(poses), "seed": 20261122,
                       "lengthMaeMm": float(errors_mm.mean()), "lengthMaxErrorMm": float(errors_mm.max()),
                       "gradientMaeMmPerRad": gradient_errors_mm.mean(axis=1).tolist(),
                       "gradientMaxErrorMmPerRad": gradient_errors_mm.max(axis=1).tolist()},
        "stiffnessNPerMm": 17, "stiffnessReference": "https://pubmed.ncbi.nlm.nih.gov/15378277/",
        "interpretation": "The cable uses an assumed slack length and a cadaver-derived whole-tract stiffness scale. The three musculotendon paths also include muscle excursion. Numerical agreement is with the source geometry and does not measure in-vivo tissue force.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
