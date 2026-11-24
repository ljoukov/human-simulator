#!/usr/bin/env python3
"""Build the hip abductor geometry table from pinned OpenSim models.

With OpenSim 4.6 and NumPy installed, run:
  python hip-study/tools/prepare_geometry.py --cache /tmp/hip-osim-reference

Downloads missing .osim model definitions. Rendering meshes are optional.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import opensim

OPENSIM_COMMIT = "d9b05d470b1a481c222372c85b75772faf8f7792"
STANSFIELD_COMMIT = "d215a656137dcc29002851e448679069477c9a63"
OFFICIAL = f"https://raw.githubusercontent.com/opensim-org/opensim-models/{OPENSIM_COMMIT}/Models/Rajagopal"
WORKED = (
    "https://raw.githubusercontent.com/katya-stanzy/thin-plate-spline_personalised_muscoloskeletal_model/"
    f"{STANSFIELD_COMMIT}/folder/worked_example/final_results"
)
MODELS = {
    "MRIInformed2023": (f"{OFFICIAL}/RajagopalLaiUhlrich2023.osim",
                         "8f30d0b64750b87eb7f705907862590535212b4afd7e919faa3fd7d1683d22ec"),
    "Rajagopal2016": (f"{OFFICIAL}/Rajagopal2016.osim",
                      "3f5c5f23e486073f2ad2aa4a4967ffe2fcdd582b1e355512bc54f70c36376bf4"),
    "genericScaledExample": (f"{WORKED}/generic_scaled/scaled_model.osim",
                              "31f6dfee40c1ac5fe302b53ae2276ce0b1e198d96e97fb979868afb05fb43bcf"),
    "MRIPersonalizedExample": (f"{WORKED}/personalized/tps_optimised.osim",
                                "af1332c908ed3460f194b86ccaf0ab03a8256d2dc21e4639342630864aed03ed"),
}
MUSCLES = [f"{group}{segment}_r" for group in ("glmed", "glmin") for segment in (1, 2, 3)]
ITB_SOURCE = (
    "https://raw.githubusercontent.com/johnjdavisiv/it-band-model/"
    "2df50ea795bc2eddb3405854bddaa9e7427c77e8/models/Davis2026.osim",
    "c6c6cf5a3e4ee3d2e04a6aa13d755d21f5412e1c6fec5cfe7977618a887ef386",
)
ITB_TRACTS = ("glmax12_ITB_r", "glmax34_ITB_r", "tfl12_ITB_r")
COORDINATES = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r"]
AXES = [np.arange(-30, 76, 5), np.arange(-25, 31, 5), np.arange(-30, 31, 10)]
REFERENCE = Path(__file__).resolve().parents[1] / "data" / "geometry.json"


def model_file(cache, url, digest):
    path = cache / url.rsplit("/", 1)[1]
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=cache) as temp:
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                            "--retry", "2", "--max-time", "90", url, "-o", temp.name], check=True)
            content = Path(temp.name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Invalid downloaded model digest for {url}")
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Invalid model digest for {path}")
    return path


class Geometry:
    def __init__(self, path):
        self.model = opensim.Model(str(path))
        self.state = self.model.initSystem()
        self.coordinates = [self.model.getCoordinateSet().get(key) for key in COORDINATES]
        self.muscles = [self.model.getMuscles().get(key) for key in MUSCLES]
        self.maximum_force = np.array([muscle.getMaxIsometricForce() for muscle in self.muscles])
        self.weights = self.maximum_force / self.maximum_force.sum()

    def pose(self, radians):
        for coordinate, angle in zip(self.coordinates, radians):
            coordinate.setValue(self.state, float(angle), False)
        self.model.realizePosition(self.state)

    def lengths(self):
        return np.array([muscle.getLength(self.state) for muscle in self.muscles])

    def arms(self):
        return np.array([[muscle.computeMomentArm(self.state, coordinate)
                          for coordinate in self.coordinates] for muscle in self.muscles])

    def effective_arm(self):
        return -sum(weight * muscle.computeMomentArm(self.state, self.coordinates[1])
                    for weight, muscle in zip(self.weights, self.muscles))


def packed(array):
    return np.asarray(array).round(8).tolist()


def distribution(values):
    absolute = np.abs(values)
    return dict(maeM=float(absolute.mean()), p95M=float(np.percentile(absolute, 95)),
                maxM=float(absolute.max()))


def itb_reaction(geometry, radians):
    """Inward projection of the vector sum of two unit tensions at each guide."""
    geometry.pose(radians)
    femur = geometry.model.getBodySet().get("femur_r")
    ground = geometry.model.getGround()
    medial_components = []
    for name in ITB_TRACTS:
        path = geometry.model.getMuscles().get(name).getGeometryPath().getCurrentPath(geometry.state)
        # getCurrentPath includes active wrap points if they intervene between
        # a named via point and the next fixed attachment.
        middle = next(i for i in range(path.getSize()) if path.get(i).getName().endswith("-trochVP"))
        locations = []
        for i in (middle-1, middle, middle+1):
            point = path.get(i).getLocationInGround(geometry.state)
            locations.append(ground.findStationLocationInAnotherFrame(geometry.state, point, femur).to_numpy())
        vectors = np.stack((locations[0] - locations[1], locations[2] - locations[1]))
        directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        # The right femur's +Z points laterally. Negative medial components
        # correspond to guide traction away from the femur.
        medial_components.append(-directions[:, 2].sum())
    return np.array(medial_components)


def reference_paths(geometry, names, model_name):
    geometry.pose(np.zeros(3))
    ground = geometry.model.getGround()
    femur = geometry.model.getBodySet().get("femur_r")
    hip_center = geometry.model.getJointSet().get("hip_r").getChildFrame().getPositionInGround(geometry.state)
    center_local = ground.findStationLocationInAnotherFrame(geometry.state, hip_center, femur).to_numpy()
    records = []
    for name in names:
        muscle = geometry.model.getMuscles().get(name)
        active_path = muscle.getGeometryPath().getCurrentPath(geometry.state)
        points = []
        for i in range(active_path.getSize()):
            point = active_path.get(i)
            location = ground.findStationLocationInAnotherFrame(
                geometry.state, point.getLocationInGround(geometry.state), femur
            ).to_numpy() - center_local
            points.append({
                "name": point.getName(), "body": point.getParentFrame().findBaseFrame().getName(),
                **dict(zip(("x", "y", "z"), packed(location))),
            })
        records.append({"name": name, "model": model_name, "points": points,
                        "neutralLengthM": muscle.getLength(geometry.state),
                        "neutralAdductionArmM": muscle.computeMomentArm(geometry.state, geometry.coordinates[1])})
    return records


def interpolate(table, pose):
    indices = [min(max(int(np.searchsorted(axis, angle, side="right")) - 1, 0), len(axis) - 2)
               for axis, angle in zip(AXES, pose)]
    fractions = [(angle - axis[i]) / (axis[i+1] - axis[i])
                 for axis, angle, i in zip(AXES, pose, indices)]
    result = 0
    for corner in itertools.product((0, 1), repeat=3):
        weight = np.prod([t if c else 1 - t for t, c in zip(fractions, corner)])
        result += weight * table[tuple(i+c for i, c in zip(indices, corner))]
    return result


def run(cache):
    opensim.Logger.removeFileSink()
    opensim.Logger.setLevelString("error")
    cache.mkdir(exist_ok=True, parents=True)
    cases = {name: Geometry(model_file(cache, *spec)) for name, spec in MODELS.items()}
    current = cases["MRIInformed2023"]
    band = Geometry(model_file(cache, *ITB_SOURCE))
    capacities = np.array([band.model.getMuscles().get(key).getMaxIsometricForce() for key in ITB_TRACTS])
    tract_weights = capacities / capacities.sum()
    poses = np.array(list(itertools.product(*AXES)))
    lengths, arms, itb_components = [], [], []
    comparison = {case: [] for case in cases}

    for pose in poses:
        radians = np.deg2rad(pose)
        for name, geometry in cases.items():
            geometry.pose(radians)
            comparison[name].append(geometry.effective_arm())
        lengths.append(current.lengths())
        arms.append(current.arms())
        itb_components.append(itb_reaction(band, radians))

    lengths, arms = np.array(lengths), np.array(arms)
    effective = np.array(comparison["MRIInformed2023"])
    itb_components = np.array(itb_components)
    inward = np.maximum(0, itb_components) @ tract_weights
    if not np.all(np.isfinite(inward)) or np.max(inward) > 2:
        raise ArithmeticError("Invalid unit-tension guide reaction")
    if not np.all(np.isfinite(arms)) or np.min(lengths) <= 0 or np.min(effective) <= 0:
        raise ArithmeticError("Nonphysical or missing model outputs")

    # A derivative of total musculotendon length wrt radians should equal minus the
    # signed moment arm for these independent hip rotational coordinates. Perturb
    # the pose before each length evaluation to avoid reusing baseline values.
    epsilon = 1e-5
    rng = np.random.default_rng(462023)
    audit_poses = rng.uniform([a[0] for a in AXES], [a[-1] for a in AXES], (256, 3))
    errors, interpolation_errors, band_errors = [], [], []
    table = effective.reshape(tuple(len(axis) for axis in AXES))
    for pose in audit_poses:
        radians = np.deg2rad(pose)
        current.pose(radians)
        baseline = current.arms()
        interpolation_errors.append(interpolate(table, pose) + current.weights @ baseline[:, 1])
        direct = np.maximum(0, itb_reaction(band, radians)) @ tract_weights
        band_errors.append(abs(direct - interpolate(inward.reshape(table.shape), pose)))
        for axis_index in range(3):
            step = epsilon * np.eye(3)[axis_index]
            current.pose(radians - step)
            before = current.lengths()
            current.pose(radians + step)
            after = current.lengths()
            errors.extend(baseline[:, axis_index] + (after-before) / (2*epsilon))

    summary = {}
    neutral = np.flatnonzero(np.all(poses == 0, axis=1)).item()
    for name, values in comparison.items():
        summary[name] = {"neutralM": values[neutral], "minM": min(values), "maxM": max(values)}
    delta = np.array(comparison["MRIPersonalizedExample"]) - comparison["genericScaledExample"]
    unit_demand_ratio = np.array(comparison["genericScaledExample"]) / comparison["MRIPersonalizedExample"]
    summary["workedExample"] = {
        "subjectCount": 1,
        "medianSignedDifferenceM": float(np.median(delta)),
        "neutralSignedDifferenceM": float(delta[neutral]),
        "unitMomentForceRatioAtNeutral": float(unit_demand_ratio[neutral]),
        "unitMomentForceRatioRange": [float(unit_demand_ratio.min()), float(unit_demand_ratio.max())],
        "interpretation": "MRI-personalized vs generic-scaled at identical coordinates in one worked example. Force ratio assumes the six segments alone balance an equal imposed abduction moment and share tension in proportion to model maximum isometric force."
    }
    if max(abs(np.array(errors))) > 5e-5:
        raise ArithmeticError("Signed OpenSim leverage does not agree with the path-length derivative")
    return {
        "model": "RajagopalLaiUhlrich2023",
        "openSimVersion": opensim.GetVersion(),
        "sources": {key: {"url": url, "sha256": digest} for key, (url, digest) in
                    {**MODELS, "Davis2026": ITB_SOURCE}.items()},
        "references": ["https://doi.org/10.1109/TBME.2016.2586891", "https://doi.org/10.1038/s41598-022-13386-9",
                       "https://doi.org/10.1371/journal.pcbi.1014073"],
        "axes": {name: axis.tolist() for name, axis in zip(("flexionDeg","adductionDeg","rotationDeg"), AXES)},
        "packing": "Arrays in flexion × adduction × rotation order, rotation varies fastest. Coordinates are positive for flexion, adduction and internal rotation, other coordinates held at defaults.",
        "effectiveArmDefinition": "Negative of the signed adduction moment arm, averaged over three medius and three minimus segments with weights proportional to maximum isometric force. This is a recruitment assumption. Distances are metres.",
        "effectiveAbductionArmM": packed(effective),
        "comparisonEffectiveAbductionArmM": {
            key: packed(values) for key, values in comparison.items() if key != "MRIInformed2023"
        },
        "referencePaths": {
            "frame": "Neutral right femur frame with origin at hip joint center, metres. +X anterior, +Y superior, +Z lateral (right). Other coordinates at model defaults. Nodes follow their named body; display on an independent atlas requires registration. OpenSim computes path wrapping anew at each pose.",
            "paths": (reference_paths(current, MUSCLES, "MRIInformed2023")
                      + reference_paths(band, ITB_TRACTS, "Davis2026")),
        },
        "itb": {
            "source": "Davis2026", "reference": "https://doi.org/10.51224/SportRxiv.1017",
            "description": "Medial guide reaction in newtons per newton total tension across the three ITB sub-tracts. Tension is distributed by maximum isometric force. At each named trochanteric via point sum the two adjacent unit tensions in the femur frame and retain the positive inward component along -Z. This regional proxy assumes frictionless redirection and a lateral reference surface. Actual tendon contact, bursa pressure and symptoms require additional measurements.",
            "tractWeights": dict(zip(ITB_TRACTS, tract_weights.tolist())),
            "inwardCoefficient": packed(inward),
            "signedMedialCoefficientByTract": {key: packed(itb_components[:,i]) for i, key in enumerate(ITB_TRACTS)},
            "maxOffGridInterpolationError": max(band_errors),
            "neutralFlexionRotationAdductionSweep": [
                {"adductionDeg": int(adduction), "guideReactionPer100NTensionN": float(100*inward[i])}
                for i, (flexion, adduction, rotation) in enumerate(poses) if flexion == rotation == 0
            ]
        },
        "muscles": {key: {
            "maxIsometricForceN": float(current.maximum_force[i]), "lengthM": packed(lengths[:,i]),
            "flexionArmM": packed(arms[:,i,0]), "adductionArmM": packed(arms[:,i,1]),
            "rotationArmM": packed(arms[:,i,2]),
        } for i, key in enumerate(MUSCLES)},
        "validation": {
            "gridPoses": len(poses), "randomOffGridPoses": len(audit_poses), "seed": 462023,
            "derivativeStepRad": epsilon, "segmentAxisDerivativeComparisons": len(errors),
            "absolutePathDerivativeError": distribution(errors),
            "effectiveArmInterpolationError": distribution(interpolation_errors)
        },
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "hip-osim-reference")
    parser.add_argument("--output", type=Path, default=REFERENCE)
    args = parser.parse_args()
    result = run(args.cache)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as tmp:
        json.dump(result, tmp, separators=(",",":"), allow_nan=False)
        tmp.write("\n")
    Path(tmp.name).replace(args.output)
    print(json.dumps({"validation": result["validation"], "summary": result["summary"]}, indent=2))
