"""Export projected Simbody inertias for the interactive perturbation solver.

Requires OpenSim 4.6, NumPy and SciPy. Download the ExperimentalData.mat and
subject1.osim files specified in hip-study/forward/reference.json, then run
python extract_articulation.py /path/to/subject1.osim /path/to/ExperimentalData.mat.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import opensim as osim
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parents[2] / "hip-study/forward/reference.json"
COORDINATES = [
    "pelvis_list", "lumbar_bending", "hip_adduction_r", "hip_adduction_l",
    "hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l",
    "ankle_angle_r", "ankle_angle_l", "arm_flex_r", "arm_flex_l",
    "arm_add_r", "arm_add_l", "elbow_flex_r", "elbow_flex_l",
]


def extract(model_path, movement_path):
    source = json.loads(REFERENCE.read_text())
    for file, name in [(model_path, "subject1.osim"), (movement_path, "ExperimentalData.mat")]:
        if hashlib.sha256(file.read_bytes()).hexdigest() != source["fileSha256"][name]:
            raise ValueError(f"{name} does not match the published source revision")
    osim.Logger.setLevelString("Error")
    osim.Logger.removeFileSink()
    model = osim.Model(str(model_path))
    state = model.initSystem()
    coords = model.updCoordinateSet()
    body_set = model.getBodySet()
    data = loadmat(movement_path, simplify_cells=True)["ExperimentalData"]
    record = data["Q"]["subject1"]["Qs"]
    means = record["mean"]
    channels = {name.strip(): i for i, name in enumerate(record["colheaders"])}

    # Explicitly determine which Simbody mobility each coordinate speed drives.
    # CoordinateSet order need not match the multibody tree's mobility order.
    mapping = np.zeros((state.getNU(), len(COORDINATES)))
    for j, name in enumerate(COORDINATES):
        set_speed = coords.get(name).setSpeedValue
        set_speed(state, 1.0)
        mapping[:, j] = state.getU().to_numpy().copy()
        set_speed(state, 0.0)
    # x forward, y up, z right. Positive pelvis_list tilts the right side down;
    # positive lumbar_bending tilts right relative to the pelvis. Express our
    # second perturbation as global frontal lean about the reference pose.
    mapping[:, 0] = -mapping[:, 0] + mapping[:, 1]
    for j in (6, 7):
        mapping[:, j] *= -1  # this OpenSim model uses negative knee flexion.
    if np.linalg.matrix_rank(mapping) != len(COORDINATES):
        raise ValueError("Selected rotational mobilities are not independent")

    n = len(means)
    nominal_defaults = {coords.get(i).getName(): coords.get(i).getDefaultValue()
                        for i in range(coords.getSize())}
    tagged = []
    for name in nominal_defaults:
        side = name[-2:]
        root = name[:-2] if side in ("_r", "_l") else name
        if root in channels:
            tagged.append((coords.get(name), channels[root], side == "_l",
                           root in ("pelvis_tx", "pelvis_ty", "pelvis_tz")))
    def set_walk_pose(phase):
        # The source publishes one side's ensemble mean. Reconstruct a
        # bilaterally symmetric nominal gait with half-cycle contralateral lag.
        for coordinate, col, left, translation in tagged:
            pos = ((phase + 0.5 * left) % 1.0) * n
            row = int(pos) % n
            alpha = pos - int(pos)
            val = (1-alpha)*means[row, col] + alpha*means[(row+1) % n, col]
            coordinate.setValue(state, val if translation else np.deg2rad(val), False)

    def projected_matrix():
        model.realizePosition(state)
        mass = osim.Matrix()
        model.getMatterSubsystem().calcM(state, mass)
        mat = mapping.T @ mass.to_numpy() @ mapping
        np.testing.assert_allclose(mat, mat.T, atol=1e-10)
        np.linalg.cholesky(mat)
        return mat

    # Sample every archived phase, then close the display stride at the first pose.
    phases = np.linspace(0, 1, n + 1)
    matrices = []
    for phase in phases:
        set_walk_pose(phase)
        matrices.append(projected_matrix())
    interpolation_errors = []
    energy_interpolation_bounds = []
    energy_errors = []
    rng = np.random.default_rng(514481)
    for i in range(len(phases)-1):
        phase = (phases[i]+phases[i+1])/2
        set_walk_pose(phase)
        exact = projected_matrix()
        approx = (matrices[i] + matrices[i+1])/2
        interpolation_errors.append(float(np.linalg.norm(approx-exact)/np.linalg.norm(exact)))
        # For any nonzero velocity v, bound vᵀ(approx-exact)v / (vᵀ exact v)
        # by the largest absolute generalized eigenvalue of this matrix pair.
        factor = np.linalg.cholesky(exact)
        whitened = np.linalg.solve(factor, approx-exact)
        whitened = np.linalg.solve(factor, whitened.T).T
        energy_interpolation_bounds.append(float(np.max(np.abs(np.linalg.eigvalsh(whitened)))))
        velocity = rng.normal(size=len(COORDINATES))
        state.setU(osim.Vector.createFromMat(mapping @ velocity))
        model.realizeVelocity(state)
        energy_errors.append(abs(0.5 * float(velocity @ exact @ velocity)
                                 - model.calcKineticEnergy(state)))
        state.setU(osim.Vector(state.getNU(), 0.0))
    if max(energy_errors) > 1e-8:
        raise ValueError(f"Simbody kinetic energy mismatch: {max(energy_errors)} J")
    # A separate designed stance favors a nearly upright loaded right leg;
    # all unlisted coordinates stay at their source-model default.
    for name, default in nominal_defaults.items():
        coords.get(name).setValue(state, default, False)
    stance_degrees = {"hip_flexion_r": 2, "hip_adduction_r": 5, "hip_flexion_l": 12,
                      "knee_angle_l": -20, "lumbar_bending": 0.8}
    for name, angle in stance_degrees.items():
        coords.get(name).setValue(state, np.deg2rad(angle), False)
    stance = projected_matrix()
    mass_kg = model.getTotalMass(state)
    def flatten(mat):
        return mat.round(8).ravel().tolist()
    return {
        "source": source["source"], "modelSha256": source["fileSha256"]["subject1.osim"],
        "experimentalSha256": source["fileSha256"]["ExperimentalData.mat"],
        "openSimVersion": osim.GetVersion(), "units": "kg m²", "massKg": mass_kg,
        "coordinates": COORDINATES,
        "basis": "First slot = positive left pelvic drop; second slot = incremental global right trunk lean. Native pelvis_list = -drop, lumbar_bending = lean + drop (local frontal projection). Knees flex positive; other coordinate signs are native.",
        "layout": "Dense row-major 16 x 16. Multiply by angular acceleration in rad/s² to obtain generalized torques in N m. Each matrix is Bᵀ M(q) B. Perturbation velocities outside this subspace are prescribed, including pelvis translation.",
        "referencePoses": "100 point mean of eleven trials from one adult. Right-side coordinate means supply the left side shifted by half a cycle. Upper-body and pelvic coordinate means use the original recording; unrecorded axes use model defaults. The mass matrices are evaluated on reference poses.",
        "stancePoseDegrees": stance_degrees,
        "segmentMassKg": {body_set.get(i).getName(): body_set.get(i).getMass()
                          for i in range(body_set.getSize())},
        "sourceMobilities": state.getNU(), "phases": phases.round(4).tolist(),
        "matrices": [flatten(mat) for mat in matrices], "stanceMatrix": flatten(stance),
        "checks": {"midpointPoses": len(interpolation_errors),
                   "maxRelativeFrobeniusInterpolationError": max(interpolation_errors),
                   "worstRelativeKineticEnergyErrorAcrossAllVelocitiesAtMidpoints": max(energy_interpolation_bounds),
                   "largestKineticEnergyErrorJ": max(energy_errors),
                   "minimumEigenvalueKgM2": float(np.linalg.eigvalsh(matrices).min())}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("movement", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "articulation.json")
    args = parser.parse_args()
    result = extract(args.model, args.movement)
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps(result["checks"]))
