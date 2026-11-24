"""Estimate net ground reaction from segment motion, then compare it with the plates."""
import numpy as np
import opensim as osim
from scipy.signal import butter, sosfiltfilt


def center_positions(model, state, coordinates, poses):
    com = []
    pelvis = []
    pelvis_frame = model.getBodySet().get("pelvis")
    for pose in poses:
        for coordinate, value in zip(coordinates, pose, strict=True):
            coordinate.setValue(state, float(value), False)
        model.realizePosition(state)
        position = model.calcMassCenterPosition(state)
        com.append([position.get(i) for i in range(3)])
        origin = pelvis_frame.getPositionInGround(state)
        pelvis.append([origin.get(i) for i in range(3)])
    return np.array(com), np.array(pelvis)


def estimate_force(model, ik, stride_time, grf):
    # Include the complete motion trace so filtering does not end at heel strike.
    time = ik["time"]
    dt = (time[-1] - time[0]) / (len(time) - 1)
    regular_time = time[0] + np.arange(len(time)) * dt
    state = model.initSystem()
    coordinate_set = model.getCoordinateSet()
    coordinates = [coordinate_set.get(i) for i in range(coordinate_set.getSize())]
    poses = []
    for coordinate in coordinates:
        recorded = ik[coordinate.getName()]
        poses.append(np.radians(recorded) if coordinate.getMotionType() == osim.Coordinate.Rotational else recorded)
    com, pelvis = center_positions(model, state, coordinates, np.array(poses).T)
    if np.max(np.abs(pelvis[:, 1] - ik["pelvis_ty"])) > 1e-8:
        raise ValueError("The pelvis-origin marker and the force proxy use different heights")
    # Match the animation's interpolation of joint coordinates, not the interpolation
    # of nonlinear positions after forward kinematics.
    display_poses = np.column_stack([np.interp(stride_time, time, coordinate) for coordinate in poses])
    display_com, display_pelvis = center_positions(model, state, coordinates, display_poses)
    if stride_time[20] <= time[0] or stride_time[-1] >= time[-1]:
        raise ValueError("Force comparison extends beyond the recorded motion")
    # Apply identical processing to the pelvis-height proxy and whole-body COM.
    positions = np.column_stack((com, ik["pelvis_ty"]))
    uniform = np.column_stack([np.interp(regular_time, time, axis) for axis in positions.T])
    filtered = sosfiltfilt(butter(2, 6.0, fs=1 / dt, output="sos"), uniform, axis=0)
    acceleration = np.gradient(np.gradient(filtered, dt, axis=0), dt, axis=0)
    gravity = np.array([model.getGravity().get(i) for i in range(3)])
    calculated = model.getTotalMass(state) * (acceleration[:, :3] - gravity)
    pelvis_force = model.getTotalMass(state) * (acceleration[:, 3] - gravity[1])

    # The trailing left foot before the first plate contact can be off the walkway.
    # Start after the first double-support interval, and include the rest of the stride.
    phase = np.arange(20, 101)
    query = stride_time[phase]
    measured_body_weight_n = 65.0 * 9.80665
    result = {
        "phasePct": phase.tolist(), "participantMassKg": 65.0, "modelMassKg": model.getTotalMass(state),
        "kinematicsLowpassHz": 6.0,
        "positionMarkers": {
            "phasePct": list(range(len(stride_time))),
            "centerOfMassGroundM": display_com.round(7).tolist(),
            "pelvisOriginGroundM": display_pelvis.round(7).tolist(),
            "clipOriginGroundM": [round(float(display_pelvis[0, 0]), 7), 0.0, round(float(display_pelvis[0, 2]), 7)],
        },
        "method": "Sum the modeled segment masses and center-of-mass accelerations; subtract model weight. Compare the net external force independently with the three force plates.",
    }
    for component, (axis, label) in enumerate(zip("xyz", ("laboratoryX", "vertical", "laboratoryZ"), strict=True)):
        estimated = np.interp(query, regular_time, calculated[:, component])
        net_plate = sum(grf[f"grforce{i}_f{axis}"] for i in (1, 2, 3))
        measured = np.interp(query, grf["time"], net_plate)
        error = estimated - measured
        rmse = float(np.sqrt(np.mean(error ** 2)))
        result[label] = {
            "computedN": estimated.round(2).tolist(), "measuredN": measured.round(2).tolist(),
            "rmseN": rmse, "biasN": float(np.mean(error)),
            "correlation": float(np.corrcoef(estimated, measured)[0, 1]),
            "rmsePctParticipantBodyWeight": 100 * rmse / measured_body_weight_n,
        }
        if label == "vertical":
            pelvis_only = np.interp(query, regular_time, pelvis_force)
            result[label]["pelvisOnlyN"] = pelvis_only.round(2).tolist()
            result[label]["pelvisOnlyRmseN"] = float(np.sqrt(np.mean((pelvis_only - measured) ** 2)))
    return result
