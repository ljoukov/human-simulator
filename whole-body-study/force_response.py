"""Immediate constrained multibody response to a decrement in gluteal path force."""
import numpy as np
import opensim as osim


def trial_pose(model, ik, t):
    state = model.initSystem()
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        c = cs.get(i)
        value = np.interp(t, ik["time"], ik[c.getName()])
        if c.getMotionType() == osim.Coordinate.Rotational:
            value = np.radians(value)
        locked = c.getLocked(state)
        if locked:
            c.setLocked(state, False)
        c.setValue(state, float(value), False)
        if locked:
            c.setLocked(state, True)
    model.realizePosition(state)
    return state


def force_response(model_file, ik, grf, t, moment_names, center_offset_m=0.0):
    model = osim.Model(str(model_file))
    state = trial_pose(model, ik, t)
    foot = model.getBodySet().get("calcaneus")
    ground = model.getGround()
    cop = np.array([np.interp(t, grf["time"], grf["grforce2_p"+axis]) for axis in "xyz"])
    cop[0] += center_offset_m
    station_world = osim.Vec3(*cop)
    station_foot = ground.findStationLocationInAnotherFrame(state, station_world, foot)
    primary_constraint = osim.PointConstraint(ground, station_world, foot, station_foot)
    primary_constraint.setName("right_foot_support_point")
    model.addConstraint(primary_constraint)
    state = trial_pose(model, ik, t)
    coords = model.getCoordinateSet()
    before = [coords.get(i).getValue(state) for i in range(coords.getSize())]
    model.assemble(state)
    rotation_change = [abs(np.degrees(coords.get(i).getValue(state)-q)) for i,q in enumerate(before)
                       if coords.get(i).getMotionType() == osim.Coordinate.Rotational]
    model.realizePosition(state)
    at_ground = foot.findStationLocationInGround(state, station_foot)
    pin_position_error = np.linalg.norm([at_ground.get(i) - cop[i] for i in range(3)])
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        muscle.overrideActuation(state, True)
        muscle.setOverrideActuation(state, 0)
    selected = [muscles.get(f"glmed{i}") for i in (1, 2, 3)]
    values = {}
    pin_acceleration_error = 0.0
    for tension in (50.0, 25.0, 0.0):
        for path in selected:
            path.setOverrideActuation(state, tension / len(selected))
        model.realizeAcceleration(state)
        values[tension] = np.array([coords.get(key).getAccelerationValue(state) for key in moment_names])
        acc = foot.findStationAccelerationInGround(state, station_foot)
        pin_acceleration_error = max(pin_acceleration_error, np.linalg.norm([acc.get(i) for i in range(3)]))
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Invalid acceleration from the full-body forward dynamics solve")
    delta = np.degrees(values[0.0] - values[50.0])
    half_delta = np.degrees(values[25.0] - values[50.0])
    return {
        "deltaDegPerSecondSquared":{key:round(float(value),3) for key,value in zip(moment_names, delta)},
        "halfForceLinearityMaxErrorDegPerSecondSquared":float(np.max(np.abs(2 * half_delta - delta))),
        "pinPositionErrorM":float(pin_position_error),
        "pinAccelerationErrorMPerSecondSquared":float(pin_acceleration_error),
        "maxAssemblyAngleChangeDeg":float(max(rotation_change)),
    }


def measure_force_response(model_file, ik, grf, stride_times, moment_names):
    samples = []
    for phase in (20, 30, 40):
        t = stride_times[phase]
        result = force_response(model_file, ik, grf, t, moment_names)
        result["phasePct"] = phase
        result["recordedRightVerticalGroundForceN"] = round(float(np.interp(t, grf["time"], grf["grforce2_fy"])), 2)
        samples.append(result)
    sensitivity = []
    for offset in (-0.01, 0.01):
        result = force_response(model_file, ik, grf, stride_times[30], moment_names, offset)
        sensitivity.append({"laboratoryXOffsetMm":round(offset*1000),
                            "deltaDegPerSecondSquared":result["deltaDegPerSecondSquared"]})
    return {
        "totalForceReductionN":50, "rightMusclePaths":["glmed1","glmed2","glmed3"],
        "forceSharing":"Equal force perturbation in each path; other muscle tensions held fixed.",
        "support":"Right calcaneus pinned at the measured right center of pressure. Rotation around the point is free. Left foot is free.",
        "initialState":"Three recorded midstance poses. Instantaneous rest at each assembled pose; output is the paired change in initial acceleration.",
        "units":"degrees per second squared, in signed native OpenSim joint coordinates",
        "outcome":"Change on removing 50 N of force, perturbed minus reference. This imposed differential force is a scenario, not a measured activation or an impairment severity.",
        "samples":samples, "supportPointSensitivityAt30Pct":sensitivity,
    }
