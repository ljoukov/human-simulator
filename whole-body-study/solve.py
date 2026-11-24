"""Reconstruct one recorded stride using the GC4-JW full-body OpenSim model.

Usage: python solve.py /path/to/ReducePoly_forMSK [--output stride.json]
Requires OpenSim 4.6, NumPy and SciPy. See README.md for the source revision.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import opensim as osim
from scipy.io import loadmat
from force_response import measure_force_response
from force_plate_check import estimate_force

TRIAL = "ngait_og5"
MODEL = Path("MuscleTendonLengthParameterizationReduction/JW All Body Model-scaled_forvisualization_leftright_v42.osim")
RECORDINGS = Path("ExampleTrackingSimulations/OpenSimModel/subject_GC")
REFERENCE = "https://github.com/gilserrancoli/ReducePoly_forMSK/tree/00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8"
PHASE = np.arange(101)
MOMENTS = {
    "hip_flexion":"Right hip flexion", "hip_adduction":"Right hip adduction",
    "knee_flexion":"Right knee flexion", "ankle_angle":"Right ankle",
    "lumbar_bending":"Lumbar bending", "lumbar_rotation":"Lumbar rotation",
    "arm_flex_r":"Right shoulder flexion", "elbow_flex_r":"Right elbow flexion",
    "hip_flexion_l":"Left hip flexion", "knee_flexion_l":"Left knee flexion",
    "ankle_angle_l":"Left ankle",
}
MUSCLE_LABELS = {"glmed2":"Right gluteus medius, middle", "gasmed":"Right medial gastrocnemius",
    "tfl":"Right tensor fasciae latae", "semiten":"Right semitendinosus", "ercspn_r":"Right erector spinae",
    "glut_med2_l":"Left gluteus medius, middle", "rect_fem_l":"Left rectus femoris"}
LIGAMENT_LABELS = {"patlig_lat": "Right lateral patellar ligament path",
                    "patlig_center": "Right central patellar ligament path",
                    "patlig_med": "Right medial patellar ligament path"}


def read_mat(path, key):
    vals = loadmat(path, simplify_cells=True)[key]
    return dict(zip(vals["colheaders"], vals["data"].T, strict=True))


def recording_file(source, tag, trial):
    return source / RECORDINGS / tag / f"{tag}_{trial}.mat"


def compare_overground_trials(model, source, ordinary):
    """Use the unchanged estimator on every complete right overground stride."""
    trials = []
    for trial, label in [(TRIAL, "Ordinary"), ("bouncy7", "Bouncy"), ("mtpgait3", "Forefoot strike")]:
        if trial == TRIAL:
            check = ordinary
        else:
            ik = read_mat(recording_file(source, "IK", trial), "Qsall")
            grf = read_mat(recording_file(source, "GRF", trial), "GRFall")
            start = crossing(grf["time"], grf["grforce2_fy"], True)
            stop = crossing(grf["time"], grf["grforce3_fy"], True)
            check = estimate_force(model, ik, np.linspace(start, stop, len(PHASE)), grf)
        v = check["vertical"]
        trials.append({"recording": trial, "condition": label, "clipId": f"{trial}-stance-1",
                       "phasePctRange": [20, 100], "forcePlateCheck": check,
                       "wholeBodyRmseN": v["rmseN"], "pelvisOnlyRmseN": v["pelvisOnlyRmseN"],
                       "wholeBodyBiasN": v["biasN"], "wholeBodyCorrelation": v["correlation"]})
    return trials


def write_storage(path, series, degrees):
    data = np.column_stack(list(series.values()))
    header = (f"{path.stem}\nversion=1\nnRows={len(data)}\nnColumns={len(series)}\n"
              f"inDegrees={'yes' if degrees else 'no'}\nendheader\n" + "\t".join(series.keys()))
    np.savetxt(path, data, delimiter="\t", fmt="%.12g", header=header, comments="")


def crossing(t, force, rising):
    on = force > 20
    change = on[1:] & ~on[:-1] if rising else on[:-1] & ~on[1:]
    indices = np.flatnonzero(change)
    if len(indices) != 1:
        raise ValueError(f"Expected one 20 N crossing, found {len(indices)}")
    i = indices[0]
    return float(t[i] + (t[i+1] - t[i]) * (20-force[i]) / (force[i+1]-force[i]))


def external_loads(work, grf):
    loads = osim.ExternalLoads()
    write_storage(work / "plate.mot", grf, False)
    loads.setDataFileName(str(work / "plate.mot"))
    for number, body in [(1, "calcaneus_l"), (2, "calcaneus"), (3, "calcaneus")]:
        force = osim.ExternalForce()
        force.setName(f"plate{number}")
        force.setAppliedToBodyName(body)
        force.setForceExpressedInBodyName("ground")
        force.setPointExpressedInBodyName("ground")
        for setter, code in [(force.setForceIdentifier, "f"), (force.setPointIdentifier, "p"), (force.setTorqueIdentifier, "t")]:
            setter(f"grforce{number}_{code}")
        loads.cloneAndAppend(force)
    loads.printToXML(str(work / "plates.xml"))


def id_solve(model, coordinates_path, work, times, suffix):
    tool = osim.InverseDynamicsTool()
    tool.setModelFileName(str(model))
    tool.setCoordinatesFileName(str(coordinates_path))
    tool.setStartTime(float(times[0] - 0.025))
    tool.setEndTime(float(times[-1] + 0.025))
    tool.setLowpassCutoffFrequency(6)
    tool.setExternalLoadsFileName(str(work / "plates.xml"))
    tool.setResultsDir(str(work))
    tool.setOutputGenForceFileName(f"id-{suffix}.sto")
    if not tool.run():
        raise RuntimeError("OpenSim inverse dynamics failed")
    path = work / f"id-{suffix}.sto"
    lines = path.read_text().splitlines()
    names = lines[lines.index("endheader")+1].split()
    data = np.loadtxt(lines[lines.index("endheader")+2:])
    return dict(zip(names, data.T, strict=True))


def tissue_paths(model, ik, query):
    state = model.initSystem()
    coords = [model.getCoordinateSet().get(i) for i in range(model.getCoordinateSet().getSize())]
    muscles = [model.getMuscles().get(i) for i in range(model.getMuscles().getSize())]
    ligaments = [osim.Ligament.safeDownCast(model.getForceSet().get(name)) for name in LIGAMENT_LABELS]
    paths = [tissue.getGeometryPath() for tissue in muscles + ligaments]
    q = np.array([np.interp(query, ik["time"], ik[c.getName()]) for c in coords]).T
    rotational = [i for i,c in enumerate(coords) if c.getMotionType() == osim.Coordinate.Rotational]
    q[:,rotational] = np.deg2rad(q[:,rotational])
    lengths = []
    for row in q:
        for coord, value in zip(coords, row):
            coord.setValue(state, float(value), False)
        model.realizePosition(state)
        lengths.append([1000 * path.getLength(state) for path in paths])
    matrix = np.array(lengths)
    if not np.isfinite(matrix).all() or np.min(matrix) <= 0:
        raise ValueError("Invalid tissue path length")
    labels = MUSCLE_LABELS | LIGAMENT_LABELS
    traces = {tissue.getName(): {"label":labels.get(tissue.getName(), tissue.getName()),
                                 "lengthMm":np.round(values, 3).tolist()}
              for tissue, values in zip(muscles + ligaments, matrix.T, strict=True)}
    return ({m.getName():traces[m.getName()] for m in muscles},
            {name:traces[name] for name in LIGAMENT_LABELS})


def arm_transforms(model, static_arms):
    """Relative arm poses belonging to exactly the same counterfactual as the ID."""
    state = model.initSystem()
    coords = model.getCoordinateSet()
    for key, values in static_arms.items():
        if key.startswith(("arm_", "elbow_", "pro_sup_", "wrist_")):
            coords.get(key).setValue(state, np.radians(values[0]), False)
    model.realizePosition(state)
    bodies = model.getBodySet()
    torso = bodies.get("torso")
    result = {}
    for side in ("r", "l"):
        for bone in ("humerus", "ulna", "radius", "hand"):
            key = f"{bone}_{side}"
            frame = bodies.get(key).findTransformBetween(state, torso)
            position = [frame.p().get(i) for i in range(3)]
            q = frame.R().convertRotationToQuaternion()
            result[key] = np.round(position + [q.get(i) for i in (1, 2, 3, 0)], 8).tolist()
    return result


def solve(source, work):
    trial_paths = {tag: recording_file(source, tag, TRIAL) for tag in ("IK", "GRF", "ID")}
    input_files = [source / MODEL, *trial_paths.values()]
    input_files += [recording_file(source, tag, trial) for trial in ("bouncy7", "mtpgait3") for tag in ("IK", "GRF")]
    ik = read_mat(trial_paths["IK"], "Qsall")
    grf = read_mat(trial_paths["GRF"], "GRFall")
    start = crossing(grf["time"], grf["grforce2_fy"], True)
    stop = crossing(grf["time"], grf["grforce3_fy"], True)
    toeoff = crossing(grf["time"], grf["grforce2_fy"], False)
    time = np.linspace(start, stop, len(PHASE))
    model = osim.Model(str(source / MODEL))
    coords = model.updCoordinateSet()
    for i in range(coords.getSize()):
        coords.get(i).setDefaultLocked(False)
    state = model.initSystem()
    muscles, ligaments = tissue_paths(model, ik, time)
    plate_check = estimate_force(model, ik, time, grf)
    comparison = compare_overground_trials(model, source, plate_check)
    counts = {"bodyCount":model.getBodySet().getSize(), "coordinateCount":coords.getSize(),
              "muscleCount":len(muscles), "jointCount":model.getJointSet().getSize(),
              "ligamentPathCount":len(ligaments), "massKg":round(model.getTotalMass(state),4)}
    # Net generalized forces include all internal contributions together. Remove
    # actuators and passive ligaments before inference; retain measured external loads.
    model.updForceSet().clearAndDestroy()
    model.finalizeFromProperties()
    model.printToXML(str(work / "id-model.osim"))
    external_loads(work, grf)
    write_storage(work / "motion.mot", ik, True)
    computed = id_solve(work / "id-model.osim", work / "motion.mot", work, time, "observed")
    # Paired counterfactual at exactly the same ground reactions and leg postures.
    # Neutralize arm motion only; the altered recording has no independent ground truth.
    static_arms = ik.copy()
    for key in ik:
        if key.startswith(("arm_", "elbow_", "pro_sup_", "wrist_")):
            static_arms[key] = np.full_like(ik[key], np.mean(np.interp(time, ik["time"], ik[key])))
    write_storage(work / "fixed-arms.mot", static_arms, True)
    no_swing = id_solve(work / "id-model.osim", work / "fixed-arms.mot", work, time, "fixed-arms")
    reference = read_mat(trial_paths["ID"], "IDall")
    moments = {}
    for key,label in MOMENTS.items():
        column = f"{key}_moment"
        a = np.interp(time, computed["time"], computed[column])
        b = np.interp(time, reference["time"], reference[column])
        c = np.interp(time, no_swing["time"], no_swing[column])
        if not np.isfinite([a,b,c]).all():
            raise ValueError(f"Nonfinite {column}")
        moments[key] = {"label":label, "computedNm":np.round(a,4).tolist(), "referenceNm":np.round(b,4).tolist(),
                        "fixedArmsNm":np.round(c,4).tolist(), "rmseNm":float(np.sqrt(np.mean((a-b)**2))),
                        "fixedArmsRmsChangeNm":round(float(np.sqrt(np.mean((a-c)**2))),4)}
    return {"source":{"url":REFERENCE, "modelPath":str(MODEL), "recording":TRIAL, "clipId":f"{TRIAL}-stance-1",
                      "subject":"GC4-JW", "timeSeconds":[start,stop], "stanceEndPct":100*(toeoff-start)/(stop-start),
                      "sha256":{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in input_files}},
            "model":counts, "phasePct":PHASE.tolist(), "moments":moments, "muscles":muscles,
            "ligaments":ligaments, "forcePlateCheck":plate_check, "forcePlateComparison":comparison,
            "fixedArmSegmentsInTorso":arm_transforms(model, static_arms),
            "forwardResponse":measure_force_response(source / MODEL, ik, grf, time, list(MOMENTS)),
            "summary":{"solver":osim.GetVersion(), "kinematicsCutoffHz":6, "phaseSamples":len(time),
                       "pathEvaluations":len(muscles)*len(time), "ligamentPathEvaluations":len(ligaments)*len(time), "inputForcePlates":3,
                       "intervention":"Both arms fixed at their stride-mean joint angles. Identical leg, trunk, pelvis and external force records."}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("stride.json"))
    args = parser.parse_args()
    osim.Logger.removeFileSink()
    osim.Logger.setLevelString("error")
    with tempfile.TemporaryDirectory() as temporary:
        result = solve(args.source.resolve(), Path(temporary).resolve())
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False)+"\n")
