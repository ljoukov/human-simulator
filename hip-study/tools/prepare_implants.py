"""Derive hip frontal-plane force balance inputs from synchronized OrthoLoad trials.

python3 hip-study/tools/prepare_implants.py --cache /tmp/orthoload-hip
Requires numpy and curl. Source data are cached locally. See IMPLANT_METHODS.md.
"""
import argparse
import hashlib
import io
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np

DATA_ROOT = "https://orthoload.com/wp-content/files/add_file/"
TRIAL_PAGE = "https://orthoload.com/hip-joint-force-and-motion-capture-data/"
# Each pair is (ipsilateral single-leg stance, level walking), in source order.
TRIALS = [
    ("H1L_060511_1_24", "H1L_060511_1_47"), ("H2R_150811_1_24", "H2R_150811_2_98"),
    ("H3L_141111_2_75", "H3L_141111_1_28"), ("H4L_270112_1_36", "H4L_270112_2_80"),
    ("H5L_050412_1_37", "H5L_050412_1_75"), ("H6R_201112_1_36", "H6R_201112_2_107"),
    ("H7R_191112_1_38", "H7R_191112_1_47"), ("H8L_240413_1_36", "H8L_240413_1_48"),
    ("H9L_301013_1_29", "H9L_301013_1_62"), ("H10R_300114_1_32", "H10R_300114_1_82"),
]
# Arithmetic means of FORCE_PLATFORM:CORNERS from all 20 companion C3Ds, mm.
PLATE_CENTERS_MM = np.array([[200., 300.], [200., 903.]])
FRACTIONS = np.linspace(0, 1, 101)


def unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dot(a, b):
    return np.sum(a * b, axis=1)


def bounded_interp(t, v, query):
    """Respect missing markers; never bridge a marker gap longer than 100 ms."""
    finite = np.isfinite(v)
    if finite.sum() < 2:
        return np.full(query.shape, np.nan)
    times, values = t[finite], v[finite]
    out = np.interp(query, times, values, left=np.nan, right=np.nan)
    right = np.searchsorted(times, query).clip(1, len(times)-1)
    gap = times[right] - times[right-1]
    return np.where((gap > .1) & (times[right] != query) & (times[right-1] != query), np.nan, out)


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(a), int(b-1)) for a, b in edges.reshape(-1, 2)]


def foot_distance(cop_mm, heel_mm, toe_mm):
    """Distance from the COP to the 3-D estimated plantar centerline in mm."""
    heel = heel_mm.copy()
    toe = toe_mm.copy()
    # Heel/toe markers are mounted above the sole. Include foot elevation in assignment.
    heel[:, 2] -= 35.
    toe[:, 2] -= 20.
    ab = toe - heel
    s = np.clip(dot(cop_mm-heel, ab) / dot(ab, ab), 0., 1.)
    return np.linalg.norm(cop_mm-heel - s[:, None]*ab, axis=1)


def process(path, trial_id, activity, split):
    raw = path.read_bytes()
    stream = io.StringIO(raw.decode("latin1"))
    meta = {}
    for line in stream:
        if line.lower().startswith("time,frame,"):
            break
        if ":," in line:
            key, val = line.strip().split(":,", 1)
            meta[key] = val
    cols = [v.strip().upper() for v in line.split(",")]
    next(stream)  # The CSV header says N for Fz, but values are kN. See check below.
    a = np.loadtxt(stream, delimiter=",")
    t = a[:, 0]
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{trial_id}: timestamps out of order")
    bw = float(meta["Bodyweight"].rstrip(" N"))
    side = trial_id.split("_")[0][-1]
    other = "R" if side == "L" else "L"
    if meta["Patient code"] != trial_id.split("_")[0]:
        raise ValueError(f"{trial_id}: participant mismatch")

    def col(name):
        return a[:, cols.index(name)]

    def point(name):
        i = cols.index(name)
        return np.column_stack([bounded_interp(t, a[:, i+j], t) for j in range(3)])

    foot = {}
    for s in ("L", "R"):
        foot[s] = (point(s+"HEEL"), (point(s+"MET01") + point(s+"MET05"))/2)
    plates = []
    for plate, center in enumerate(PLATE_CENTERS_MM, start=1):
        def channel(name):
            return col(f"{name}{plate}")
        fz = channel("FZ") * 1000 / bw
        cop_xy = np.c_[channel("COPX"), channel("COPY")] + center
        # Mx [Nm] = local COPy [m] * Fz [N]. Check exported effective units.
        valid = (fz > .2) & (np.abs(channel("COPY")) > 20)
        if valid.any():
            predicted_mx = channel("COPY")[valid] * fz[valid]*bw/1000
            if np.nanmedian(np.abs(channel("MX")[valid] - predicted_mx)) > 1:
                raise ValueError(f"{trial_id}: plate {plate} unexpected Fz/CoP/Mx convention")
        cop = np.c_[cop_xy, np.zeros(len(t))]
        distances = {s:foot_distance(cop, *foot[s]) for s in foot}
        plates.append((fz, cop, np.c_[channel("FX"), channel("FY")]/bw, distances))

    l_as, r_as = point("LASIS"), point("RASIS")
    asis, psis = (l_as + r_as)/2, (point("LPSIS") + point("RPSIS"))/2
    width = np.linalg.norm(l_as-r_as, axis=1)
    depth = np.linalg.norm(asis-psis, axis=1)
    left = unit(l_as - r_as)
    anterior = unit(asis-psis - left*dot(asis-psis, left)[:, None])
    superior = unit(np.cross(anterior, left))
    outward = left * (1 if side == "L" else -1)
    hjc = (asis + outward*(.33*width + 7.3)[:, None]
           - anterior*(.24*depth + 9.9)[:, None] - superior*(.30*width + 10.9)[:, None])
    knee = (point(side+"MCO") + point(side+"LCO"))/2
    thigh = knee - hjc
    adduction = np.degrees(np.arctan2(-dot(thigh,outward), -dot(thigh,superior)))
    flexion = np.degrees(np.arctan2(dot(thigh,anterior), -dot(thigh,superior)))
    fres = bounded_interp(t, col("FRES")/bw, t)
    candidates = []
    for index,(fz, cop, force_xy, distances) in enumerate(plates):
        mask = fz > (.05 if activity == "walk" else .80)
        if activity == "stand":
            opposite_load = np.nan_to_num(plates[1-index][0])
            # Opposite foot elevation also distinguishes two feet on one platform.
            opposite_floor = np.minimum(foot[other][0][:, 2], foot[other][1][:, 2])
            stance_floor = np.minimum(foot[side][0][:, 2], foot[side][1][:, 2])
            mask &= (opposite_load < .10) & (opposite_floor > stance_floor + 25)
        for i, j in runs(mask):
            if t[j]-t[i] < (.25 if activity == "walk" else .40):
                continue
            sel = slice(i, j+1)
            own = float(np.nanmedian(distances[side][sel]))
            opposite = float(np.nanmedian(distances[other][sel]))
            if not np.isfinite(own) or own >= opposite or own > 150.:
                continue
            # Prioritize the longest ipsilateral bout. Selection uses no implant loads.
            candidates.append((t[j]-t[i], index, i, j, own, opposite))
    if not candidates:
        raise ValueError(f"{trial_id}: ipsilateral contact could not be established")
    _, plate, i, j, own, opposite = max(candidates)
    fz, cop, force_xy, _ = plates[plate]
    out_xy = unit(outward[:, :2])
    lateral_force = dot(force_xy, out_xy)
    medial_lever = dot(hjc[:,:2] - cop[:,:2], out_xy) / 1000
    moment = medial_lever*fz - hjc[:,2]/1000*lateral_force
    query = np.linspace(t[i], t[j], len(FRACTIONS))
    data = {
        "grfBw":fz, "adductionMomentBwM":moment, "medialLeverM":medial_lever,
        "hipAdductionDeg":adduction, "hipFlexionDeg":flexion,
        "horizontalBw":lateral_force, "measuredBw":fres,
    }
    frames = [{"phase":round(float(p), 2)} for p in FRACTIONS]
    for key, values in data.items():
        out = bounded_interp(t[i:j+1], values[i:j+1], query)
        if not np.isfinite(out).all():
            raise ValueError(f"{trial_id}: missing {key} within selected contact")
        for row, value in zip(frames, out):
            row[key] = round(float(value), 6)
    return {
        "id":trial_id, "subject":trial_id.split("_")[0], "activity":activity,
        "side":side, "split":split, "durationS":round(float(t[j]-t[i]), 5),
        "startS":float(t[i]), "endS":float(t[j]), "plate":plate+1,
        "bodyWeightN":bw, "copToFootMm":{"ipsilateral":round(own,1),"contralateral":round(opposite,1)},
        "file":path.name, "sourceUrl":DATA_ROOT+path.name,
        "sourceSha256":hashlib.sha256(raw).hexdigest(), "frames":frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"data"/"implants.json")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    trials = []
    for number, pair in enumerate(TRIALS, start=1):
        for trial_id, activity in zip(pair, ["stand", "walk"]):
            filename = trial_id.lower()+"_mov.csv"
            target = args.cache/filename
            if not target.exists():
                temp = target.with_suffix(".part")
                subprocess.run(["curl","--fail","--silent","--show-error","--location","--retry","2",
                                DATA_ROOT+filename,"--output",str(temp)], check=True)
                temp.replace(target)
            record = process(target, trial_id, activity, "calibration" if number <= 6 else "evaluation")
            print(trial_id, record["startS"], record["endS"], record["plate"], record["copToFootMm"], flush=True)
            trials.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    study = {"source":{"title":"OrthoLoad Hip Joint III","url":TRIAL_PAGE,
                       "accessed":date.today().isoformat(),
                       "attribution":"Bergmann G. and Damm P. (eds.), Julius Wolff Institute, Berlin Institute of Health at Charité Universitätsmedizin Berlin (2008), www.OrthoLoad.com.",
                       "terms":"https://orthoload.com/infos/legal-information/"}, "trials":trials}
    temporary = args.output.with_suffix(".part")
    temporary.write_text(json.dumps(study, separators=(",",":"), allow_nan=False)+"\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
