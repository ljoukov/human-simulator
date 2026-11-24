"""Compare frontal hip excursion in two published walking cohorts.

Run with the original Robinson/Fearon GTPS workbook, Bertaux subject-ID ARFF
ZIP (or extracted members) and Bertaux demographic workbook. See GTPS.md.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
import zipfile

import numpy as np

from prepare_cohort import ci, contact_indices, group_summary, member_text, participants, trials


WORKBOOK_SHA256 = "95f2b5fa6ac78367eddc4ce294c235e62f49a21169df70a9f8e2bed35a5470d6"
PERCENT_STANCE = np.arange(101)


def read_gtps(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != WORKBOOK_SHA256:
        raise ValueError("Pass the original version 1 Final Data workbook")
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(item.itertext()) for item in
                   ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = {}
    for row in sheet.findall(".//{*}row"):
        number = int(row.get("r"))
        if number > 103:
            break
        cells = {}
        for cell in row:
            value = cell.findtext("{*}v")
            if value is not None:
                column = cell.get("r").removesuffix(str(number))
                cells[column] = strings[int(value)] if cell.get("t") == "s" else value
        rows[number] = cells
    if not all(int(rows[t + 3]["A"]) == t for t in PERCENT_STANCE):
        raise ValueError("Workbook phase column must span 0–100% stance")
    curves = {}
    for col, variable in rows[2].items():
        if variable != "RHipAddAbd":
            continue
        pid = f"GTPS{int(rows[1][col])}"
        curve = np.array([float(rows[t + 3][col]) for t in PERCENT_STANCE])
        if not np.all(np.isfinite(curve)) or not np.ptp(curve):
            raise ValueError(f"Invalid GTPS frontal hip series for {pid}")
        if pid in curves or any(np.array_equal(curve, previous) for previous in curves.values()):
            raise ValueError(f"Duplicate GTPS identifier or complete curve for {pid}")
        curves[pid] = curve
    return curves


def read_oa(source, metadata):
    patients = participants(metadata)
    waveforms = defaultdict(list)
    audit = Counter()
    for side in ("L", "R"):
        stem = f"HOA_M0_{side}"
        angle, _ = trials(member_text(source, f"Angles/{stem}HipAngles_Y.arff"))
        force, _ = trials(member_text(source, f"CoM & GRF/{stem}NormalisedGRF_Z.arff"))
        angle_time, _ = trials(member_text(source, f"Angles/{stem}HipAngles_time.arff"))
        force_time, _ = trials(member_text(source, f"CoM & GRF/{stem}NormalisedGRF_time.arff"))
        for key, curve in angle.items():
            pid, affected, _ = key
            if patients[pid]["side"] != affected:
                raise ValueError(f"OA side mismatch for {pid}")
            if affected != side or patients[pid]["sex"] != "Female":
                continue
            audit["affectedSideFemaleTrials"] += 1
            if not np.all(np.isfinite(curve)) or key not in force:
                continue
            if not np.array_equal(angle_time[key], force_time[key]):
                raise ValueError(f"Force/angle time mismatch for {key}")
            indices = contact_indices(force[key])
            if indices is None:
                continue
            normalized = np.linspace(indices[0], indices[-1], 101)
            waveforms[pid].append(np.interp(normalized, indices, curve[indices]))
            audit["usableContacts"] += 1
    audit["eligibleWomen"] = sum(p.startswith("HOA") and v["sex"] == "Female" for p, v in patients.items())
    audit["includedWomen"] = len(waveforms)
    return ({p: np.mean(curves, axis=0) for p, curves in waveforms.items()},
            {p: len(curves) for p, curves in waveforms.items()}, patients, dict(audit))


def excursions(curves, trim_percent):
    window = slice(trim_percent, 101 - trim_percent)
    return {pid: float(np.ptp(curve[window])) for pid, curve in curves.items()}


def describe(v, rng):
    return {**group_summary(v, rng), "medianDeg": round(float(np.median(v)), 3),
            "p10p90Deg": np.quantile(v, [.1, .9]).round(3).tolist(),
            "minMaxDeg": [round(float(v.min()), 3), round(float(v.max()), 3)]}


def comparison(gtps, oa):
    rng = np.random.default_rng(661291)
    gv, ov = [np.array(list(group.values())) for group in (gtps, oa)]
    gs, os = describe(gv, rng), describe(ov, rng)
    delta = rng.choice(gv, (10000, len(gv))).mean(axis=1) - rng.choice(ov, (10000, len(ov))).mean(axis=1)
    low = max(np.quantile(gv, .1), np.quantile(ov, .1))
    high = min(np.quantile(gv, .9), np.quantile(ov, .9))
    overlap = {"sharedP10p90Deg": [round(float(low), 3), round(float(high), 3)] if low <= high else None,
               "gtpsInside": int(np.count_nonzero((gv >= low) & (gv <= high))),
               "oaInside": int(np.count_nonzero((ov >= low) & (ov <= high)))}
    drop_one = (gv.sum() - gv) / (len(gv) - 1) - ov.mean()
    return {
        "gtps": gs, "oa": os,
        "comparison": {"gtpsMinusOaDeg": round(float(gv.mean() - ov.mean()), 3),
                       "difference95CI": ci(delta), "overlap": overlap,
                       "omitAnyOneGtpsDifferenceRangeDeg": [round(float(x), 3)
                                                           for x in [drop_one.min(), drop_one.max()]]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtps", required=True, type=Path)
    parser.add_argument("--oa", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/gtps.json")
    args = parser.parse_args()
    gtps_curves = read_gtps(args.gtps)
    oa_curves, counts, patients, audit = read_oa(args.oa, args.metadata)
    gtps, oa = excursions(gtps_curves, 20), excursions(oa_curves, 20)
    primary = comparison(gtps, oa)
    records = [{"id": p, "group": name, "excursionDeg": round(values[p], 4)}
               for name, values in (("GTPS", gtps), ("HOA", oa)) for p in sorted(values)]
    for record in records:
        p = record["id"]
        if p in oa:
            record.update(age=patients[p]["age"], side=patients[p]["side"], contacts=counts[p])
    ages = np.array([patients[p]["age"] for p in oa])
    data = {
        "title": "Frontal hip excursion during stance in two symptomatic cohorts",
        "measure": "Adduction/abduction range over 20–80% of normalized stance (degrees)",
        "sources": {"gtpsDataset": "https://doi.org/10.17632/66mcwyk854.1",
                    "gtpsPaper": "https://doi.org/10.1016/j.gaitpost.2019.02.031",
                    "gtpsWorkbookSha256": WORKBOOK_SHA256,
                    "oaDataset": "https://doi.org/10.6084/m9.figshare.13656233.v4",
                    "oaMetadata": "https://doi.org/10.6084/m9.figshare.13655975.v10",
                    "oaPaper": "https://doi.org/10.1038/s41597-022-01483-3",
                    "license": "https://creativecommons.org/licenses/by/4.0/"},
        "method": [
            "Untaped baseline GTPS workbook, RHipAddAbd, one 101-point affected-foot stance trace per numeric ID.",
            "Female preoperative OA patients, affected side only. Deduplicate repeated angle curves within trials. "
            "Require a complete interior force contact on the same time grid. Interpolate each loaded stance "
            "to 101 points and average across contacts within a participant before measuring excursion.",
            "Use max minus min within 20–80% of normalized stance. This omits contact edges and is invariant "
            "to a constant angle offset or left/right sign reversal. Repeat over 10–90% and 0–100%.",
            "10,000 independent within-cohort participant bootstrap resamples. Shared central range is "
            "the intersection of the two empirical 10th–90th percentile intervals."
        ],
        "limitations": [
            "Diagnosis and laboratory coincide. The GTPS site used the UWA model and the OA site used "
            "Plug-in Gait. Differences in coordinate systems, force gating, recruitment and severity remain.",
            "The GTPS workbook supplies 51 unique baseline IDs and curves. The publication reports 50 women. "
            "All complete records are retained and single-record omission sensitivity is shown.",
            "Both cohorts walked barefoot at their preferred speed. Individual GTPS ages and walking speeds "
            "are unavailable in the gait workbook, precluding individual age and speed matching.",
            "The observations describe selected women and group overlap. Coexisting GTPS in OA is unknown "
            "and cross-site differences cannot establish clinical classification performance."
        ],
        "audit": {"gtpsWorkbookRecords": len(gtps), "gtpsPaperParticipants": 50,
                  "gtpsAgePublished": {"meanYears": 57.4, "sdYears": 13.8},
                  "oa": {**audit, "meanAgeYears": round(float(np.mean(ages)), 2),
                         "sdAgeYears": round(float(np.std(ages, ddof=1)), 2)}},
        **primary, "people": records,
        "sensitivity": {f"{edge}to{100 - edge}": comparison(excursions(gtps_curves, edge),
                                                           excursions(oa_curves, edge))
                        for edge in (0, 10)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
