"""Summarize measured sagittal hip excursion in the Bertaux gait archive.

Input is the published subject-identifiable ARFF zip (Figshare file 26821445)
or a directory containing the M0 hip-flexion and vertical GRF values plus time channels.
Use --metadata with the authors' participant workbook (Figshare file 34177680).
Dependencies: Python 3.10+, NumPy and SciPy. See the sources in the JSON output.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from scipy.optimize import linear_sum_assignment


def member_text(source, member):
    if source.is_dir():
        return (source / Path(member).name).read_text(encoding="utf-8-sig")
    with zipfile.ZipFile(source) as archive:
        return archive.read(member).decode("utf-8-sig")


def trials(text):
    header, data = text.split("@DATA", 1)
    columns = [line.split()[1] for line in header.splitlines() if line.startswith("@ATTRIBUTE")]
    expected = [f"t{i}" for i in range(101)] + [
        "patients_class", "OAside_class", "trial_class", "cycle_class",
    ]
    if columns != expected:
        raise ValueError("Unexpected ARFF schema for the subject-identifiable release")
    vectors = {}
    row_count = 0
    for row in csv.reader(io.StringIO(data.strip())):
        if not row:
            continue
        row_count += 1
        subject, side, trial, _cycle = row[101:]
        key = (subject, side, trial)
        values = np.array([float(item) if item != "?" else np.nan for item in row[:101]])
        if key in vectors:
            # In this release the same exported waveform occurs once per cycle label.
            if not np.array_equal(vectors[key], values, equal_nan=True):
                raise ValueError(f"Different waveforms within {key}: revisit within-trial averaging")
        else:
            vectors[key] = values
    return vectors, row_count


def participants(path):
    with zipfile.ZipFile(path) as workbook:
        strings = [
            "".join(s.itertext()) for s in ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
        ]
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    people = {}
    for row in sheet.findall(".//{*}row"):
        cells = {}
        for cell in row:
            value = cell.findtext("{*}v")
            if value is not None:
                column = re.sub(r"\d+", "", cell.get("r"))
                cells[column] = strings[int(value)] if cell.get("t") == "s" else value
        group = cells.get("A")
        if group not in ("HEA", "HOA") or (group == "HOA" and cells["C"] != "M0"):
            continue
        people[group + str(int(cells["B"]))] = {
            "age": float(cells["H"]), "sex": cells["G"], "side": cells["F"],
            "disturbance": cells["E"] if group == "HOA" else None,
        }
    return people


def contact_indices(force):
    if not np.all(np.isfinite(force)) or force.max() < 60:
        return None
    # Use a fraction of the trial peak to avoid dependence on force magnitude.
    contact = np.flatnonzero(force > 0.05 * force.max())
    if not 40 <= contact.size <= 80:
        return None
    if contact[0] == 0 or contact[-1] == 100 or np.max(np.diff(contact)) > 1:
        return None
    return contact


def measure(source, people):
    values = {name: defaultdict(lambda: defaultdict(list)) for name in ("cycle", "loadedStance")}
    counts = {}
    for group in ("HOA", "HEA"):
        audit = Counter()
        for side in ("L", "R"):
            angle, nrows = trials(member_text(source, f"Angles/{group}_M0_{side}HipAngles_X.arff"))
            grf, _ = trials(member_text(source, f"CoM & GRF/{group}_M0_{side}NormalisedGRF_Z.arff"))
            angle_time, _ = trials(member_text(source, f"Angles/{group}_M0_{side}HipAngles_time.arff"))
            force_time, _ = trials(member_text(source, f"CoM & GRF/{group}_M0_{side}NormalisedGRF_time.arff"))
            if angle.keys() != angle_time.keys() or grf.keys() != force_time.keys():
                raise ValueError(f"Missing time channels in {group} {side}")
            audit["sourceAngleRowsBothSides"] += nrows
            audit["uniqueAngleTrialSidesBeforeSideSelection"] += len(angle)
            for key, curve in angle.items():
                person, oa_side, _trial = key
                if people[person]["side"] != oa_side:
                    raise ValueError(f"Inconsistent OA side for {person}")
                if group == "HOA" and side != oa_side:
                    continue
                if not np.all(np.isfinite(curve)):
                    audit["incompleteSelectedAngleTraces"] += 1
                    continue
                values["cycle"][person][side].append(float(np.ptp(curve)))
                audit["usableWholeCycleTrialSides"] += 1
                if key not in grf:
                    audit["missingSelectedForceTrials"] += 1
                    continue
                if not np.array_equal(angle_time[key], force_time[key]):
                    raise ValueError(f"Time-grid mismatch in {key}")
                indices = contact_indices(grf[key])
                if indices is None:
                    audit["incompleteSelectedContacts"] += 1
                    continue
                values["loadedStance"][person][side].append(float(np.ptp(curve[indices])))
                audit["usableLoadedStanceTrialSides"] += 1
        counts[group] = dict(audit)
    by_person = {}
    for measure_name, traces in values.items():
        # Equal limb weights in controls. Each participant receives equal cohort weight.
        by_person[measure_name] = {p: float(np.mean([np.mean(v) for v in sides.values()]))
                                   for p, sides in traces.items()}
    trial_counts = {name: {p: sum(len(trials) for trials in sides.values())
                           for p, sides in per_person.items()} for name, per_person in values.items()}
    return by_person, counts, trial_counts


def matched_pairs(oa, controls, people):
    # Maximize same-sex pairs within three years first, minimize total age difference second.
    costs = np.full((len(oa), len(controls) + len(oa)), 1000.)
    for i, left in enumerate(oa):
        for j, right in enumerate(controls):
            age_diff = abs(people[left]["age"] - people[right]["age"])
            eligible = people[left]["sex"] == people[right]["sex"] and age_diff <= 3
            costs[i, j] = age_diff if eligible else 1e6
    rows, cols = linear_sum_assignment(costs)
    return [(oa[i], controls[j]) for i, j in zip(rows, cols) if costs[i, j] <= 3]


def ci(array):
    return [round(float(q), 3) for q in np.quantile(array, [0.025, 0.975])]


def group_summary(v, rng):
    bootstrap = rng.choice(v, (10000, len(v)), replace=True).mean(axis=1)
    return {"participants": len(v), "meanDeg": round(float(v.mean()), 3),
            "sdDeg": round(float(v.std(ddof=1)), 3), "mean95CI": ci(bootstrap)}


def observed_gait(cycle, people):
    labels = {"D0": "No marked asymmetry", "D1": "Trendelenburg", "D2": "Avoidance",
              "D3": "Duchenne", "D4": "Unclassified"}
    codes = defaultdict(list)
    for p, excursion in cycle.items():
        if p.startswith("HOA"):
            code = people[p]["disturbance"]
            if code not in labels:
                raise ValueError(f"Unknown preoperative video classification for {p}: {code}")
            codes[code].append(excursion)
    return {
        "method": "Two experienced physicians rated frontal and sagittal gait videos. "
                  "The public metadata give one consensus code per patient and visit.",
        "scope": "Labels describe visible gait patterns within the preoperative OA cohort. "
                 "Concurrent gluteal tendinopathy was not assessed in the archive.",
        "groups": [{"code": k, "label": v, "participants": len(codes[k]),
                    "meanCycleExcursionDeg": round(float(np.mean(codes[k])), 3) if codes[k] else None}
                   for k, v in labels.items()],
    }


def summarize(values, people):
    rng = np.random.default_rng(92219)
    ids = {group: sorted(p for p in values if p.startswith(group)) for group in ("HEA", "HOA")}
    pairs = matched_pairs(ids["HOA"], ids["HEA"], people)
    groups = {g: group_summary(np.array([values[p] for p in persons]), rng)
              for g, persons in ids.items()}
    delta = np.array([values[control] - values[oa] for oa, control in pairs])
    groups["ageSexMatched"] = {
        "pairs": len(pairs),
        "meanAbsoluteAgeDifferenceYears": round(float(np.mean([
            abs(people[a]["age"] - people[b]["age"]) for a, b in pairs])), 3),
        "oaMeanDeg": round(float(np.mean([values[a] for a, b in pairs])), 3),
        "controlMeanDeg": round(float(np.mean([values[b] for a, b in pairs])), 3),
        "controlMinusOaDeg": round(float(delta.mean()), 3),
        "pairedDifference95CI": ci(rng.choice(delta, (10000, len(delta))).mean(axis=1)),
    }
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/cohort.json")
    args = parser.parse_args()
    people = participants(args.metadata)
    values, counts, trial_counts = measure(args.input, people)
    result = {
        "title": "Measured hip excursion in unilateral hip osteoarthritis",
        "sources": {
            "attribution": "Bertaux et al., Scientific Data (2022). Public gait data and metadata, CC BY 4.0.",
            "data": "https://doi.org/10.6084/m9.figshare.13656233.v4",
            "arffZip": "https://ndownloader.figshare.com/files/26821445",
            "demographics": "https://ndownloader.figshare.com/files/34177680",
            "metadataDoi": "https://doi.org/10.6084/m9.figshare.13655975.v10",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "figshareMd5": {"arffZip": "22c7820b17c4828b926eee6b3ef28804",
                            "demographics": "5e1aa0dd687aa6afe00a57f47325916d"},
            "descriptor": "https://doi.org/10.1038/s41597-022-01483-3",
            "comparison": "https://doi.org/10.1016/j.gaitpost.2018.02.011",
        },
        "method": [
            "Plug-in Gait hip flexion/extension (HipAngles_X) range, max minus min in degrees. "
            "HOA includes the affected side before surgery. HEA includes asymptomatic controls.",
            "All 101-point hip angle waveforms under multiple cycle labels in a trial-side were identical. "
            "Each distinct trial-side counts once. Average trials within side, available sides within "
            "person, then people within cohort.",
            "Cycle uses the full normalized gait cycle. Loaded stance requires a matched ipsilateral GRF "
            "trace on the identical 101-sample time grid with peak at least 60 in the archive's normalized units and one interior contiguous "
            "contact above 5% of that peak lasting 40–80% of the cycle. Phase edges and partial contacts "
            "are excluded. Loaded-stance excursion omits the brief low-force ends of stance.",
            "Matching pairs a control with a patient of the same sex and age within three years without "
            "replacement. Among maximum-cardinality matchings, minimize total absolute age difference. "
            "10,000 seeded percentile bootstrap resamples of people, or fixed pairs for matched contrasts.",
            "Controls and patients differ in other characteristics, including body mass and walking speed. "
            "Requiring a complete interior contact preferentially selects the limb whose stance falls inside the exported stride. "
            "Force completeness restricts the loaded-stance subset. Findings concern preoperative OA and "
            "asymptomatic gait, without patient-level gluteal tendinopathy labels.",
        ],
        "audit": counts,
        "observedGait": observed_gait(values["cycle"], people),
        "people": [
            {
                "id": p, "group": p[:3], "side": people[p]["side"] if p.startswith("HOA") else "both",
                "age": people[p]["age"], "sex": people[p]["sex"],
                "disturbance": people[p]["disturbance"],
                "cycleDeg": round(values["cycle"][p], 4),
                "loadedStanceDeg": (round(values["loadedStance"][p], 4)
                                    if p in values["loadedStance"] else None),
                "cycleTrialSides": trial_counts["cycle"][p],
                "stanceTrialSides": trial_counts["loadedStance"].get(p, 0),
            } for p in sorted(values["cycle"])
        ],
        "cycle": summarize(values["cycle"], people),
        "loadedStance": summarize(values["loadedStance"], people),
        "publishedStance": {
            "source": "Allison et al., Gait & Posture (2018), Table 2",
            "osteoarthritis": {"participants": 73, "meanDeg": 33.7, "sdDeg": 7.5},
            "glutealTendinopathy": {"participants": 60, "meanDeg": 44.9, "sdDeg": 5.3},
            "reportedDifferenceDeg": 11.3, "difference95CI": [9.1, 13.5],
            "scope": "Independent symptomatic cohorts and a different gait-lab protocol. "
                     "The Bertaux force-gated interval approximates stance. Patient-level classification "
                     "requires independently labeled GT and OA recordings.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
