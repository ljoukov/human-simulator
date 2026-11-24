"""Check whether two walking-angle excursions help estimate point-in-time pain in the Robinson GTPS deposit.

Run with NumPy and SciPy:
python3 hip-study/tools/gait_pain_check.py GAIT_WORKBOOK.xlsx PAIN_WORKBOOK.xlsx
"""
import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.stats import rankdata


GAIT_SHA = "95f2b5fa6ac78367eddc4ce294c235e62f49a21169df70a9f8e2bed35a5470d6"
PAIN_SHA = "061deeaede08f825e014d95abfc76bfc5b21d1daeb0e42cbeaee5b3293642f0f"
CHANNELS = ("RHipAddAbd", "ThorLatFlex")
BOOTSTRAP_SEED = 20261122


def read_rows(archive, sheet_number, last_row):
    """Read the pinned worksheet's cached cell values, retaining column keys across empty cells."""
    strings = ["".join(item.itertext()) for item in ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))]
    sheet = ElementTree.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    rows = {}
    for row in sheet.findall(".//{*}row"):
        row_number = int(row.attrib["r"])
        if row_number > last_row:
            break
        values = {}
        for cell in row:
            value = cell.findtext("{*}v")
            if value is not None:
                key = cell.attrib["r"].removesuffix(str(row_number))
                values[key] = strings[int(value)] if cell.get("t") == "s" else float(value)
        rows[row_number] = values
    return rows


def excursions(rows):
    if [rows[phase + 3]["A"] for phase in range(101)] != list(range(101)):
        raise ValueError("Expected 0–100% of stance in 101 samples")
    people = {}
    for col, channel in rows[2].items():
        if channel not in CHANNELS or not isinstance(rows[1].get(col), float):
            continue
        person = int(rows[1][col])
        curve = np.array([rows[phase + 3].get(col, np.nan) for phase in range(101)], dtype=float)
        if np.isfinite(curve).all():
            measured = people.setdefault(person, {})
            if channel in measured:
                raise ValueError(f"Duplicate participant/channel in gait worksheet: {person}, {channel}")
            measured[channel] = np.ptp(curve[20:81])
    return {person: np.array([values[channel] for channel in CHANNELS])
            for person, values in people.items() if all(channel in values for channel in CHANNELS)}


def read_pain(path):
    with ZipFile(path) as archive:
        rows = read_rows(archive, 1, 52)
    headers = {"A": "part_ID", "B": "baseline", "C": "active", "D": "postactivetape",
               "F": "shame", "G": "postshamtape"}
    if any(rows[1][column] != label for column, label in headers.items()):
        raise ValueError("Unexpected pain workbook layout")
    # The second block on this sheet repeats participants with complete pain triplets.
    people = {}
    for row in range(2, 53):
        values = rows[row]
        person = int(values["A"])
        if person in people or (values["C"], values["F"]) != ("A", "B"):
            raise ValueError("Duplicate participant or unexpected active/sham mapping")
        people[person] = [values.get(c) if isinstance(values.get(c), float) else None for c in ("B", "D", "G")]
    return people


def rank_correlation(x, y):
    x = rankdata(x, axis=-1)
    y = rankdata(y, axis=-1)
    x -= x.mean(axis=-1, keepdims=True)
    y -= y.mean(axis=-1, keepdims=True)
    return np.sum(x * y, axis=-1) / np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))


def summarize(rows):
    values = np.array(rows)
    y, hip, trunk = values.T
    n = len(y)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(5000, n))
    correlations = {}
    for label, x in zip(CHANNELS, (hip, trunk)):
        interval = np.quantile(rank_correlation(x[draws], y[draws]), [0.025, 0.975])
        correlations[label] = {"rho": float(rank_correlation(x, y)), "bootstrap95": interval.tolist()}

    errors = {}
    for label, columns in (("meanOnly", []), ("hip", [hip]), ("hipAndTrunk", [hip, trunk])):
        design = np.column_stack([np.ones(n), *columns])
        predictions = np.empty(n)
        for held in range(n):
            train = np.arange(n) != held
            weights = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
            predictions[held] = design[held] @ weights
        residual = y - predictions
        errors[label] = {"maePainPoints": float(np.abs(residual).mean()),
                         "rmsePainPoints": float(np.sqrt(np.mean(residual ** 2)))}
    return {"people": n, "meanPainPoints": float(y.mean()), "rangePainPoints": [float(y.min()), float(y.max())],
            "spearman": correlations, "leaveOnePersonOut": errors}


def run(gait_path, pain_path):
    for path, expected in ((gait_path, GAIT_SHA), (pain_path, PAIN_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Use the original version 1 workbook: {path.name}")
    with ZipFile(gait_path) as archive:
        baseline, active, sham = [excursions(read_rows(archive, sheet, 103)) for sheet in (1, 2, 3)]
    pain = read_pain(pain_path)
    if set(baseline) != set(pain):
        raise ValueError("Baseline participants differ between the two original workbooks")
    baseline_rows, paired_rows = [], []
    numeric_triples = 0
    numeric_pairs = 0
    for person, (base_nrs, active_nrs, sham_nrs) in pain.items():
        if base_nrs is not None:
            baseline_rows.append([base_nrs, *baseline[person]])
        if active_nrs is not None and sham_nrs is not None:
            numeric_pairs += 1
            numeric_triples += int(base_nrs is not None)
            if person in active and person in sham:
                paired_rows.append([active_nrs - sham_nrs, *(active[person] - sham[person])])
    return {
        "study": "Exploratory same-person walking kinematics and point-in-time hip pain in women with GTPS",
        "sources": {"dataset": "https://doi.org/10.17632/66mcwyk854.1",
                    "paper": "https://doi.org/10.1016/j.gaitpost.2019.02.031",
                    "gaitWorkbook": {"fileId": "b26da87d-3bca-49a9-b7e2-b3655f3dff71", "sha256": GAIT_SHA},
                    "painWorkbook": {"fileId": "3568f447-3fc5-469a-b915-61b5baa3366b", "sha256": PAIN_SHA}},
        "methods": [
            "Join each gait worksheet's numeric participant headers to part_ID in the pain worksheet's first block.",
            "Optical affected-foot walking stance. For hip adduction/abduction and trunk lateral flexion, take maximum minus minimum in degrees from 20% through 80% inclusive. A complete 101-point curve is required.",
            "Hip pain was rated at baseline and immediately after each taped walk on a 0–10 scale. Keep numeric responses, including zero. Exclude missing values and an interval-valued post-sham response.",
            "Fit a training mean, linear regression on hip excursion, and linear regression on hip and trunk excursions with an intercept. Hold out one person and refit on every fold. Keep predictions continuous and unclipped.",
            "Spearman rank correlations use average ranks for ties. Percentile intervals resample participant IDs 5,000 times (seed 20261122) and recalculate ranks in each resample.",
            "Sensitivity analysis subtracts each person's sham values from their active values for both excursions and pain. The pain sheet identifies Tape A as active and Tape B as sham for every ID. Crossover order was randomized.",
        ],
        "audit": {"randomizedPeopleInPaper": 50, "uniqueMatchingBaselineIdsInWorkbooks": len(pain),
                  "missingBaselinePain": len(pain) - len(baseline_rows),
                  "bothNumericPostTapePainRatings": numeric_pairs, "allThreeNumericPainRatings": numeric_triples,
                  "numericPainPairsMissingOneOrMoreGaitFeatures": numeric_pairs - len(paired_rows)},
        "baselineWalking": summarize(baseline_rows),
        "activeMinusShamSensitivity": summarize(paired_rows),
        "interpretation": "In this cohort, adding trunk excursion to a linear model of hip excursion did not improve held-out baseline pain error. The paired crossover change analysis also favored the training mean.",
        "limitations": [
            "The deposit contains 51 participant IDs while the paper reports 50 randomized. The reason is unknown.",
            "Pain refers to the recording condition. It has no within-stride timing. Only walking was measured, and the data contain one clinical group.",
            "The analysis uses two optical angular excursions. It does not test the loaded-stance mechanism classifier, direct tissue force, or symptoms in another population. Correlations do not establish a cause of pain.",
            "The crossover comparison uses complete pairs. Tape can affect pain through several mechanisms. No adjustment for period or carryover was possible in these workbooks.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gait", type=Path)
    parser.add_argument("pain", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/gait-pain-check.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.gait, args.pain), indent=2, allow_nan=False) + "\n")
