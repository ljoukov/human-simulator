#!/usr/bin/env python3
"""Convert Harba et al.'s released Grand Challenge measurements to stance frames."""

from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen
import io
import json

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

COMMIT = "00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8"
REPOSITORY = "https://github.com/gilserrancoli/ReducePoly_forMSK"
SOURCE_PATH = "ExampleTrackingSimulations/OpenSimModel/subject_GC"
RAW = f"https://raw.githubusercontent.com/gilserrancoli/ReducePoly_forMSK/{COMMIT}"
MASS_KG = 65.0
HEIGHT_M = 1.66
GRAVITY = 9.80665
LBF_TO_N = 4.4482216153
CONTACT_THRESHOLD_N = 20
STANCE_POINTS = 101

# Right plates are 2 and 3 overground, and belt 2 on the treadmill (getGRF_GC.m).
# The optional time window selects one complete, steady treadmill contact near
# the study's published tracking interval. Overground recordings contain two.
RECORDINGS = (
    ("ngait_og5", "normal", (2, 3), None),
    ("bouncy7", "bouncy", (2, 3), None),
    ("mtpgait3", "forefoot_strike", (2, 3), None),
    ("ngait_tm_fast1", "fast_treadmill", (2,), (5.1, 6.1)),
)


def source_table(directory, prefix, trial, key):
    filename = f"{prefix}/{prefix}_{trial}.mat"
    if directory:
        contents = directory / filename
    else:
        with urlopen(f"{RAW}/{SOURCE_PATH}/{filename}", timeout=30) as response:
            contents = io.BytesIO(response.read())
    table = loadmat(contents, simplify_cells=True)[key]
    return dict(zip(table["colheaders"], table["data"].T, strict=True))


def contact_intervals(time, vertical_force):
    """Yield complete contacts of at least 0.3 s using interpolated threshold crossings."""
    inside = vertical_force > CONTACT_THRESHOLD_N
    changes = np.diff(np.r_[False, inside, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
        if start == 0 or stop == len(time) or time[stop] - time[start] < 0.3:
            continue
        onset = np.interp(
            CONTACT_THRESHOLD_N, vertical_force[start - 1:start + 1], time[start - 1:start + 1]
        )
        offset = np.interp(
            CONTACT_THRESHOLD_N, vertical_force[stop - 1:stop + 1][::-1], time[stop - 1:stop + 1][::-1]
        )
        yield float(onset), float(offset)


def sample(time, values, query, signal):
    if not np.all(np.isfinite(values)) or not np.all(np.diff(time) > 0):
        raise ValueError(f"Invalid signal {signal}")
    if query[0] < time[0] or query[-1] > time[-1]:
        raise ValueError(f"{signal} does not cover the full stance")
    return np.interp(query, time, values)


def implant_forces(implant):
    time = implant["Time(sec)"]
    # Harba et al.'s get_KCF.m uses a second-order Butterworth in each direction.
    numerator, denominator = butter(2, 6.0, fs=1 / np.diff(time).mean())
    medial = (implant["PM"] + implant["AM"]) * LBF_TO_N
    lateral = (implant["AL"] + implant["PL"]) * LBF_TO_N
    # Match MATLAB's endpoint reflection length for the same transfer function.
    padding = 3 * (len(numerator) - 1)
    return (
        filtfilt(numerator, denominator, medial, padlen=padding),
        filtfilt(numerator, denominator, lateral, padlen=padding),
    )


def recording_stances(directory, trial, condition, right_plates, selection):
    grf = source_table(directory, "GRF", trial, "GRFall")
    dynamics = source_table(directory, "ID", trial, "IDall")
    kinematics = source_table(directory, "IK", trial, "Qsall")
    implant = source_table(directory, "KCF", trial, "dataKCF")
    medial, lateral = implant_forces(implant)
    normalized_stance = np.linspace(0, 100, STANCE_POINTS)
    body_weight = MASS_KG * GRAVITY

    for index, plate in enumerate(right_plates, start=1):
        vertical_force = grf[f"grforce{plate}_fy"]
        intervals = list(contact_intervals(grf["time"], vertical_force))
        if selection:
            intervals = [(start, end) for start, end in intervals if selection[0] <= start < end <= selection[1]]
        if len(intervals) != 1:
            raise ValueError(f"Expected one complete right stance in {trial} force plate {plate}, got {intervals}")
        start, end = intervals[0]
        time = np.linspace(start, end, STANCE_POINTS)
        signals = {
            "stancePct": normalized_stance,
            "kamNmKg": -sample(dynamics["time"], dynamics["knee_adduction_moment"], time, "adduction moment") / MASS_KG,
            "kfmNmKg": sample(dynamics["time"], dynamics["knee_flexion_moment"], time, "flexion moment") / MASS_KG,
            "grfBw": sample(grf["time"], vertical_force, time, "vertical force") / body_weight,
            "compressionBw": sample(dynamics["time"], dynamics["knee_ty_force"], time, "net axial force") / body_weight,
            "kneeFlexionDeg": -sample(kinematics["time"], kinematics["knee_flexion"], time, "knee angle"),
            "medialBw": sample(implant["Time(sec)"], medial, time, "medial force") / body_weight,
            "lateralBw": sample(implant["Time(sec)"], lateral, time, "lateral force") / body_weight,
        }
        signals["totalBw"] = signals["medialBw"] + signals["lateralBw"]
        yield {
            "id": f"{trial}-stance-{index}",
            "subject": "GC4-JW",
            "condition": condition,
            "recording": trial,
            "stanceIndex": index,
            "massKg": MASS_KG,
            "heightM": HEIGHT_M,
            "stanceSeconds": round(end - start, 6),
            "stanceStartSeconds": round(start, 6),
            "forcePlate": plate,
            "role": "calibration" if condition == "normal" and index == 1 else "evaluation",
            "frames": [
                {key: round(float(value), 6) for key, value in zip(signals, frame, strict=True)}
                for frame in zip(*signals.values(), strict=True)
            ],
        }


def build_dataset(directory):
    return {
        "source": {
            "title": "Grand Challenge knee implant measurements republished by Harba et al.",
            "url": f"{REPOSITORY}/tree/{COMMIT}/{SOURCE_PATH}",
            "citation": "Harba, Badia & Serrancolí (2026), DOI 10.1002/cnm.70143; Fregly et al. (2012), DOI 10.1002/jor.22023.",
            "license": "MIT (ReducePoly_forMSK); see HARBA-LICENSE.txt.",
            "kind": "trial",
            "recordedSide": "right",
            "notes": [
                "One participant with a force-measuring right knee replacement, 65 kg and 1.66 m as reported by Harba et al.",
                "Four original recordings yield seven right stances. Two stances per overground recording are repeated measurements from the same walking pass.",
                "The treadmill recording varies both speed and surface. Its direct comparison with natural overground walking cannot isolate either effect.",
                "Each stance is delimited by 20 N crossings on the measured right vertical ground-reaction trace and resampled linearly to 101 points.",
                "Medial (PM + AM) and lateral (AL + PL) implant signals use the source's compartment sums, lbf-to-newton conversion, and 6 Hz forward-backward filter. Total is their compressive sum.",
                "Knee angle and moments come from source inverse kinematics and inverse dynamics. The right knee coordinate is negative in flexion, so its positive internal extension moment corresponds to positive external flexion. The reported external adduction moment reverses the internal adduction sign.",
                "compressionBw is the signed net proximal force from OpenSim inverse dynamics (knee_ty_force), whereas grfBw is the vertical force plate measurement. They are separate inputs.",
                "One normal stance is labeled for calibration. All other stances are evaluation data; the second normal stance is from the same recording.",
            ],
        },
        "trials": [
            stance
            for recording in RECORDINGS
            for stance in recording_stances(directory, *recording)
        ],
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, help=f"Local copy of upstream {SOURCE_PATH}")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/gait.json")
    args = parser.parse_args()
    data = build_dataset(args.source_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(data['trials'])} recorded stances to {args.output}")


if __name__ == "__main__":
    main()
