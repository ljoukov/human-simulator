#!/usr/bin/env python3
"""Summarize Open Knee(s) oks001 Tekscan pressure movies at fixed loads."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import json
import math
import re


SOURCE = (
    "http://archive.simtk.org/oks/MRI_JointMech/joint_mechanics/"
    "joint_mechanics-oks001.zip"
)
DIRECTORY = "joint_mechanics-oks001/PatellofemoralJoint/ContactPressure/"
ANGLES = (0, 15, 30, 45, 60)
LOADS = (20, 100, 200, 300, 400, 500, 600)
CONDITION = re.compile(r"oks001_intact_(\d+)_(\d+)(?:_(\d+))?_M\.csv")
FRAME_HEADER = re.compile(r"Frame (\d+),")


def parse_movie(data):
    header, samples = data.decode("cp1252").split("ASCII_DATA @@", 1)
    metadata = dict(line.split(maxsplit=1) for line in header.splitlines() if line.strip())
    rows, columns = int(metadata["ROWS"]), int(metadata["COLS"])
    if metadata["UNITS"] != "MPa" or metadata["SENSOR_TYPE"] != "5051":
        raise ValueError("Expected calibrated MPa output from a Tekscan 5051 sensor")
    if (metadata["MIRROR_ROW"], metadata["MIRROR_COL"]) != ("0", "0"):
        raise ValueError("The source sensor axes were mirrored")

    lines = samples.splitlines()
    frames = []
    numbers = []
    for index, line in enumerate(lines):
        match = FRAME_HEADER.match(line)
        if match is None:
            continue
        frame_rows = [list(map(float, row.split(","))) for row in lines[index + 1:index + 1 + rows]]
        if len(frame_rows) != rows or any(len(row) != columns for row in frame_rows):
            raise ValueError(f"Frame {match[1]} does not match the declared sensor shape")
        frame = [value for row in frame_rows for value in row]
        if any(not math.isfinite(value) or value < 0 for value in frame):
            raise ValueError(f"Frame {match[1]} contains invalid pressure")
        frames.append(frame)
        numbers.append(int(match[1]))
    if numbers != list(range(int(metadata["START_FRAME"]), int(metadata["END_FRAME"]) + 1)):
        raise ValueError("The numbered source frames do not match the declared range")

    grid = {
        "rows": rows,
        "columns": columns,
        "rowSpacingMm": float(metadata["ROW_SPACING"].split()[0]),
        "columnSpacingMm": float(metadata["COL_SPACING"].split()[0]),
        "layout": "row-major",
    }
    return grid, metadata, frames


def median_field(frames):
    return [round(value, 4) if (value := median(samples)) else 0 for samples in zip(*frames, strict=True)]


def orientation_checks(archive):
    checks = {}
    for direction in ("lateral", "medial", "inferior"):
        name = DIRECTORY + f"oks001_{direction}_M.csv"
        data = archive.read(name)
        grid, _, frames = parse_movie(data)
        pressure = median_field(frames)
        columns = grid["columns"]
        total = sum(pressure)
        checks[direction] = {
            "row": round(sum((index // columns) * value for index, value in enumerate(pressure)) / total, 2),
            "column": round(sum((index % columns) * value for index, value in enumerate(pressure)) / total, 2),
            "sourceFile": name,
            "sha256": sha256(data).hexdigest(),
        }
    return checks


def prepare(archive):
    sources = {}
    for name in archive.namelist():
        if not name.startswith(DIRECTORY):
            continue
        match = CONDITION.fullmatch(name.removeprefix(DIRECTORY))
        if match is None:
            continue
        angle, load, repeat = match.groups()
        sources.setdefault((int(angle), int(load)), []).append((int(repeat) if repeat else 1, name))

    grid = None
    conditions = []
    for angle in ANGLES:
        for load in LOADS:
            # The 30° sequence restarted after the first 20 N and 100 N measurements.
            # Use the later repeated trace, matching the cadaver tracking audit.
            repeats = sorted(sources[(angle, load)])
            trial, name = repeats[-1]
            data = archive.read(name)
            trial_grid, metadata, frames = parse_movie(data)
            if grid is not None and trial_grid != grid:
                raise ValueError(f"The sensor geometry changed in {name}")
            grid = trial_grid

            pressure = median_field(frames)
            conditions.append({
                "flexionDeg": angle,
                "quadricepsLoadN": load,
                "pressureMpa": pressure,
                "peakMpa": max(pressure),
                "sampleCount": len(frames),
                "samplePeriodSeconds": float(metadata["SECONDS_PER_FRAME"]),
                "recordedTimeLocal": metadata["TIME"],
                "sourceTrial": trial,
                "sourceFile": name,
                "sha256": sha256(data).hexdigest(),
                "earlierRepeatFiles": [earlier for _, earlier in repeats[:-1]],
            })

    checks = orientation_checks(archive)
    if not checks["medial"]["column"] > checks["lateral"]["column"] or not checks["inferior"]["row"] > max(
        checks["medial"]["row"], checks["lateral"]["row"]
    ):
        raise ValueError("The source direction checks no longer support the documented sensor orientation")
    grid["orientation"] = (
        "Native sensor coordinates. The source checks place medial toward higher columns "
        "and inferior toward higher rows. Those checks do not provide a 3D cartilage registration."
    )
    grid["orientationChecks"] = checks
    return {
        "specimen": "oks001",
        "source": {
            "title": "Open Knee(s), Cleveland Clinic",
            "url": "https://doi.org/10.18735/4e78-1311",
            "archive": SOURCE,
            "protocol": "https://doi.org/10.1016/j.dib.2021.106824",
            "license": "CC BY 4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        },
        "grid": grid,
        "flexionAnglesDeg": ANGLES,
        "quadricepsLoadsN": LOADS,
        "pressureRangeMpa": [0, max(case["peakMpa"] for case in conditions)],
        "summary": (
            "Median calibrated pressure at each Tekscan sensel over the recorded fixed-load movie. "
            "Flexion angles and quadriceps loads are nominal test settings. Each surface represents "
            "a separate controlled cadaver test and uses the original sensor coordinates."
        ),
        "conditions": conditions,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help=f"Public Open Knee(s) archive from {SOURCE}")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/patellar-pressure.json")
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        pressure = prepare(archive)
    args.output.write_text(json.dumps(pressure, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(pressure['conditions'])} measured pressure conditions to {args.output}")


if __name__ == "__main__":
    main()
