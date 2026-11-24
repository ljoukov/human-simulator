#!/usr/bin/env python3
"""Compare the limb model with optical patella tracking from Open Knee(s) oks001.

Requires NumPy and npTDMS. Pass the local joint_mechanics-oks001.zip archive.
"""

import argparse
import hashlib
import io
import json
import re
import struct
from configparser import ConfigParser
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from nptdms import TdmsFile


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = "http://archive.simtk.org/oks/MRI_JointMech/"
PREFIX = "joint_mechanics-oks001/PatellofemoralJoint/KinematicsKinetics/"
PASSIVE_TRIAL = "006_Passive Flexion 0-60, optimized"
PATELLA_COORDINATES = ("Medial", "Posterior", "Superior", "Flexion", "Valgus", "Internal Rotation")
LOAD_TRIAL_START = {0: 7, 15: 14, 30: 24, 45: 31, 60: 38}
LOADS = (20, 100, 200, 300, 400, 500, 600)
OPENSIM_TO_JCS = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


def rotation_xyz(alpha, beta, gamma):
    a, b, c = np.cos([alpha, beta, gamma])
    u, v, w = np.sin([alpha, beta, gamma])
    return np.array([
        [b * c, -b * w, v],
        [u * v * c + a * w, -u * v * w + a * c, -u * b],
        [-a * v * c + u * w, a * v * w + u * c, a * b],
    ])


def relative_angle_degrees(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def femur_relative_pose(coordinates, offsets):
    # Open Knee(s) Knee Coordinate Systems v2.1, section 4.2. simVITRO
    # records values relative to the offsets, which must be restored first.
    medial, posterior, superior = coordinates[:3] + 1000 * offsets[:3]
    flexion, valgus, rotation = np.radians(coordinates[3:]) + offsets[3:]
    orientation = rotation_xyz(flexion, valgus, rotation)
    position = (
        np.array([medial, posterior * np.cos(flexion), posterior * np.sin(flexion)])
        + superior * orientation[:, 2]
    )
    return position, orientation


def read_trial(archive, trial):
    folder = PREFIX + trial
    entry = f"{folder}/Data/{trial}_main_processed.tdms"
    tdms_bytes = archive.read(entry)
    config_entry = f"{folder}/Configuration/Project.sVprj"
    config_bytes = archive.read(config_entry)
    with ZipFile(io.BytesIO(config_bytes)) as project:
        config_name, = [name for name in project.namelist() if name.endswith("State.cfg")]
        config = ConfigParser(interpolation=None, strict=False)
        config.read_string(project.read(config_name).decode("latin-1"))

    channel_config = config["Knee PTFJ"]
    offsets = np.fromstring(channel_config["Position Offset (m,rad)"].split(">")[1].strip(' "'), sep=" ")
    if offsets.shape != (6,) or not np.isfinite(offsets).all():
        raise ValueError(f"Expected six finite patellofemoral offsets in {config_entry}")
    frame = tuple(channel_config[key] for key in ("T_Sensor1_RB1", "T_Sensor2_RB2"))

    tdms = TdmsFile.read(io.BytesIO(tdms_bytes))
    flex_channel = tdms["State.Knee JCS"]["Knee JCS Flexion"]
    patella = [tdms["State.Knee PTFJ"]["Knee PTFJ " + key] for key in PATELLA_COORDINATES]
    units = [channel.properties["unit_string"] for channel in [flex_channel, *patella]]
    if units != ["deg", "mm", "mm", "mm", "deg", "deg", "deg"]:
        raise ValueError(f"Unexpected TDMS units in {entry}: {units}")
    flexion = flex_channel[:]
    coordinates = np.column_stack([channel[:] for channel in patella])
    if flexion.shape != (len(coordinates),) or not np.isfinite(np.column_stack((flexion, coordinates))).all():
        raise ValueError(f"Misaligned or invalid optical tracking channels in {entry}")
    provenance = {
        "archiveMember": entry,
        "sha256": hashlib.sha256(tdms_bytes).hexdigest(),
        "configurationMember": config_entry,
        "configurationSha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    return flexion, coordinates, offsets, frame, provenance


def read_glb_patella_rotation(path):
    with path.open("rb") as glb:
        magic, version, _, json_length, chunk_type = struct.unpack("<4sIIII", glb.read(20))
        if (magic, version, chunk_type) != (b"glTF", 2, 0x4E4F534A):
            raise ValueError(f"Invalid glTF 2.0 header in {path}")
        description = json.loads(glb.read(json_length))
        _, binary_type = struct.unpack("<II", glb.read(8))
        if binary_type != 0x004E4942:
            raise ValueError(f"Missing GLB binary chunk in {path}")
        binary = glb.tell()

        def accessor(index, width):
            acc = description["accessors"][index]
            view = description["bufferViews"][acc["bufferView"]]
            if acc["componentType"] != 5126 or view.get("byteStride", width * 4) != width * 4:
                raise ValueError("Expected an unpacked float32 animation accessor")
            glb.seek(binary + view.get("byteOffset", 0) + acc.get("byteOffset", 0))
            return np.frombuffer(glb.read(acc["count"] * width * 4), dtype="<f4").reshape(-1, width)

        animation, = [item for item in description["animations"] if item["name"] == "Knee flexion"]
        node, = [i for i, item in enumerate(description["nodes"]) if item["name"] == "PatellaFrame"]
        channel, = [
            item for item in animation["channels"]
            if item["target"] == {"node": node, "path": "rotation"}
        ]
        sampler = animation["samplers"][channel["sampler"]]
        if sampler.get("interpolation", "LINEAR") != "LINEAR":
            raise ValueError("Expected glTF spherical interpolation")
        degrees = accessor(sampler["input"], 1)[:, 0].astype(float) * 120
        quaternions = accessor(sampler["output"], 4).astype(float)
    return degrees, quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)


def interpolate_orientation(degrees, quaternions, angles):
    indices = np.searchsorted(degrees, angles)
    if (indices == 0).any() or (indices == len(degrees)).any():
        raise ValueError("Comparison angles must be within the baked animation clip")
    weight = (angles - degrees[indices - 1]) / (degrees[indices] - degrees[indices - 1])
    first, second = quaternions[indices - 1], quaternions[indices]
    dot = np.sum(first * second, axis=1)
    second = second * np.where(dot < 0, -1, 1)[:, None]
    theta = np.arccos(np.clip(np.abs(dot), 0, 1))
    # sinc keeps the equivalent linear interpolation at coincident rotations.
    lower = (1 - weight) * np.sinc((1 - weight) * theta / np.pi) / np.sinc(theta / np.pi)
    upper = weight * np.sinc(weight * theta / np.pi) / np.sinc(theta / np.pi)
    return lower[:, None] * first + upper[:, None] * second


def raw_reference(path):
    raw = json.loads(path.read_text())
    samples = [record for record in raw["samples"] if record["degrees"] <= 70]
    degrees = np.array([record["degrees"] for record in samples])
    tibia = np.array([record["tibia"] for record in samples])
    patella = np.array([record["patella"] for record in samples])[:, :, :3]
    jcs = OPENSIM_TO_JCS @ tibia[:, :, :3] @ OPENSIM_TO_JCS.T
    measured_axis = np.degrees(np.arctan2(-jcs[:, 1, 2], jcs[:, 2, 2]))
    if not np.allclose(patella[:, :, 2], [0, 0, 1], atol=1e-8):
        raise ValueError("Expected rotations exclusively about the Rajagopal model's lateral axis")
    patella_degrees = np.degrees(np.unwrap(np.arctan2(patella[:, 1, 0], patella[:, 0, 0])))
    return degrees, measured_axis, patella_degrees


def compare_passive(archive, reference, glb):
    flexion, coordinates, offsets, _, source = read_trial(archive, PASSIVE_TRIAL)
    high = int(np.argmax(flexion))
    increasing = np.arange(len(flexion)) <= high
    samples = []
    measured_rotations = []
    for center in range(5, 56, 5):
        window = increasing & (np.abs(flexion - center) <= 0.25)
        if window.sum() < 10:
            raise ValueError(f"Insufficient optical samples near {center} degrees")
        position, rotation = femur_relative_pose(np.median(coordinates[window], axis=0), offsets)
        measured_rotations.append(rotation)
        samples.append({
            "targetKneeFlexionDeg": center,
            "measuredKneeFlexionDeg": round(float(np.median(flexion[window])), 4),
            "samples": int(window.sum()),
            "measuredPatellaOriginInFemurMm": position.round(4).tolist(),
        })

    raw_pose, raw_flexion, raw_degrees = reference
    abscissa = np.array([row["measuredKneeFlexionDeg"] for row in samples])
    model_angles = np.interp(abscissa, raw_flexion, raw_pose)
    raw_at_measured = np.interp(model_angles, raw_pose, raw_degrees)
    raw_at_measured = np.abs(raw_at_measured - raw_at_measured[0])
    glb_rotations = interpolate_orientation(*glb, model_angles)
    glb_at_measured = np.degrees(2 * np.arccos(np.clip(np.abs(glb_rotations @ glb_rotations[0]), 0, 1)))
    measured = np.array([
        relative_angle_degrees(measured_rotations[0], rotation)
        for rotation in measured_rotations
    ])
    for i, row in enumerate(samples):
        row.update(
            modelKneeInputDeg=round(float(model_angles[i]), 4),
            measuredPatellaRotationChangeDeg=round(float(measured[i]), 4),
            rajagopalPatellaRotationChangeDeg=round(float(raw_at_measured[i]), 4),
            glbPatellaRotationChangeDeg=round(float(glb_at_measured[i]), 4),
        )

    return {
        **source,
        "acquiredSamples": len(flexion),
        "measuredPeakKneeFlexionDeg": round(float(flexion[high]), 4),
        "measuredKneeFlexionRangeDeg": [5, 55],
        "method": "Increasing flexion up to the optical maximum. Take medians of samples within 0.25° of each target, then measure the 3D angle from the 5° reference rotation. Match each model by its tibiofemoral flexion component around the femoral mediolateral axis.",
        "rawRotationMagnitudeRmseDeg": round(float(np.sqrt(np.mean((raw_at_measured[1:] - measured[1:]) ** 2))), 4),
        "glbRotationMagnitudeRmseDeg": round(float(np.sqrt(np.mean((glb_at_measured[1:] - measured[1:]) ** 2))), 4),
        "samples": samples,
    }


def compare_loads(archive):
    rows = []
    changes = []
    for target, first in LOAD_TRIAL_START.items():
        for trial_number, load in enumerate(LOADS, start=first):
            flexion, coordinates, offsets, frame, source = read_trial(
                archive, f"{trial_number:03}_{target}flex_{load}N",
            )
            position, rotation = femur_relative_pose(np.median(coordinates, axis=0), offsets)
            if load == LOADS[0]:
                low_position, low_rotation, low_frame = position, rotation, frame
            if frame != low_frame:
                raise ValueError("The optical bone frames changed across loads at the same flexion angle")
            displacement = position - low_position
            row = {
                **source,
                "targetKneeFlexionDeg": target,
                "measuredKneeFlexionDeg": round(float(np.median(flexion)), 4),
                "nominalQuadricepsLoadN": load,
                "samples": len(flexion),
                "patellaOriginInFemurMm": position.round(4).tolist(),
                "patellaTranslationFrom20NMm": displacement.round(4).tolist(),
                "patellaTranslationMagnitudeFrom20NMm": round(float(np.linalg.norm(displacement)), 4),
                "patellaRotationFrom20NDeg": round(relative_angle_degrees(low_rotation, rotation), 4),
            }
            rows.append(row)
            if load == LOADS[-1]:
                changes.append({
                    "targetKneeFlexionDeg": target,
                    "translationMm": row["patellaTranslationMagnitudeFrom20NMm"],
                    "rotationDeg": row["patellaRotationFrom20NDeg"],
                })
    return {
        "method": "Medians of all synchronized optical samples from each separately recorded held-load trial. Compare the 20 N and other trials at each fixed commanded knee angle. Loads are nominal actuator targets carried by the archive trial names.",
        "translationAxisOrder": ["medial", "posterior", "superior"],
        "changeFrom20To600N": changes,
        "trials": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Open Knee(s) joint_mechanics-oks001.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "knee-loading/data/patella-oks001.json")
    args = parser.parse_args()
    reference_path = ROOT / "lower-limb/data/knee-motion.json"
    glb_path = ROOT / "lower-limb/assets/lower-limb.glb"
    with ZipFile(args.archive) as archive:
        report = {
            "specimen": "oks001",
            "side": "right",
            "source": {
                "dataset": "Open Knee(s), Cleveland Clinic",
                "datasetDoi": "https://doi.org/10.18735/4e78-1311",
                "protocolDoi": "https://doi.org/10.1016/j.dib.2021.106824",
                "archive": ARCHIVE + "joint_mechanics/joint_mechanics-oks001.zip",
                "license": "CC BY 4.0",
                "coordinateConvention": ARCHIVE + "doc/Knee%20Coordinate%20Systems.pdf",
                "channelConvention": ARCHIVE + "doc/2013CB-031-002.B%20simVITRO%20Data%20File%20Contents_Open%20Knee.pdf",
            },
            "modelInputs": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (reference_path, glb_path)
            },
            "scope": [
                "One cadaver specimen. The rotation-magnitude comparison is geometric and does not establish equivalent loading or subject-specific prediction.",
                "The model and specimen patella origins denote different landmarks. Absolute position errors and axis-specific rotational errors require registration to common anatomical landmarks.",
                "The exported model retains Rajagopal's prescribed patella rotation and changes translation for atlas fit. Neither model predicts joint response to applied quadriceps loads.",
            ],
            "passive": compare_passive(archive, raw_reference(reference_path), read_glb_patella_rotation(glb_path)),
            "controlledLoads": compare_loads(archive),
        }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
