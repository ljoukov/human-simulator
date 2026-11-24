#!/usr/bin/env python3
"""Evaluate selected OpenSim GeometryPaths on the four recorded GC4 strides.

Usage: python prepare_muscle_paths.py /path/to/ReducePoly_forMSK
Requires the same pinned source revision, Python packages and movement files as
prepare_whole_body.py. No online service or OpenSim runtime is needed in the viewer.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import opensim

from prepare_data import COMMIT, SOURCE_PATH, source_table
from prepare_whole_body import MODEL_PATH, sample_coordinates, simtk_values


# Keep the native right/left asymmetry. The implanted right knee has distinct
# ligament bundles and uses the source quadriceps routes ending at the patella.
# Each pair shares an arclength sample count to resolve the shortest bends.
PAIRS = [
    ("Rectus femoris", "recfem_withreallength", "rect_fem_l", 32),
    ("Vastus lateralis", "vaslat_withreallength", "vas_lat_l", 24),
    ("Vastus medialis", "vasmed_withreallength", "vas_med_l", 24),
    ("Biceps femoris, long head", "bflh", "bifemlh_l", 28),
    ("Semitendinosus", "semiten", "semiten_l", 28),
    ("Medial gastrocnemius", "gasmed", "med_gas_l", 32),
    ("Lateral gastrocnemius", "gaslat", "lat_gas_l", 32),
    ("Soleus", "soleus", "soleus_l", 8),
    ("Tibialis anterior", "tibant", "tib_ant_l", 24),
    ("Fibularis longus", "perlong", "per_long_l", 34),
    ("Gluteus medius, middle path", "glmed2", "glut_med2_l", 8),
    ("External oblique", "extobl_r", "extobl_l", 8),
    ("Erector spinae", "ercspn_r", "ercspn_l", 8),
]
LIGAMENTS = [("Lateral", "patlig_lat"), ("Central", "patlig_center"), ("Medial", "patlig_med")]
DATA = Path(__file__).resolve().parents[1] / "data"
FRAME_COUNT = 51


def path_geometry(model, state, name):
    force = model.getForceSet().get(name)
    actuator = (opensim.Ligament if name.startswith("patlig") else opensim.Muscle).safeDownCast(force)
    geometry = actuator.getGeometryPath()
    current = geometry.getCurrentPath(state)
    positions = []
    for i in range(current.getSize()):
        point = current.get(i)
        wrap = opensim.PathWrapPoint.safeDownCast(point)
        if wrap:
            # The second tangency point carries the intermediate surface polyline,
            # expressed in its parent body. Some wrap objects omit the last vertex.
            curve = wrap.getWrapPath(state)
            frame = wrap.getParentFrame()
            for j in range(curve.getSize()):
                ground = frame.findStationLocationInGround(state, curve.get(j))
                positions.append(simtk_values(ground, 3))
        # PathWrapPoint can retain a cached ground position from the previous
        # wrap solve. Transform its freshly evaluated local position explicitly.
        ground = point.getParentFrame().findStationLocationInGround(state, point.getLocation(state))
        positions.append(simtk_values(ground, 3))
    return np.asarray(positions)


def sample_path(points, count):
    distance = np.r_[0.0, np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()]
    keep = np.r_[True, np.diff(distance) > 1e-9]
    distance, points = distance[keep], points[keep]
    if len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Degenerate muscle path")
    u = np.linspace(0, distance[-1], count)
    return np.column_stack([np.interp(u, distance, component) for component in points.T])


def export(repo, output):
    opensim.Logger.setLevelString("error")
    model = opensim.Model(str(repo / MODEL_PATH))
    state = model.initSystem()
    coords = [model.getCoordinateSet().get(i) for i in range(model.getCoordinateSet().getSize())]
    for coord in coords:
        if coord.getLocked(state):
            coord.setLocked(state, False)
    pelvis = model.getBodySet().get("pelvis")
    motion = json.loads((DATA / "whole-body-motion.json").read_text())
    gait = {trial["id"]: trial for trial in json.loads((DATA / "gait.json").read_text())["trials"]}
    paths = [
        {"name": f"{side} {label.lower()}", "source": muscle, "kind": "muscle", "count": count}
        for label, right, left, count in PAIRS
        for side, muscle in (("Right", right), ("Left", left))
    ] + [
        {"name": f"Right patellar ligament, {label.lower()} bundle", "source": name,
         "kind": "ligament", "count": 2} for label, name in LIGAMENTS
    ]
    archive = {
        "revision": COMMIT, "sourceModel": MODEL_PATH, "frameCount": FRAME_COUNT, "paths": paths,
        "encoding": "Little-endian int16, 0.1 mm. Coordinates relative to pelvis translation in laboratory axes. Each frame concatenates the paths in order. Each path is sampled at equal arclength in its OpenSim GeometryPath, including active moving and wrapping points.",
        "clips": {},
    }
    for clip in motion["clips"]:
        if clip["phase"] != "stride":
            continue
        recording = clip["recording"]
        ik = source_table(repo / SOURCE_PATH, "IK", recording, "Qsall")
        start = gait[clip["id"]]["stanceStartSeconds"]
        times = np.linspace(start, start + clip["durationSeconds"], FRAME_COUNT)
        frames = []
        for sample in sample_coordinates(ik, times, coords):
            for coord, value in zip(coords, sample, strict=True):
                coord.setValue(state, float(value), False)
            model.realizePosition(state)
            center = simtk_values(pelvis.getPositionInGround(state), 3)
            paths_at_time = [sample_path(path_geometry(model, state, path["source"]), path["count"])
                             for path in paths]
            frames.append(np.concatenate(paths_at_time) - center)
        ticks = np.rint(np.array(frames) * 10000)
        if np.max(np.abs(ticks)) >= 32768:
            raise ValueError("Path leaves representable 3.27 m range")
        archive["clips"][clip["id"]] = base64.b64encode(ticks.astype("<i2").tobytes()).decode("ascii")
    output.write_text(json.dumps(archive, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--output", type=Path, default=DATA / "muscle-paths.json")
    args = parser.parse_args()
    export(args.repository, args.output)
