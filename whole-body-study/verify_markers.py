"""Check the exported mass centers against the transforms used by the animation."""
import argparse
import json
from pathlib import Path
import numpy as np
import opensim as osim
from scipy.spatial.transform import Rotation
from solve import MODEL


def check(source, study, motion):
    model = osim.Model(str(source / MODEL))
    state = model.initSystem()
    bodies = model.getBodySet()
    segments = motion["segmentOrder"]
    clips = {clip["id"]: clip for clip in motion["clips"]}
    # Both small implant components are rigidly welded to the recorded bones.
    attached_to = {"femoral_component": "femur", "tibial_tray": "tibia"}
    weighted_stations = []
    for index in range(bodies.getSize()):
        body = bodies.get(index)
        parent = attached_to.get(body.getName(), body.getName())
        station = body.findStationLocationInAnotherFrame(state, body.getMassCenter(), bodies.get(parent))
        weighted_stations.append((segments.index(parent), np.array([station.get(i) for i in range(3)]), body.getMass()))

    for trial in study["forcePlateComparison"]:
        markers = trial["forcePlateCheck"]["positionMarkers"]
        frames = np.array(clips[trial["clipId"]]["frames"]).reshape(len(markers["phasePct"]), len(segments), 7)
        centers = np.zeros((len(frames), 3))
        for segment, station, mass in weighted_stations:
            transforms = frames[:, segment]
            world_station = transforms[:, :3] + Rotation.from_quat(transforms[:, 3:]).apply(station)
            centers += mass * world_station
        centers /= model.getTotalMass(state)
        origin = np.array(markers["clipOriginGroundM"])
        errors = []
        for key, truth in [("centerOfMassGroundM", centers), ("pelvisOriginGroundM", frames[:, segments.index("pelvis"), :3])]:
            observed = np.array(markers[key]) - origin
            errors.append(float(np.max(np.linalg.norm(observed - truth, axis=1))))
        # The released clip rounds translations to 1e-5 m and quaternions to 1e-6.
        if max(errors) > 1e-5:
            raise ValueError(f"{trial['recording']}: marker registration exceeds clip precision: {errors}")
        print(f"{trial['recording']}: max whole-body error {errors[0]:.8f} m, pelvis error {errors[1]:.8f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--study", type=Path, default=directory / "stride.json")
    parser.add_argument("--motion", type=Path, default=directory.parent / "knee-loading/data/whole-body-motion.json")
    args = parser.parse_args()
    osim.Logger.removeFileSink()
    osim.Logger.setLevelString("error")
    check(args.source.resolve(), json.loads(args.study.read_text()), json.loads(args.motion.read_text()))
