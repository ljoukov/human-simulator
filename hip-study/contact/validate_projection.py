"""Check material displacement, layer constraints and membrane shape across replays.

Run after python hip-study/tools/prepare_hip_contact.py. Requires the same atlas
dependencies. Includes all replay frames and nine interpolated poses per interval.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
from trimesh import Trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh.triangles import points_to_barycentric

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from prepare_hip_contact import C_SOURCE, DATA, MODEL, WASM, HipContact
from prepare_hip_deformation import POSES, deformation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thin_pairs(rest, faces):
    """Paired material points on opposite walls of thin regions of the atlas."""
    mesh = Trimesh(rest, faces, process=False)
    centers = mesh.triangles_center
    normals = mesh.face_normals
    source = np.flatnonzero(normals[:, 0] < -0.35)
    hit, row, xyz = RayMeshIntersector(mesh).intersects_id(
        centers[source] - 1e-5 * normals[source],
        -normals[source],
        multiple_hits=False,
        return_locations=True,
    )
    start = source[row]
    distances = np.linalg.norm(xyz - centers[start], axis=1)
    # Keep a ray that exits an opposing wall, away from the TFL-filled pocket.
    keep = (
        ((normals[start] * normals[hit]).sum(axis=1) < -0.5)
        & (distances > 0.00015)
        & (distances < 0.004)
    )
    start = start[keep]
    end = faces[hit[keep]]
    bary = points_to_barycentric(rest[end], xyz[keep])
    return faces[start], end, bary


def summarize(values, probabilities=(0.5, 0.9, 0.99, 1)):
    return np.round(np.quantile(values, probabilities), 6).tolist()


def raw_poses(clip):
    return np.array(
        [
            [
                f.get(k, 0.0)
                for k in ("hipFlexionDeg", "hipAdductionDeg", "hipRotationDeg")
            ]
            for f in clip["frames"]
        ]
    )


def get_pose(angles, phase):
    x = phase * (len(angles) - 1)
    i = min(int(x), len(angles) - 2)
    a = x - i
    return (1 - a) * angles[i] + a * angles[i + 1]


def point_distances(x, pairs):
    a, b, bary = pairs
    start = x[a].mean(axis=1)
    end = np.einsum("ij,ijk->ik", bary, x[b])
    return np.linalg.norm(end - start, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DATA / "hip-contact-validation.json"
    )
    args = parser.parse_args()
    config = json.loads((DATA / "hip-contact.json").read_text())
    assert (
        sha(C_SOURCE) == config["solverSourceSha256"]
        and sha(MODEL) == config["contactInputSha256"]
    )
    assert sha(WASM) == config["wasmSha256"]
    contact = HipContact()
    geo = contact.triangles
    edges = np.unique(
        np.sort(
            np.concatenate((geo[:, [0, 1]], geo[:, [0, 2]], geo[:, [1, 2]])), axis=1
        ),
        axis=0,
    )
    a, b = edges.T
    pairs = thin_pairs(contact.rest, geo)
    proximal = contact.weights == 0
    distal = contact.weights == 1
    medial = np.array(config["interfaces"]["tflMedialVertexIds"])

    def evaluate(pose):
        posed = contact.positions(pose) - contact.center
        free = deformation(contact.rest, contact.weights, pose)
        offsets = contact.offsets.copy() * 1000
        correction = posed - free
        raw_lengths = np.linalg.norm(free[a] - free[b], axis=1)
        corrected_lengths = np.linalg.norm(posed[a] - posed[b], axis=1)
        return (
            correction,
            offsets,
            contact.metrics.copy(),
            100 * np.abs(corrected_lengths / raw_lengths - 1),
            1000 * np.abs(point_distances(posed, pairs) - point_distances(free, pairs)),
        )

    def checks(pose):
        correction, offsets, metrics, edge, wall = evaluate(pose)
        return dict(
            poseDeg=list(map(float, pose)),
            displacementMmP50P90P99Max=summarize(offsets),
            proximalPinMaxMm=float(offsets[proximal].max()),
            distalSectionMaxMm=float(offsets[distal].max()),
            protectedMedialLayerMaxMm=float(offsets[medial].max()),
            correctableConstraintMaxDeficitMm=1000 * metrics[1],
            scopedCapShortfallMm=1000 * metrics[3],
            rawDeficitIncludingEmbeddedCornersMm=1000 * metrics[21],
            scopedCapShortfallByObstacleMm=np.round(metrics[4:12] * 1000, 6).tolist(),
            rawDeficitByObstacleMm=np.round(metrics[22:30] * 1000, 6).tolist(),
            addedEdgeChangeAbsPercentP95P99Max=summarize(edge, (0.95, 0.99, 1)),
            addedOpposingWallDistanceChangeMmP95P99Max=summarize(wall, (0.95, 0.99, 1)),
        )

    motion = json.loads((DATA / "motion.json").read_text())
    reports = []
    for clip in motion["clips"]:
        angles = raw_poses(clip)
        steps = (len(angles) - 1) * 10
        on = []
        between = []
        shifts = []
        edges_at_frames = []
        wall_at_frames = []
        delta_at_frames = []
        delta_grid = []
        times = []
        previous = None
        previous_frame = None
        for i in range(steps + 1):
            pose = get_pose(angles, i / steps)
            start = time.perf_counter()
            posed = contact.positions(pose) - contact.center
            times.append(1000 * (time.perf_counter() - start))
            metrics = contact.metrics.copy()
            (on if i % 10 == 0 else between).append(metrics)
            correction = posed - deformation(contact.rest, contact.weights, pose)
            if previous is not None:
                delta_grid.append(1000 * np.linalg.norm(correction - previous, axis=1))
            previous = correction
            if i % 10 == 0:
                _, offsets, _, edge, wall = evaluate(pose)
                shifts.append(offsets)
                edges_at_frames.append(edge)
                wall_at_frames.append(wall)
                if previous_frame is not None:
                    delta_at_frames.append(
                        1000 * np.linalg.norm(correction - previous_frame, axis=1)
                    )
                previous_frame = correction

        def deficits(matrix):
            d = np.array(matrix)
            return dict(
                n=len(d),
                maxFeasibleResidualMm=float(1000 * d[:, 1].max()),
                maxCapShortfallMm=float(1000 * d[:, 3].max()),
                posesExceedingScopedClearanceByOneMicron=int(np.sum(d[:, 3] > 1e-6)),
                maxRawDeficitIncludingEmbeddedCornersMm=float(1000 * d[:, 21].max()),
                maxCapShortfallByObstacleMm=np.round(
                    d[:, 4:12].max(axis=0) * 1000, 6
                ).tolist(),
                maxRawDeficitByObstacleMm=np.round(
                    d[:, 22:30].max(axis=0) * 1000, 6
                ).tolist(),
            )

        sh = np.array(shifts)
        reports.append(
            dict(
                clip=clip["id"],
                frames=deficits(on),
                betweenFrames=deficits(between),
                correctionsMmP50P90P99Max=summarize(sh),
                fractionUnmoved=float(np.mean(sh < 1e-7)),
                proximalPinMaxMm=float(sh[:, proximal].max()),
                distalSectionMaxMm=float(sh[:, distal].max()),
                protectedMedialLayerMaxMm=float(sh[:, medial].max()),
                addedEdgeChangeAbsPercentP95P99Max=summarize(
                    edges_at_frames, (0.95, 0.99, 1)
                ),
                addedOpposingWallDistanceChangeMmP95P99Max=summarize(
                    wall_at_frames, (0.95, 0.99, 1)
                ),
                adjacentFrameCorrectionChangeMmP95P99Max=summarize(
                    delta_at_frames, (0.95, 0.99, 1)
                ),
                tenthFrameCorrectionChangeMmP95P99Max=summarize(
                    delta_grid, (0.95, 0.99, 1)
                ),
                nativeCallMedianMs=float(np.median(times)),
            )
        )
    known = []
    for label, phi in [("predsim-hip-75", 0.6783752274), ("h1l-walk", 0.996792016)]:
        angles = raw_poses(next(c for c in motion["clips"] if c["id"] == label))
        left = contact.positions(get_pose(angles, phi - 0.5e-9))
        right = contact.positions(get_pose(angles, phi + 0.5e-9))
        known.append(
            dict(
                clip=label,
                phase=phi,
                phaseDelta=1e-9,
                maxMaterialStepMm=1000 * np.linalg.norm(left - right, axis=1).max(),
            )
        )
    result = dict(
        method="Quasi-static frontal ordering in the half-rotated lateral frame of the tract. Each clipped band–obstacle triangle polygon supplies affine unilateral gap constraints. A slope-limited displacement envelope couples nearby points over the surface and locally across thin walls. The band stays fixed at its proximal anchors, distal section and tracked deep TFL interface.",
        constraints="The nominal gap is 0.25 mm measured along the projection direction. Scoped constraints exclude the deep TFL lamina and registered iliac attachments, and tolerate positive contributions of immobile corners. The cap shortfall is the target-clearance deficit remaining after projection, before allowing fixed-point/slope-cap infeasibility. The raw deficit restores the fixed-corner contribution on participating faces. Triangle intersections, including anatomical interfaces deliberately left embedded, are evaluated independently.",
        shapeMeasures="Edge percentage is the absolute change in an atlas edge length compared with the same pose before contact. Opposing-wall measurements use neutral outer-face centroids and corresponding material points hit on the opposite band wall with neutral separation 0.15–4 mm and opposing normals. The reported change is the absolute difference in their Euclidean separation at the same pose.",
        timeMeasures="Adjacent-step statistics compare the correction vectors relative to the underlying kinematic skinning. Between-frame samples interpolate the displayed angles and call the solver; the time sequence is sampled and does not prove continuous-time nonintersection.",
        scope="Generic atlas illustration; quasi-static unilateral contact with idealized bounded in-sheet shear. No constitutive stress, tissue pressure or disease estimate.",
        sources={
            k: config[k]
            for k in [
                "atlasSha256",
                "attachmentsSha256",
                "contactInputSha256",
                "solverSourceSha256",
                "wasmSha256",
            ]
        },
        obstacles=[s["label"] for s in config["models"][1:]],
        wallPairs=len(pairs[0]),
        edges=len(edges),
        proximalPins=int(proximal.sum()),
        distalCutPins=int(distal.sum()),
        deepTflInterface=len(medial),
        neutral=checks([0, 0, 0]),
        reportedFrame=checks(raw_poses(motion["clips"][0])[51]),
        stressPoses=[checks(p) for p in POSES[1:]],
        clips=reports,
        localizedContinuityChecks=known,
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
