"""Independently intersect triangle meshes on the displayed outer IT-band surface.

Requires NumPy and SciPy. Units are mm in the narrow-phase geometry tests.
Pass --corrected to apply the display's contact solver, or use --band-poses
for an externally generated NPZ of corrected welded band positions
(metres relative to the atlas hip centre in unrotated pelvic coordinates).
Keys are clip IDs with shape (frames, vertices, 3), plus 'neutral' of shape
(1, vertices, 3). A supplied array must include every selected pose.

Run the same command with and without --corrected to compare the arc-only
display with the unilateral contact correction. --all-clips includes the
simulated cohort. Results count open-interior crossings between triangles
and sum their exposed line lengths. Tangencies and coplanar attachments are
excluded. Zero crossings at the sampled poses cannot establish positive
clearance, physical pressure or clearance between successive poses.

Reference-pair metadata identifies pairs of facets already intersecting in the
undeformed source, including source intersections occluded in lateral view.
The pelvic/distal pin categories require all three ITB face vertices to have
the corresponding exact attachment weight. The material interface metric
tracks the area-weighted fraction of originally enclosed tensor fasciae latae
surface samples still enclosed by the band. This metric describes the source
dissection, with sampling sensitivity near attachment boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from prepare_hip_deformation import ATLAS, DATA, UNIT, load_meshes


BAND = "Iliotibial tract"
DEFAULT_OBSTACLES = [
    "Femur", "Hip bone", "Gluteus maximus muscle", "Gluteus medius muscle", "Gluteus minimus muscle",
    "Tensor fasciae latae", "Vastus lateralis muscle", "Trochanteric bursa of gluteus maximus",
]


def pose_rotation(angles):
    flex, add, twist = angles
    return Rotation.from_euler("ZXY", [add, -flex, twist], degrees=True)


def attachment(label, keys, manifest, binary):
    rec = manifest["surfaces"][label]
    if "constant" in rec:
        return np.full(len(keys), rec["constant"])
    stride = rec.get("recordBytes", manifest["recordBytes"])
    dtype = np.dtype({"names": ["position", "weight"], "formats": [("<i4", 3), "<u2"],
                      "offsets": [0, 12], "itemsize": stride})
    rows = np.frombuffer(binary, dtype, count=len(keys), offset=rec["offsetBytes"])
    if not np.array_equal(rows["position"], keys):
        raise ValueError("Attachment field differs from atlas vertices: " + label)
    return rows["weight"] / 65535


def femur_frame(reference, weights, rotation):
    # The two rotations share an axis. Work in the femur frame to keep the
    # definition of lateral fixed through the stride. Left-side reflection and
    # global pelvic pose can then be omitted on both surfaces.
    return Rotation.from_rotvec((weights - 1)[:, None] * rotation.as_rotvec()).apply(reference)


def intersect_edges(edges, target):
    """Möller–Trumbore segment-triangle test, including edges of the target."""
    starts = edges
    directions = np.roll(starts, -1, axis=1) - starts
    e1, e2 = (target[:, 1] - target[:, 0])[:, None, :], (target[:, 2] - target[:, 0])[:, None, :]
    h = np.cross(directions, e2)
    determinant = np.sum(e1 * h, axis=2)
    obj = starts - target[:, :1, :]
    q = np.cross(obj, e1)
    regular = np.abs(determinant) > 1e-10
    inv = np.divide(1, determinant, out=np.zeros_like(determinant), where=regular)
    u = inv * np.sum(obj * h, axis=2)
    v = inv * np.sum(directions * q, axis=2)
    t = inv * np.sum(e2 * q, axis=2)
    tol = 1e-9
    valid = regular & (u >= -tol) & (v >= -tol) & (u + v <= 1+tol) & (t >= -tol) & (t <= 1+tol)
    return starts + np.clip(t, 0, 1)[:, :, None] * directions, valid


def transverse_segments(a, b):
    """Require both triangle interiors to pierce the other's supporting plane."""
    def crosses_plane(tri, plane):
        normal = np.cross(plane[:, 1] - plane[:, 0], plane[:, 2] - plane[:, 0])
        d = np.sum((tri - plane[:, :1]) * normal[:, None], axis=2)
        tolerance = 1e-6 * np.linalg.norm(normal, axis=1)
        return (d.min(axis=1) < -tolerance) & (d.max(axis=1) > tolerance)

    candidates = np.flatnonzero(crosses_plane(a, b) & crosses_plane(b, a))
    a, b = a[candidates], b[candidates]
    points_a, hits_a = intersect_edges(a, b)
    points_b, hits_b = intersect_edges(b, a)
    points = np.concatenate([points_a, points_b], axis=1)
    hits = np.concatenate([hits_a, hits_b], axis=1)
    has_two = np.flatnonzero(hits.sum(axis=1) >= 2)
    points, hits = points[has_two], hits[has_two]
    # Choose the farthest actual pair when a shared vertex is hit twice.
    i, j = np.triu_indices(6, 1)
    distances = ((points[:, i] - points[:, j]) ** 2).sum(axis=2)
    distances[~(hits[:, i] & hits[:, j])] = -1
    best = np.argmax(distances, axis=1)
    rows = np.arange(len(best))
    valid = distances[rows, best] > 1e-12
    return candidates[has_two[valid]], points[rows, i[best]][valid], points[rows, j[best]][valid]


def visible_length(p, q, band):
    """Clip each transverse line by every occluding band triangle in -X view.

    Along a line, projected barycentric coordinates and relative depth are
    affine. Intersect their four one-dimensional half-spaces, then take the
    union of the occluded intervals. This retains even short exposed segments.
    """
    a, b, c = band[:, 0], band[:, 1], band[:, 2]
    ab, ac = b-a, c-a
    det = ab[:, 1]*ac[:, 2] - ab[:, 2]*ac[:, 1]
    usable = np.abs(det) > 1e-12
    inverse = np.divide(1, det, out=np.zeros_like(det), where=usable)

    def constraints(points):
        delta = points[:, None] - a
        u = (delta[:, :, 1]*ac[:, 2] - delta[:, :, 2]*ac[:, 1]) * inverse
        v = (ab[:, 1]*delta[:, :, 2] - ab[:, 2]*delta[:, :, 1]) * inverse
        # The 2 µm tolerance absorbs the neutral atlas coordinate quantization.
        behind = delta[:, :, 0] - u*ab[:, 0] - v*ac[:, 0] - .002
        return u, v, 1-u-v, behind

    lo = np.zeros((len(p), len(band)))
    hi = np.ones_like(lo)
    hi[:, ~usable] = -1
    for start, end in zip(constraints(p), constraints(q)):
        change = end - start
        moving = np.abs(change) > 1e-13
        zero = np.divide(-start, change, out=np.zeros_like(start), where=moving)
        lo = np.where((change > 0) & moving, np.maximum(lo, zero), lo)
        hi = np.where((change < 0) & moving, np.minimum(hi, zero), hi)
        hi[(~moving) & (start < 0)] = -1
    invalid = hi <= lo
    lo[invalid] = hi[invalid] = 1
    order = np.argsort(lo, axis=1)
    lo = np.take_along_axis(lo, order, axis=1)
    hi = np.take_along_axis(hi, order, axis=1)
    covered_before = np.maximum.accumulate(hi, axis=1)
    previous = np.column_stack([np.zeros(len(p)), covered_before[:, :-1]])
    covered = np.maximum(0, hi - np.maximum(previous, lo)).sum(axis=1)
    return np.clip(1-covered, 0, 1)*np.linalg.norm(q-p, axis=1)


def candidate_pairs(band, obstacle):
    # Bounding spheres followed by an exact AABB filter. A sphere enclosing
    # each full triangle also encloses intersections through either face.
    ca, cb = band.mean(axis=1), obstacle.mean(axis=1)
    ra = np.linalg.norm(band - ca[:, None], axis=2).max(axis=1)
    rb = np.linalg.norm(obstacle - cb[:, None], axis=2).max(axis=1)
    nearby = cKDTree(ca).query_ball_point(cb, rb + ra.max() + .000001)
    counts = np.fromiter(map(len, nearby), dtype=np.int64, count=len(nearby))
    ib = np.repeat(np.arange(len(cb)), counts)
    ia = np.concatenate(nearby).astype(np.int64)
    keep = np.all(band.min(axis=1)[ia] <= obstacle.max(axis=1)[ib] + 1e-7, axis=1)
    keep &= np.all(obstacle.min(axis=1)[ib] <= band.max(axis=1)[ia] + 1e-7, axis=1)
    return ia[keep], ib[keep]


def measure(band, obstacle, band_zones=None, reference_pairs=None, contact_mask=None, source_height=None):
    ia, ib = candidate_pairs(band, obstacle)
    selected, p, q = transverse_segments(band[ia], obstacle[ib])
    if not len(p):
        return dict(pairs=0, lengthMm=0.0)
    lengths = np.concatenate([visible_length(p[i:i+100], q[i:i+100], band)
                              for i in range(0, len(p), 100)])
    positive = lengths > 1e-6
    lengths, ia, ib = lengths[positive], ia[selected][positive], ib[selected][positive]
    result = dict(pairs=len(lengths), lengthMm=round(float(lengths.sum()), 4))
    if len(lengths) and band_zones is not None:
        zones = band_zones[ia]
        result['byBandRegion'] = {name: dict(pairs=int((zones == name).sum()), lengthMm=round(float(lengths[zones == name].sum()), 4))
                                  for name in np.unique(zones)}
    if len(lengths) and reference_pairs is not None:
        novel = ~np.isin(ia * len(obstacle) + ib, reference_pairs)
        result['newFacePairs'] = int(novel.sum())
        result['newFacePairLengthMm'] = round(float(lengths[novel].sum()), 4)
        if novel.any() and source_height is not None:
            # Map complete line endpoints into material coordinates. Partial
            # occlusion can only narrow the reported source-height interval.
            faces = ia[novel]
            tri = band[faces]
            a, u, v = tri[:, 0], tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
            uu, uv, vv = (u*u).sum(axis=1), (u*v).sum(axis=1), (v*v).sum(axis=1)
            heights = []
            for end in [p[positive][novel], q[positive][novel]]:
                d = end - a
                du, dv = (d*u).sum(axis=1), (d*v).sum(axis=1)
                denom = uu*vv - uv*uv
                beta, gamma = (du*vv-dv*uv)/denom, (dv*uu-du*uv)/denom
                bary = np.column_stack([1-beta-gamma, beta, gamma])
                heights.extend((bary*source_height[faces]).sum(axis=1))
            result['newLineSourceHeightAboveDistalCutMm'] = [round(float(min(heights)), 4), round(float(max(heights)), 4)]
        if contact_mask is not None:
            selected = contact_mask[ia]
            result['onSelectedFaces'] = dict(pairs=int(selected.sum()), lengthMm=round(float(lengths[selected].sum()),4),
                                             newFacePairs=int((novel & selected).sum()),
                                             newFacePairLengthMm=round(float(lengths[novel & selected].sum()),4))
    return result


def enclosed(points, triangles, direction):
    """Parity of positive oblique ray hits against the closed tract shell."""
    axis = np.array(direction) / np.linalg.norm(direction)
    transverse = np.cross(axis, [0, 0, 1])
    transverse /= np.linalg.norm(transverse)
    basis = np.column_stack([axis, transverse, np.cross(axis, transverse)])
    points, triangles = points @ basis, triangles @ basis
    yz = triangles[:, :, 1:]
    center = yz.mean(axis=1)
    radius = np.linalg.norm(yz-center[:, None], axis=2).max(axis=1)
    neighbors = cKDTree(points[:, 1:]).query_ball_point(center, radius+1e-7)
    counts = np.fromiter(map(len, neighbors), dtype=int, count=len(neighbors))
    i = np.concatenate(neighbors).astype(int)
    j = np.repeat(np.arange(len(triangles)), counts)
    origin = triangles[j, 0]
    ab, ac = triangles[j, 1]-origin, triangles[j, 2]-origin
    delta = points[i]-origin
    det = ab[:, 1]*ac[:, 2] - ab[:, 2]*ac[:, 1]
    inv = np.divide(1, det, out=np.zeros_like(det), where=np.abs(det)>1e-12)
    u = (delta[:, 1]*ac[:, 2] - delta[:, 2]*ac[:, 1])*inv
    v = (ab[:, 1]*delta[:, 2] - ab[:, 2]*delta[:, 1])*inv
    hit = (np.abs(det)>1e-12) & (u>=0) & (v>=0) & (u+v<=1) & (u*ab[:, 0]+v*ac[:, 0]>delta[:, 0])
    return np.bincount(i[hit], minlength=len(points)) % 2 == 1


def enclosed_tfl(tfl, band):
    # Three symmetric barycentric samples, each representing 1/3 of the
    # original triangle's area. Track identical material points in every pose.
    samples = ((np.eye(3)*.5 + 1/6) @ tfl).reshape(-1, 3)
    first = enclosed(samples, band, [1, .2131, -.3571])
    second = enclosed(samples, band, [-.8821, .232, 1.153])
    if not np.array_equal(first, second):
        raise ValueError('Disagreement classifying the material interface')
    return first


def verify_geometry():
    outer = np.array([[0., 0, 0], [0, 2, 0], [0, 0, 2]])
    inner = outer + [.2, 0, 0]
    obstacle = np.array([[-1, .25, .25], [1, .25, .25], [0, .25, 1.25]])
    crossing = measure(np.stack([outer, inner]), obstacle[None])
    if crossing != dict(pairs=1, lengthMm=1.0) or measure(outer[None], outer[None])['pairs']:
        raise RuntimeError("Intersection or exposure check failed on analytical triangles")
    # Only the first 0.5 mm of this 1 mm line is covered in lateral projection.
    cover = np.array([[-1, 0, -.5], [-1, 2, -.5], [-1, 0, .75]])
    if not np.isclose(visible_length(np.array([[0., 0, .25]]), np.array([[0., 0, 1.25]]), cover[None])[0], .5):
        raise RuntimeError("Partial occlusion check failed")


def clip_summary(records):
    totals = []
    for record in records:
        hits = list(record['hits'].values())
        selected = [h.get('onSelectedFaces', h) for h in hits]
        tensor = record['hits'].get('Tensor fasciae latae', {})
        cover = tensor.get('onSelectedFaces', {})
        all_pairs = sum(h['pairs'] for h in hits)
        new_pairs = sum(h['newFacePairs'] for h in hits)
        fixed = sum(h['byBandRegion'].get('pelvisPinned', {}).get('pairs', 0)
                    + h['byBandRegion'].get('distalPinned', {}).get('pairs', 0) for h in hits)
        totals.append(dict(frame=record['frame'], pairs=all_pairs, new=new_pairs, pinned=fixed,
                           selected=sum(h['pairs'] for h in selected), newSelected=sum(h['newFacePairs'] for h in selected),
                           length=sum(h['lengthMm'] for h in hits),
                           selectedLength=sum(h['lengthMm'] for h in selected),
                           newSelectedLength=sum(h['newFacePairLengthMm'] for h in selected),
                           tflPairs=tensor.get('pairs', 0), tflCoverPairs=cover.get('pairs', 0),
                           tflCoverLength=cover.get('lengthMm', 0),
                           tflRestLength=tensor.get('lengthMm', 0)-cover.get('lengthMm', 0)))
    # A pair-frame is one exposed pair of facets in one sampled pose.
    by_structure = {}
    for record in records:
        for label, hit in record['hits'].items():
            if not hit['newFacePairs']:
                continue
            item = by_structure.setdefault(label, dict(frames=0, maxNewExposedLineMm=0))
            item['frames'] += 1
            if hit['newFacePairLengthMm'] >= item['maxNewExposedLineMm']:
                item.update(maxNewExposedLineMm=hit['newFacePairLengthMm'], peakFrame=record['frame'],
                            peakSourceHeightAboveDistalCutMm=hit['newLineSourceHeightAboveDistalCutMm'])
    return dict(frames=len(records), framesWithCrossing=sum(t['pairs']>0 for t in totals),
                newCrossingByStructure=by_structure,
                framesWithNewCrossing=sum(t['new']>0 for t in totals),
                framesWithNewSelectedCrossing=sum(t['newSelected']>0 for t in totals),
                crossedPairFrames=sum(t['pairs'] for t in totals),
                sourceExistingPairFrames=sum(t['pairs']-t['new'] for t in totals),
                fixedAttachmentPairFrames=sum(t['pinned'] for t in totals),
                newSelectedPairFrames=sum(t['newSelected'] for t in totals),
                newUnselectedPairFrames=sum(t['new']-t['newSelected'] for t in totals),
                maxExposedLineMm=round(max(t['length'] for t in totals), 4),
                maxSelectedLineMm=round(max(t['selectedLength'] for t in totals), 4),
                maxNewSelectedLineMm=round(max(t['newSelectedLength'] for t in totals), 4),
                tflFramesOnSelectedOuterCover=sum(t['tflCoverPairs']>0 for t in totals),
                tflFramesOutsideSelectedCover=sum(t['tflPairs']>t['tflCoverPairs'] for t in totals),
                maxTflLineOnSelectedCoverMm=max(t['tflCoverLength'] for t in totals),
                maxTflLineOutsideSelectedCoverMm=round(max(t['tflRestLength'] for t in totals), 4),
                retainedNeutralTflAreaPctRange=[min(r['enclosedTflRetainedPct'] for r in records),
                                                max(r['enclosedTflRetainedPct'] for r in records)],
                maxPelvicPinChangeMm=max(r['maxPelvicPinChangeMm'] for r in records),
                maxDistalPinChangeMm=max(r['maxDistalPinChangeMm'] for r in records))


def comparison(source_path, corrected_path, original_path=None):
    source, contact = (json.loads(p.read_text()) for p in (source_path, corrected_path))
    common = ('auditCodeSha256', 'atlasSha256', 'motionSha256', 'interfaceMaskSha256', 'obstacles',
              'visibilityToleranceMm', 'distalCutSourceYM', 'tflInterface')
    if source['correctedBand'] or not contact['correctedBand'] or source['weightsSha256'] != contact['weightsSha256']:
        raise ValueError('Compare revised arc and contact with identical attachment fields')
    conditions = {'reboundArc': source, 'contact': contact}
    if original_path:
        original = json.loads(original_path.read_text())
        if original['correctedBand']:
            raise ValueError('The original binding comparison must use arc-only pose')
        conditions = {'originalArc': original, **conditions}
    clips = {}
    for name, data in conditions.items():
        if any(data[k]!=source[k] for k in common) or list(data['clips']) != list(source['clips']):
            raise ValueError('Inputs or masks differ between compared conditions')
        for clip, records in data['clips'].items():
            if [r['frame'] for r in records] != [r['frame'] for r in source['clips'][clip]]:
                raise ValueError('Frame indices differ for '+clip)
            clips.setdefault(clip, {})[name] = clip_summary(records)
    paths = {'reboundArc': source_path, 'contact': corrected_path}
    if original_path:
        paths['originalArc'] = original_path
    artifacts = dict(recordsSha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                     weightsSha256={name: data['weightsSha256'] for name, data in conditions.items()}, **contact['model'])
    commands = []
    if original_path:
        commands.extend(['python hip-study/tools/prepare_hip_deformation.py --uncoupled-tfl --output /tmp/hip-original-weights',
                         'python hip-study/tools/audit_hip_contact.py --attachment-dir /tmp/hip-original-weights --all-clips --output /tmp/hip-original.json'])
    flag = ' --original-records /tmp/hip-original.json' if original_path else ''
    commands.extend(['python hip-study/tools/audit_hip_contact.py --all-clips --output /tmp/hip-arc.json',
                     'python hip-study/tools/audit_hip_contact.py --corrected --all-clips --output /tmp/hip-contact.json',
                     'python hip-study/tools/audit_hip_contact.py --compare /tmp/hip-arc.json /tmp/hip-contact.json' + flag + ' --output hip-study/data/contact-audit.json'])
    out = dict(schemaVersion=1, **{k: source[k] for k in common}, **artifacts,
               method='Exact bidirectional segment/triangle intersections between open triangle interiors, then analytic line-interval occlusion by the complete tract along the femoral lateral axis. Tangency and coplanar contact are excluded. Welded atlas coordinates have 1 µm resolution.',
               interpretation='Pair-frames count repeated observations across poses. Line length sums exposed intersections at a pose. Penetration depth and sliding distance require separate measurements. Source-height intervals bound the contributing line endpoints relative to the distal material section. These display geometry checks apply to the sampled poses.',
               sourcePairs='Source-existing facet pairs intersect in the undeformed atlas, including its occluded interfaces. A pair first intersecting during motion counts as new.',
               masks='TFL selected faces are the solver’s designated outer cover. Hip-bone selected faces exclude the designated iliac attachment patch. All faces are selected for each remaining obstacle. Counts outside those masks remain in the all-face results.',
               attachments='Fixed-attachment counts require all three band face vertices to have exactly zero femoral interpolation weight (pelvic pins) or exactly unit weight (distal pins). Pin changes are distances from the corresponding arc-only positions. Area retention tracks originally enclosed TFL material with three quadrature points per triangle and two-direction ray parity.',
               clips=clips, reproduce=commands)
    for key, clip, frame in [('neutral', 'neutral', 0), ('oa51','bertaux-hoa001',51)]:
        if clip in source['clips'] and any(r['frame']==frame for r in source['clips'][clip]):
            out[key] = {'clip':clip, **{name:next(r for r in data['clips'][clip] if r['frame']==frame)
                                        for name, data in conditions.items()}}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    correction = parser.add_mutually_exclusive_group()
    correction.add_argument('--band-poses', type=Path)
    correction.add_argument('--corrected', action='store_true', help='Audit the display contact solver.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--attachment-dir', type=Path, default=DATA, help='Directory containing the two attachment field files.')
    parser.add_argument('--original-records', type=Path, help='Additional arc-only records from the original uncoupled TFL binding, for --compare.')
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('ARC', 'CONTACT'),
                        help='Summarize two matched audit JSON files without rerunning the geometry tests.')
    parser.add_argument('--clip', action='append', help='Limit to a named motion clip. Default uses recorded clips.')
    parser.add_argument('--all-clips', action='store_true')
    parser.add_argument('--frame', type=int, help='Only test this zero-based frame in each clip.')
    parser.add_argument('--obstacle', action='append', help='Override the default muscle, femur and bursa labels.')
    args = parser.parse_args()
    if args.compare:
        if not args.output:
            parser.error('--compare requires --output')
        args.output.write_text(json.dumps(comparison(*args.compare, args.original_records), indent=2)+'\n')
        return
    verify_geometry()
    meshes, center, atlas_hash = load_meshes(ATLAS)
    manifest = json.loads((args.attachment_dir/'hip-deformation.json').read_text())
    binary = (args.attachment_dir/'hip-attachments.bin').read_bytes()
    if manifest['sourceAtlasSha256'] != atlas_hash or hashlib.sha256(binary).hexdigest() != manifest['weightsSha256']:
        raise ValueError('Atlas and deformation weights are inconsistent')
    motion = json.loads((DATA/'motion.json').read_text())
    obstacles = args.obstacle or DEFAULT_OBSTACLES
    tfl_name = 'Tensor fasciae latae'
    labels = list(dict.fromkeys([BAND, tfl_name, *obstacles]))
    surfaces = {}
    for label in labels:
        k, triangles, _ = meshes[label]
        surfaces[label] = ((k / UNIT - center) * 1000, triangles, attachment(label, k, manifest, binary))
    corrected = np.load(args.band_poses) if args.band_poses else None
    solver = None
    model = json.loads((DATA/'hip-contact.json').read_text()) if (DATA/'hip-contact.json').exists() else {}
    if args.corrected:
        from prepare_hip_contact import C_SOURCE, MODEL, WASM, HipContact
        files = {'solverSourceSha256': C_SOURCE, 'contactInputSha256': MODEL, 'wasmSha256': WASM}
        if model['atlasSha256'] != atlas_hash or model['attachmentsSha256'] != manifest['weightsSha256']:
            raise ValueError('The contact model uses a different atlas or attachment field')
        for key, path in files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != model[key]:
                raise ValueError('Rebuild the contact model before auditing ' + path.name)
        solver = HipContact()
    jobs = [('neutral', [(0, [0,0,0])])]
    seen = set()
    for clip in motion['clips']:
        if args.clip and clip['id'] not in args.clip:
            continue
        if not args.clip and not args.all_clips and clip['kind'] == 'simulated':
            continue
        frames = list(enumerate(clip['frames']))
        if args.frame is not None:
            if not 0 <= args.frame < len(frames):
                parser.error('--frame is outside ' + clip['id'])
            frames = [frames[args.frame]]
        seen.add(clip['id'])
        jobs.append((clip['id'], [(i, [f.get(k,0) for k in ('hipFlexionDeg','hipAdductionDeg','hipRotationDeg')]) for i, f in frames]))
    if args.clip and set(args.clip) != seen:
        raise ValueError('Unknown clip ID')
    report = dict(auditCodeSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), atlasSha256=atlas_hash, weightsSha256=manifest['weightsSha256'], obstacles=obstacles,
                  motionSha256=hashlib.sha256((DATA/'motion.json').read_bytes()).hexdigest(),
                  visibilityToleranceMm=.002, correctedBand=args.corrected or corrected is not None, clips={})
    if solver is not None:
        report['model'] = {k: model[k] for k in files}
    elif corrected is not None:
        report['bandPosesSha256'] = hashlib.sha256(args.band_poses.read_bytes()).hexdigest()
    xband, band_faces, wband = surfaces[BAND]
    xtfl, tfl_faces, _ = surfaces[tfl_name]
    tfl = xtfl[tfl_faces]
    face_weights = wband[band_faces]
    source_height = xband[band_faces, 1] - xband[:, 1].min()
    report['distalCutSourceYM'] = round(float(center[1]+xband[:,1].min()/1000), 6)
    band_zones = np.where(np.max(face_weights, axis=1) == 0, 'pelvisPinned',
                          np.where(np.min(face_weights, axis=1) == 1, 'distalPinned', 'blended'))
    contact_masks = {}
    if 'interfaces' in model and model.get('atlasSha256') == atlas_hash:
        interfaces = model['interfaces']
        indices = np.arange(len(band_faces))
        contact_masks[tfl_name] = np.isin(indices, interfaces['tflCoverFaceIds'])
        contact_masks['Hip bone'] = ~np.isin(indices, interfaces['iliacAttachmentFaces'])
        report['interfaceMaskSha256'] = hashlib.sha256(json.dumps(interfaces, sort_keys=True).encode()).hexdigest()
    reference_pairs = {}
    for label in obstacles:
        x, faces, _ = surfaces[label]
        ia, ib = candidate_pairs(xband[band_faces], x[faces])
        selected, _, _ = transverse_segments(xband[band_faces[ia]], x[faces[ib]])
        reference_pairs[label] = ia[selected] * len(faces) + ib[selected]
    area = np.linalg.norm(np.cross(tfl[:, 1]-tfl[:, 0], tfl[:, 2]-tfl[:, 0]), axis=1)/2
    mass = np.repeat(area/3, 3)
    original = enclosed_tfl(tfl, xband[band_faces])
    reference_area = mass[original].sum()
    report['tflInterface'] = dict(neutralEnclosedPercent=100*reference_area/mass.sum(),
                                  materialSamples=int(original.sum()), totalSamples=len(mass),
                                  pelvicPins=int((wband==0).sum()), distalPins=int((wband==1).sum()))
    for clip_id, frames in jobs:
        records = []
        for i, angles in frames:
            r = pose_rotation(angles)
            posed = {lab: femur_frame(x, w, r)[faces] for lab, (x, faces, w) in surfaces.items()}
            base_band = femur_frame(xband, wband, r)
            band = base_band
            if corrected is not None or solver is not None:
                x, triangles, _ = surfaces[BAND]
                xyz = solver.positions(angles) - center if solver is not None else corrected[clip_id][i]
                if xyz.shape != x.shape or not np.isfinite(xyz).all():
                    raise ValueError('Invalid band frame')
                band = r.inv().apply(xyz * 1000)
                posed[BAND] = band[triangles]
            hits = {}
            for lab in obstacles:
                metric = measure(posed[BAND], posed[lab], band_zones, reference_pairs[lab], contact_masks.get(lab), source_height)
                if metric['pairs']:
                    hits[lab] = metric
            inside = enclosed_tfl(posed[tfl_name], posed[BAND])
            delta = np.linalg.norm(band-base_band, axis=1)
            records.append(dict(frame=i, hits=hits, enclosedTflRetainedPct=round(100*mass[original & inside].sum()/reference_area, 4),
                                maxPelvicPinChangeMm=round(float(delta[wband==0].max()), 5),
                                maxDistalPinChangeMm=round(float(delta[wband==1].max()), 5)))
        report['clips'][clip_id] = records
        worst = max(records, key=lambda v: sum(m['lengthMm'] for m in v['hits'].values()))
        print(clip_id, 'frames with intersections', sum(bool(v['hits']) for v in records), '/', len(records),
              'worst-frame', json.dumps(worst), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
