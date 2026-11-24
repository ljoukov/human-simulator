"""Track a reference patch of iliotibial tract beside the trochanteric bursa.

Requires numpy, scipy, trimesh and the native hip contact solver. Run after
preparing the atlas, its attachment fields and its contact model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from prepare_hip_contact import HipContact, C_SOURCE, MODEL, WASM, read_weights
from prepare_hip_deformation import ATLAS, DATA, UNIT, deformation, load_meshes

BURSA = 'Trochanteric bursa of gluteus maximus'
BAND = 'Iliotibial tract'
THREE_POINTS = np.eye(3) / 2 + 1/6
POSES = [[0, 0, 0], [0, -20, 0], [0, 20, 0], [30, 0, 0], [0, 0, 20],
         [30, 15, -15], [75, 30, 30], [-30, -25, -30]]


def sha256(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def closest_surface(queries, triangles):
    # Exhaustive, exact closest point on each triangle, including edge and vertex cases.
    ids = []
    closest = []
    inside = []
    boundary = []
    for chunk in np.array_split(queries, max(1, (len(queries) + 15) // 16)):
        candidate = trimesh.triangles.closest_point(
            np.tile(triangles, (len(chunk), 1, 1)), np.repeat(chunk, len(triangles), axis=0)
        ).reshape(len(chunk), len(triangles), 3)
        selected = np.square(candidate - chunk[:, None]).sum(axis=2).argmin(axis=1)
        ids.extend(selected)
        matched = candidate[np.arange(len(chunk)), selected]
        closest.extend(matched)
        on_surface = np.linalg.norm(matched-chunk, axis=1) <= 1e-6
        boundary.extend(on_surface)
        # Solid angle winding for the closed, consistently oriented ITB solid.
        v = triangles[None] - chunk[:, None, None]
        u = v / np.maximum(np.linalg.norm(v, axis=-1)[..., None], 1e-30)
        a, b, c = u[:, :, 0], u[:, :, 1], u[:, :, 2]
        numerator = (a * np.cross(b, c)).sum(axis=-1)
        denominator = 1 + (a*b).sum(axis=-1) + (b*c).sum(axis=-1) + (c*a).sum(axis=-1)
        winding = np.arctan2(numerator, denominator).sum(axis=1) / (2*np.pi)
        if np.any(np.abs(winding[~on_surface] - np.rint(winding[~on_surface])) > 1e-6):
            raise ValueError('Unresolved winding number near band boundary')
        inside.extend((np.abs(winding) > .5) & ~on_surface)
    return np.array(ids), np.array(closest), np.array(inside), np.array(boundary)


def ray_parity(points, triangles):
    """Independent interior check by counting positive ray/triangle crossings."""
    direction = np.array([.317, .847, -.427])
    a, b, c = triangles.transpose(1, 0, 2)
    e1, e2 = b - a, c - a
    h = np.cross(direction, e2)
    det = (e1 * h).sum(axis=1)
    valid = np.abs(det) > 1e-13
    inv_det = np.divide(1, det, out=np.zeros_like(det), where=valid)
    s = points[:, None] - a
    u = (s*h).sum(axis=-1) * inv_det
    q = np.cross(s, e1)
    v = (q*direction).sum(axis=-1) * inv_det
    t = (q*e2).sum(axis=-1) * inv_det
    crossings = valid & (u > 0) & (v > 0) & (u+v < 1) & (t > 0)
    return crossings.sum(axis=1) % 2 == 1


def build_patch(bursa, band, rule):
    cross = np.cross(bursa.triangles[:, 1] - bursa.triangles[:, 0],
                     bursa.triangles[:, 2] - bursa.triangles[:, 0])
    # Right atlas has negative X outward. Projected area naturally downweights
    # the thin lateral rim, and the medial side contributes zero.
    projected = -cross[:, 0] / 2
    keep = projected > 0
    rest = (rule @ bursa.triangles[keep]).reshape(-1, 3)
    mass = np.repeat(projected[keep] / len(rule), len(rule))
    normals = np.repeat(bursa.face_normals[keep], len(rule), axis=0)
    faces, closest, inside, boundary = closest_surface(rest, band.triangles)
    bary = trimesh.triangles.points_to_barycentric(band.triangles[faces], closest)
    if np.min(bary) < -1e-7 or np.max(bary) > 1 + 1e-7:
        raise ValueError('Invalid ITB point')
    ids = band.faces[faces]
    return dict(rest=rest, mass=mass, normals=normals, ids=ids, bary=bary,
                closest=closest, inside=inside, boundary=boundary, faces=int(keep.sum()))


def rms_tangent(delta, normals, mass):
    tangent = delta - (delta * normals).sum(axis=1)[:, None] * normals
    return float(1000*np.sqrt(np.average(np.square(tangent).sum(axis=1), weights=mass)))


def hip_rotation(pose):
    return Rotation.from_euler('ZXY', [pose[1], -pose[0], pose[2]], degrees=True)


def material_displacement(patch, vertices):
    return (patch['bary'][:, :, None] * vertices[patch['ids']]).sum(axis=1)


def relative_travel(patch, vertex_delta):
    return rms_tangent(material_displacement(patch, vertex_delta), patch['normals'], patch['mass'])


def direct_travel(patch, world, rot, center):
    # Independent formulation in the fixed pelvis frame. Take the present
    # band-to-bursa vector and remove the rotated neutral separation before
    # projecting on the moving bursa's current face normals.
    now = material_displacement(patch, world)
    bursa = center + rot.apply(patch['rest'] - center)
    delta = now - bursa - rot.apply(patch['closest'] - patch['rest'])
    return rms_tangent(delta, rot.apply(patch['normals']), patch['mass'])


def refined_rule():
    # Four equal-area subtriangles, each carrying the three-point quadrature.
    a, b, c = np.eye(3)
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return np.concatenate([THREE_POINTS @ t for t in (
        [a,ab,ca], [ab,b,bc], [ca,bc,c], [ab,bc,ca])])


def compact_evaluator(patch, neutral):
    vertices, inverse = np.unique(patch['ids'], return_inverse=True)
    triangles = inverse.reshape(patch['ids'].shape)
    records = np.column_stack((patch['mass']/patch['mass'].sum(), patch['normals'],
                               triangles, patch['bary']))
    return dict(bandVertexCount=len(neutral),
                vertexColumns=['weldedBandId', 'neutralXM', 'neutralYM', 'neutralZM'],
                sampleColumns=['areaFraction', 'nx', 'ny', 'nz', 'vertex0', 'vertex1',
                               'vertex2', 'barycentric0', 'barycentric1', 'barycentric2'],
                vertices=np.column_stack((vertices, neutral[vertices])).round(12).tolist(),
                samples=records.round(10).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', type=Path, default=ATLAS)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=DATA/'bursa-mechanics.json')
    args = parser.parse_args()
    meshes, center, atlas_hash = load_meshes(args.atlas)
    manifest = json.loads((args.data/'hip-deformation.json').read_text())
    binary = (args.data/'hip-attachments.bin').read_bytes()
    if atlas_hash != manifest['sourceAtlasSha256'] or hashlib.sha256(binary).hexdigest() != manifest['weightsSha256']:
        raise ValueError('Rebuild the attachment field against the current atlas')
    if manifest['surfaces'][BURSA].get('constant') != 1:
        raise ValueError('Bursal reference frame requires rigid femur attachment')
    def mesh(label):
        key, triangles, _ = meshes[label]
        obj = trimesh.Trimesh(key / UNIT, triangles, process=False)
        if not obj.is_volume:
            raise ValueError('Expected closed outward-facing mesh for '+label)
        return obj

    bursa, band = mesh(BURSA), mesh(BAND)
    w = read_weights(BAND, meshes[BAND][0], manifest, binary)
    xyz = band.vertices - center
    contact_source_hash, contact_data_hash = sha256(C_SOURCE), sha256(MODEL)
    wasm_hash = sha256(WASM)
    contact_manifest = json.loads((args.data/'hip-contact.json').read_text())
    recorded = tuple(contact_manifest[k] for k in ('solverSourceSha256','contactInputSha256','wasmSha256'))
    if recorded != (contact_source_hash, contact_data_hash, wasm_hash):
        raise ValueError('Rebuild the matching native and browser contact artifacts first')
    contact = HipContact()
    if not np.allclose(contact.rest, xyz, atol=1e-8, rtol=0) or not np.array_equal(contact.triangles, band.faces):
        raise ValueError('Contact solver and rendered atlas differ in IT-band vertex ordering or topology')
    neutral_world = np.asarray(contact.positions(POSES[0])).copy()
    if neutral_world.shape != band.vertices.shape or np.linalg.norm(neutral_world - band.vertices, axis=1).max() > .1:
        raise ValueError('Wrong band coordinate frame or welded vertex order in contact solver')
    neutral = neutral_world - center
    corrected_band = trimesh.Trimesh(neutral_world, band.faces, process=False)
    patch = build_patch(bursa, corrected_band, THREE_POINTS)
    finer = build_patch(bursa, corrected_band, refined_rule())
    centroid = build_patch(bursa, corrected_band, np.ones((1, 3))/3)
    raw = build_patch(bursa, band, THREE_POINTS)

    # Cache coordinates once for the direct check and quadrature controls. A
    # native solver may reuse its buffers on the next call, hence the copy.
    history = {}
    def evaluate(pose):
        key = tuple(pose)
        if key not in history:
            rot = hip_rotation(pose)
            world = np.asarray(contact.positions(pose)).copy()
            delta = rot.inv().apply(world - center) - neutral
            old_delta = deformation(xyz, w-1, pose) - xyz
            history[key] = {
                'value':relative_travel(patch, delta),
                'world':direct_travel(patch, world, rot, center),
                'finer':relative_travel(finer, delta),
                'centroid':relative_travel(centroid, delta),
                'uncorrected':relative_travel(raw, old_delta),
            }
        return history[key]

    def overlap_check(pose):
        # Cross-check the contact model on the actual facing side of the bursa.
        # This is quadrature on a region, independent of a mesh intersection
        # test on the full shell, and it cannot certify zero intersections.
        rot = hip_rotation(pose)
        band_rest_frame = center + rot.inv().apply(np.asarray(contact.positions(pose)) - center)
        _, _, inside, on_surface = closest_surface(patch['rest'], band_rest_frame[band.faces])
        return {
            'sampledAreaInsideBandFraction':float(np.average(inside, weights=patch['mass'])),
            'sampledAreaOnBandFraction':float(np.average(on_surface, weights=patch['mass'])),
        }

    clip_checks = []
    all_poses = list(POSES)
    motion_file = args.data/'motion.json'
    for clip in json.loads(motion_file.read_text())['clips']:
        poses = [[f.get(key, 0) for key in ('hipFlexionDeg','hipAdductionDeg','hipRotationDeg')]
                 for f in clip['frames']]
        values = np.array([evaluate(p)['value'] for p in poses])
        uncoupled = np.array([evaluate(p)['uncorrected'] for p in poses])
        source_phases = [f['phase'] for f in clip['frames']]
        chart_phases = np.linspace(0, 1, 1001)
        interpolated = np.column_stack([np.interp(chart_phases, source_phases, angle)
                                        for angle in np.asarray(poses).T])
        # The UI interpolates source angles first. Evaluate the contact solve
        # at these poses, rather than interpolating its nonlinear RMS output.
        chart = []
        for pose in interpolated:
            world = np.asarray(contact.positions(pose))
            delta = hip_rotation(pose).inv().apply(world - center) - neutral
            chart.append(relative_travel(patch, delta))
        chart = np.asarray(chart)
        i = int(values.argmax())
        clip_checks.append(dict(clip=clip['id'], frames=len(values), excursionMm=values.round(4).tolist(),
                                chartExcursionMm=chart.round(4).tolist(),
                                sourceTraceLinearInterpolationErrorMm=float(np.max(np.abs(
                                    chart-np.interp(chart_phases, source_phases, values)))),
                                rangeMm=values[[values.argmin(), i]].round(4).tolist(),
                                maxChangeFromUncorrectedMm=round(float(np.max(np.abs(values-uncoupled))), 6),
                                peakPhase=clip['frames'][i]['phase'], peakPoseDeg=poses[i],
                                overlapAtPeak=overlap_check(poses[i])))
        all_poses += poses
    for p in POSES:
        evaluate(p)
    checks = list(history.values())
    direct_error = max(abs(d['value'] - d['world']) for d in checks)
    refine_error = max(abs(d['value'] - d['finer']) for d in checks)
    if direct_error > 1e-8 or evaluate(POSES[0])['value'] > 1e-7:
        raise AssertionError('Bursa frame and world-frame evaluations disagree')
    strict = ~finer['boundary']
    if not np.array_equal(ray_parity(finer['rest'][strict], corrected_band.triangles), finer['inside'][strict]):
        raise AssertionError('Winding number and independent ray classification disagree')
    def inside_frac(p):
        return float(np.average(p['inside'], weights=p['mass']))

    # Pick an actual vertex near the projected-area centroid of the lateral surface.
    centroid3 = np.average(patch['rest'], axis=0, weights=patch['mass'])
    idx = int(np.linalg.norm(bursa.vertices-centroid3, axis=1).argmin())
    result = dict(
        version=2, metric='IT-band tangential travel from neutral, projected-area-weighted RMS (mm)',
        bursa=BURSA, band=BAND, landmarkM=bursa.vertices[idx].tolist(),
        source=dict(url='https://caskanatomy.info/open3dmodelfiles/lower-limb/lower-limb-glb.zip',
                    anatomy='https://pubmed.ncbi.nlm.nih.gov/18245587/',
                    publisherGlbSha256='5ecb1ef4c07a2f6aa17021ee6f6d0bb0587595821467ec8c6a14f24ee1091173',
                    atlasSha256=atlas_hash, attachmentSha256=manifest['weightsSha256'], motionSha256=sha256(motion_file),
                    contactSourceSha256=contact_source_hash, contactInputSha256=contact_data_hash,
                    contactWasmSha256=wasm_hash),
        reference=dict(bursaTriangles=len(bursa.faces), bandTriangles=len(band.faces), lateralFacingTriangles=patch['faces'],
                       samples=len(patch['rest']), projectedAreaMm2=float(1e6*patch['mass'].sum()),
                       fractionProjectedAreaInsideRawBandAtNeutral=inside_frac(raw),
                       fractionProjectedAreaInsideCorrectedBandAtNeutral=inside_frac(patch),
                       refinedFractionInsideAtNeutral=inside_frac(finer),
                       fractionProjectedAreaAtCorrectedBoundary=float(np.average(patch['boundary'],weights=patch['mass'])),
                       boundaryToleranceMm=.001,
                       correctedNeutralUnsignedDistanceRangeMm=(1000*np.quantile(np.linalg.norm(patch['rest']-patch['closest'],axis=1),[0,1])).tolist()),
        method='At neutral, match three samples per outward lateral-facing bursa triangle to the closest points on the corrected IT-band surface. At each pose apply the same geometric contact solver as the renderer, track those same band material points and subtract the corrected neutral separation in the moving bursa frame. Project the displacement onto each bursa triangle tangent plane. Report RMS magnitude weighted by projected lateral bursa area.',
        interpretation='Kinematic excursion from an atlas with modeled tissue attachments and geometric contact correction. Neutral registration, triangulation and layer order affect the value. This calculation has no tissue constitutive law or pressure, friction, strain or inflammation measurement.',
        validation=dict(posesChecked=len(all_poses), independentWorldFrameMaxErrorMm=direct_error,
                        allRigidNullMaxMm=max(direct_travel(patch, center+hip_rotation(p).apply(neutral),
                                                          hip_rotation(p), center) for p in POSES),
                        twelvePointMaxDifferenceMm=refine_error, neutralMm=evaluate(POSES[0])['value'],
                        rayAndWindingClassificationsAgreed=int(strict.sum()),
                        centroidMaxDifferenceMm=max(abs(d['value']-d['centroid']) for d in checks),
                        maxDifferenceFromUncorrectedMm=max(abs(d['value']-d['uncorrected']) for d in checks)),
        poses=[dict(flexionAdductionInternalRotationDeg=p, excursionMm=round(evaluate(p)['value'],6),
                    uncorrectedMm=round(evaluate(p)['uncorrected'],6),
                    overlap=overlap_check(p)) for p in POSES],
        chartSampling='1001 uniformly spaced phases per clip. Hip angles are linearly interpolated from source frames; material excursion is recomputed by solving contact at each interpolated pose.',
        clipChecks=clip_checks, evaluator=compact_evaluator(patch, neutral))
    if (sha256(C_SOURCE), sha256(MODEL), sha256(WASM)) != recorded:
        raise RuntimeError('Contact model changed during the run; regenerate the bursa measurements.')
    args.output.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    for name in ('reference','validation','poses','landmarkM'):
        print(name,result[name])


if __name__ == '__main__':
    main()
