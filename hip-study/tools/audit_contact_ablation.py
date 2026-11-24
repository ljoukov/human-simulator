"""Measure the two changes to the OA51 outer-hip artifact on identical viewing rays.

Run with the geometry-preparation dependencies (NumPy, SciPy, trimesh, embreex and
clang). By default, reconstruct the original attachment field in a temporary folder.
An existing `prepare_hip_deformation.py --uncoupled-tfl` output can be passed with
--legacy-data. The atlas, both production attachment files and the solver stay intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation

import prepare_hip_contact as contact
from prepare_hip_deformation import ATLAS, DATA, UNIT, deformation, load_meshes

BAND, TFL = 'Iliotibial tract', 'Tensor fasciae latae'
CENTERS = np.array([[422, 251], [422, 269], [433, 299]])  # Top-left-origin pixel indices.
CLIPS = [('bertaux-hoa001', 51), ('predsim-hip-50', 20)]
CLEARANCES_MM = (.125, .25, .5)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_weights(directory, meshes, atlas_hash):
    manifest = json.loads((directory / 'hip-deformation.json').read_text())
    binary = (directory / 'hip-attachments.bin').read_bytes()
    digest_value = hashlib.sha256(binary).hexdigest()
    if manifest['sourceAtlasSha256'] != atlas_hash or manifest['weightsSha256'] != digest_value:
        raise ValueError(f'Attachment inputs mismatch in {directory}')
    fields = {name: contact.read_weights(name, meshes[name][0], manifest, binary) for name in (BAND, TFL)}
    return fields, digest_value


def camera(pixels):
    """Orthographic view in the surface pane, 800 × 550, right hip, yaw +26°, elevation +10°."""
    target = np.array([.003, -.004, -.021])
    yaw, elevation = np.deg2rad([26, 10])
    back = np.array([-np.cos(yaw)*np.cos(elevation), np.sin(elevation), np.sin(yaw)*np.cos(elevation)])
    right = np.cross([0, 1, 0], back)
    right /= np.linalg.norm(right)
    view = np.column_stack([right, np.cross(back, right), back])
    offsets = (pixels + .5 - [400, 275]) * [1, -1] * (.342 / 550)
    return offsets + target @ view[:, :2], view


def frontmost(vertices, triangles, pixels):
    """Largest depth toward the camera at an in-triangle barycentric intersection."""
    queries, view = camera(pixels)
    projected = vertices[triangles] @ view
    origin = projected[:, 0]
    ab = projected[:, 1] - origin
    ac = projected[:, 2] - origin
    determinant = ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]
    valid = abs(determinant) > 1e-15
    inv = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
    depths = []
    for start in range(0, len(queries), 50):
        delta = queries[start:start+50, None] - origin[None, :, :2]
        u = (delta[:, :, 0]*ac[:, 1] - delta[:, :, 1]*ac[:, 0])*inv
        v = (ab[:, 0]*delta[:, :, 1] - ab[:, 1]*delta[:, :, 0])*inv
        inside = valid & (u >= 0) & (v >= 0) & (u+v <= 1)
        z = origin[:, 2] + ab[:, 2]*u + ac[:, 2]*v
        depths.extend(np.where(inside, z, -np.inf).max(axis=1))
    return np.array(depths)


def posed(surfaces, frame, weights, solver=None):
    angles = [frame.get(key, 0) for key in ('hipFlexionDeg', 'hipAdductionDeg', 'hipRotationDeg')]
    pelvis = Rotation.from_euler('ZXY', [frame.get(key, 0) for key in
                                 ('pelvisObliquityDeg', 'pelvisTiltDeg', 'pelvisRotationDeg')], degrees=True)
    geometry = {}
    for name, (rest, faces) in surfaces.items():
        vertices = deformation(rest, weights[name], angles)
        if name == BAND and solver is not None:
            vertices = solver.positions(angles) - solver.center
        geometry[name] = (pelvis.apply(vertices), faces)
    return geometry


def fixed_patch_rays(geometry):
    # Include neighboring intact band so every selected component has a closed boundary.
    x0, y0, x1, y1 = 400, 195, 465, 320
    rows, cols = np.indices((y1-y0, x1-x0))
    pixels = np.column_stack([cols.ravel()+x0, rows.ravel()+y0])
    band = frontmost(*geometry[BAND], pixels)
    tfl = frontmost(*geometry[TFL], pixels)
    mask = (np.isfinite(band) & (tfl > band)).reshape(rows.shape)
    components, _ = label(mask)  # Four-neighbour connectivity in the pixel grid.
    selected = components[CENTERS[:, 1]-y0, CENTERS[:, 0]-x0]
    boundary = np.concatenate([components[0], components[-1], components[:, 0], components[:, -1]])
    if len(set(selected)) != 3 or 0 in selected or np.isin(selected, boundary).any():
        raise ValueError('Marked rays must identify three separate, enclosed original patches.')
    keep = np.isin(components.ravel(), selected)
    counts = [int((components == component).sum()) for component in selected]
    return pixels[keep], counts


def clearance_variants(directory):
    # Each variant changes only the C solver's clearance literal in an isolated build.
    source = contact.C_SOURCE.read_text()
    models = {}
    for clearance in CLEARANCES_MM:
        edited, count = re.subn(r'clearance\s*=\s*\.00025\b', f'clearance={clearance/1000:.6f}', source)
        if count != 1:
            raise ValueError('Expected one 0.25 mm contact clearance in the solver source.')
        variant = directory / f'clearance-{clearance}.c'
        variant.write_text(edited)
        with patch.object(contact, 'C_SOURCE', variant):
            models[clearance] = contact.HipContact()
    return models


def run(directory, legacy):
    if legacy is None:
        legacy = directory / 'legacy'
        subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_hip_deformation.py')),
                        '--uncoupled-tfl', '--output', str(legacy)], stdout=subprocess.DEVNULL, check=True)
    meshes, center, atlas_hash = load_meshes(ATLAS)
    surfaces = {name: (meshes[name][0]/UNIT-center, meshes[name][1]) for name in (BAND, TFL)}
    old, original_hash = load_weights(legacy, meshes, atlas_hash)
    new, revised_hash = load_weights(DATA, meshes, atlas_hash)
    if not np.array_equal(old[BAND], new[BAND]) or np.array_equal(old[TFL], new[TFL]):
        raise ValueError('Expected unchanged band attachments and a revised TFL attachment field.')
    model = json.loads((DATA/'hip-contact.json').read_text())
    provenance = {'atlasSha256': atlas_hash, 'attachmentsSha256': revised_hash,
                  'solverSourceSha256': digest(contact.C_SOURCE), 'contactInputSha256': digest(contact.MODEL),
                  'wasmSha256': digest(contact.WASM)}
    if any(model[key] != value for key, value in provenance.items()):
        raise ValueError('The installed contact assets have different source hashes. Rebuild them first.')
    provenance.update(legacyAttachmentsSha256=original_hash, motionSha256=digest(DATA/'motion.json'),
                      auditSourceSha256=digest(Path(__file__)))
    clips = {clip['id']: clip for clip in json.loads((DATA/'motion.json').read_text())['clips']}
    first_clip, index = CLIPS[0]
    pixels, counts = fixed_patch_rays(posed(surfaces, clips[first_clip]['frames'][index], old))
    rays = np.vstack([CENTERS, pixels])
    variants = [('original', old, None, None), ('revisedAttachmentOnly', new, None, None)]
    for clearance, solver in clearance_variants(directory).items():
        variants.append(('revisedAttachmentAndContact', new, clearance, solver))
    frames = []
    for clip_id, index in CLIPS:
        if clips[clip_id]['side'] != 'right':
            raise ValueError('The camera fixture assumes the right hip.')
        frame = clips[clip_id]['frames'][index]
        entries = []
        for condition, weights, clearance, solver in variants:
            geometry = posed(surfaces, frame, weights, solver)
            depths = [frontmost(*geometry[name], rays) for name in (BAND, TFL)]
            if not np.isfinite(depths).all():
                raise ValueError('A marked camera ray missed one of the two tissues.')
            separation = 1000 * (depths[0]-depths[1])
            entries.append(dict(condition=condition, clearanceMm=clearance,
                                centersBandAheadMm=separation[:3].round(5).tolist(),
                                minBandAheadMm=round(float(separation[3:].min()), 5),
                                tflAheadRayCount=int(np.count_nonzero(separation[3:] < 0))))
        frames.append(dict(clip=clip_id, frame=index, phase=frame['phase'], conditions=entries))
    return dict(provenance=provenance, camera=dict(viewportPx=[800, 550], halfHeightM=.171,
                targetM=[.003, -.004, -.021], yawFromLateralTowardAnteriorDeg=26, elevationDeg=10),
                raySet=dict(referenceClip=first_clip, referenceFrame=CLIPS[0][1],
                    definition='Four-connected components of pixels in the original OA frame where the TFL surface lies closer to the camera than the tract. Each component contains one marked pixel and is fully inside the search rectangle. The same screen rays are retained for every condition and the contrast frame.',
                    topLeftOriginPixelIndices=CENTERS.tolist(), sampleAt='pixel centre (x+0.5, y+0.5)',
                    componentPixelCounts=counts, totalRays=len(pixels),
                    pixelSetSha256=hashlib.sha256(pixels.astype('<i4').tobytes()).hexdigest()),
                metric='Signed orthographic distance along the camera ray, in mm. A positive value places the tract ahead of TFL. Both surfaces are tested independently on the atlas triangles with welded coordinates rounded to 1 micrometre. These screen-ray distances differ from clearance along the solver axis.',
                clearancesMm=list(CLEARANCES_MM), frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-data', type=Path, help='Existing directory containing uncoupled attachment files.')
    parser.add_argument('--output', type=Path, default=DATA/'contact-ablation.json')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        report = run(Path(directory), args.legacy_data)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f"Saved {len(report['frames'])} frames and {report['raySet']['totalRays']} fixed rays to {args.output}")


if __name__ == '__main__':
    main()
