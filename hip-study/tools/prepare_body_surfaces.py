"""Reproduce the bilateral hip surfaces registered to the BodyParts3D skeleton.

Requires numpy, trimesh, and fast-simplification. Supply a local input cache to
reuse the labeled BodyParts3D STL files. The study stores the derived mesh.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import trimesh

REVISION = 'f0eeb6e843380cfe6b83797cf8c3e1af74de5e61'
BASE = f'https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/{REVISION}/assets/BodyParts3D_data/stl'
MESHES = {
    # Right and left FMA labels followed by the per-surface triangle budget.
    'maximus': ('FMA22328', 'FMA22329', 5500),
    'medius': ('FMA22330', 'FMA22331', 4500),
    'minimus': ('FMA22332', 'FMA22333', 4300),
    'tensor': ('FMA22425', 'FMA22426', 3000),
    'band': ('FMA58776', 'FMA58777', 4500),
    'iliacus': ('FMA22322', 'FMA22323', 3500),
    'psoas': ('FMA22342', 'FMA22343', 4000),
    'piriformis': ('FMA22340', 'FMA22341', 2200),
    'quadratus': ('FMA22338', 'FMA22339', 2200),
    'obturator': ('FMA22324', 'FMA22325', 2600),
    'pectineus': ('FMA22450', 'FMA22451', 2500),
    'adductorLongus': ('FMA22456', 'FMA22457', 3500),
    'adductorMagnus': ('FMA22459', 'FMA22460', 4900),
    'rectus': ('FMA38928', 'FMA38929', 4100),
    'vastus': ('FMA38930', 'FMA38931', 4700),
    'semitendinosus': ('FMA22358', 'FMA22359', 3500),
}
# These are the transform fields from the OrganUI model's pinned manifest.
CENTER_MM = np.array([-0.667999267578125, -109.74665069580078, 817.0244853496552])


def generate(cache, output):
    cache.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for tissue, (right_id, left_id, triangle_budget) in MESHES.items():
        for side, fma in (('right', right_id), ('left', left_id)):
            cache_file = cache / f'{fma}.stl'
            if not cache_file.exists():
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                f'{BASE}/{fma}.stl', '--output', str(cache_file)], check=True)
            mesh = trimesh.load(cache_file, force='mesh')
            mesh = mesh.simplify_quadric_decimation(face_count=triangle_budget)
            points = (mesh.vertices - CENTER_MM) * 0.0025
            mesh.vertices = points[:, [0, 2, 1]] * [1, 1, -1]
            name = f'{side}_{tissue}'
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.metadata['credit'] = 'BodyParts3D, copyright DBCLS, CC BY-SA 2.1 Japan'
    output.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'assets/body-soft-tissue.glb')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        generate(args.cache or Path(tmp), args.output)
