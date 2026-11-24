#!/usr/bin/env python3
"""Extract right hip rest surfaces from the repository's Open3D Project lower-limb atlas.

Requires NumPy, SciPy, trimesh, embreex, DracoPy, mapbox-earcut and Pillow. Coordinates retain the source's metre,
Y-up, +Z-anterior convention.
"""
import argparse
import json
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import struct
from urllib.request import urlopen
from zipfile import ZipFile

import DracoPy
import mapbox_earcut
from hip_atlas_materials import attach_materials

import numpy as np
from trimesh import Trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

ROOT = Path(__file__).resolve().parents[2]
PUBLISHER = "https://caskanatomy.info/open3dmodelfiles/lower-limb/lower-limb-glb.zip"
SOURCE_HASH = "5ecb1ef4c07a2f6aa17021ee6f6d0bb0587595821467ec8c6a14f24ee1091173"
OUTPUT = ROOT / "upper-limb/hip-study/assets/hip-atlas.glb"
GROUPS = {
    "bones": [60, 80, 55],
    "joint": [137, 191, 127, 211],
    "abductors": [279, 280, 316],
    "fascia": [169, 168],
    "superficial": [278, 166],
    "rotators": [297, 314, 283, 293, 292, 304],
    "iliopsoas": [282, 302],
    "adductors": [258, 259, 260, 281, 295],
    "thigh": [308, 307, 321, 320, 322, 287, 311, 310, 265],
    "ligaments": [220, 248, 242, 241, 225, 182, 215, 212, 413, 436, 431, 418, 437],
    "nerves": [381, 334, 386, 341],
    "context": [67, 48, 132],
    "bursae": [123, 124, 125, 111, 97, 88, 95, 102, 103, 114, 115],
    "vessels": [6, 17, 18, 23, 28, 445, 451, 457, 460],
}
CUT_Y_M = 0.695
SUPERIOR_Y_M = 1.045
LAYER = {"Bones": "bones", "Cartilages": "cartilage", "Ligaments": "ligaments",
         "Overlays": "ligaments", "Bursae": "bursae", "Fascia": "fascia",
         "Muscles": "muscles", "Nerves": "nerves", "Arteries": "arteries", "Veins": "veins"}
# Fallback materials remain useful in standalone glTF viewers.
COLORS = {
    "bones": [0.87, 0.78, 0.64], "cartilage": [0.60, 0.83, 0.83],
    "muscles": [0.52, 0.16, 0.13], "tendons": [0.84, 0.79, 0.68],
    "fascia": [0.86, 0.79, 0.72], "ligaments": [0.75, 0.68, 0.48],
    "bursae": [0.20, 0.58, 0.65], "nerves": [0.93, 0.68, 0.27],
    "arteries": [0.66, 0.13, 0.10], "veins": [0.23, 0.35, 0.55],
}


def source_geometry(path):
    if path:
        content = path.read_bytes()
    else:
        with urlopen(PUBLISHER, timeout=90) as response:
            content = response.read()
    if content.startswith(b"PK"):
        with ZipFile(BytesIO(content)) as archive:
            content = archive.read("lower-limb.glb")
    if sha256(content).hexdigest() != SOURCE_HASH:
        raise ValueError("Original Open3D lower-limb asset checksum mismatch")
    size = struct.unpack_from("<I", content, 12)[0]
    return json.loads(content[20:20+size]), content[28+size:]


def unpack(doc, binary, primitive):
    compressed = primitive["extensions"]["KHR_draco_mesh_compression"]
    view = doc["bufferViews"][compressed["bufferView"]]
    start = view.get("byteOffset", 0)
    raw = DracoPy.decode(binary[start:start + view["byteLength"]])
    attrs = {k: raw.get_attribute_by_unique_id(i)["data"].astype(float)
             for k, i in compressed["attributes"].items()}
    return attrs, raw.faces.astype(int)


def closed_rims(segments, points):
    # Connect seam copies by physical position. A cut through an open surface can
    # end at its border; only closed loops can represent a section face.
    neighbors = defaultdict(set)
    representatives = {}
    def representative(idx):
        key = tuple(np.rint(np.asarray(points[idx]) * 1e7).astype(int))
        return representatives.setdefault(key, idx)
    for a, b in segments:
        a, b = representative(a), representative(b)
        if a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)
    seen = set()
    for start in neighbors:
        if start in seen:
            continue
        component, todo = set(), [start]
        while todo:
            current = todo.pop()
            if current not in component:
                component.add(current)
                todo.extend(neighbors[current] - component)
        seen.update(component)
        if len(component) < 3 or any(len(neighbors[i]) != 2 for i in component):
            continue
        rim = [start, min(neighbors[start])]
        while True:
            nxt, = neighbors[rim[-1]] - {rim[-2]}
            if nxt == start:
                break
            rim.append(nxt)
        yield rim


def clipped_surface(attrs, faces, level, keep_above):
    points = attrs["POSITION"]
    values = {k: v.tolist() for k, v in attrs.items()}
    inside = points[:, 1] >= level if keep_above else points[:, 1] <= level
    if inside.all():
        return attrs, faces, 0
    table = {}
    triangles = faces[inside[faces].all(axis=1)].tolist()
    segments = []
    def intersection(a, b):
        edge = tuple(sorted((a, b)))
        if edge not in table:
            idx = len(values["POSITION"])
            t = (level - points[a, 1]) / (points[b, 1] - points[a, 1])
            for key, initial in attrs.items():
                val = initial[a] + t * (initial[b] - initial[a])
                if key == "NORMAL":
                    val /= max(np.linalg.norm(val), 1e-12)
                values[key].append(val.tolist())
            table[edge] = idx
        return table[edge]
    crossing = inside[faces].sum(axis=1)
    for triangle in faces[(crossing == 1) | (crossing == 2)].tolist():
        polygon, cut = [], []
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            if inside[a]:
                polygon.append(a)
            if inside[a] != inside[b]:
                idx = intersection(a, b)
                polygon.append(idx)
                cut.append(idx)
        for j in range(1, len(polygon) - 1):
            triangles.append([polygon[0], polygon[j], polygon[j + 1]])
        segments.append(cut)

    # Each anatomical component has an independent section face. Open fascial
    # sheets retain an open edge. Section normals are perpendicular to the cut.
    cap_count = 0
    normal_y = -1 if keep_above else 1
    for rim in closed_rims(segments, values["POSITION"]):
        xyz = np.asarray([values["POSITION"][i] for i in rim])
        polygon = np.ascontiguousarray(xyz[:, [0, 2]])
        cap = mapbox_earcut.triangulate_float64(polygon, np.array([len(rim)], dtype=np.uint32)).reshape(-1, 3)
        if not len(cap):
            continue
        idx = len(values["POSITION"])
        signed = np.cross(xyz[cap[:,1]] - xyz[cap[:,0]], xyz[cap[:,2]] - xyz[cap[:,0]])[:, 1]
        cap[signed * normal_y < 0] = cap[signed * normal_y < 0, ::-1]
        for key, out in values.items():
            if key == "NORMAL":
                out.extend([[0, normal_y, 0]] * len(rim))
            else:
                out.extend([out[i] for i in rim])
        triangles.extend((cap + idx).tolist())
        cap_count += 1
    keep, remap = np.unique(np.asarray(triangles, dtype=int), return_inverse=True)
    return {k: np.asarray(v)[keep] for k, v in values.items()}, remap.reshape(-1, 3), cap_count


def fit_head(points):
    p = points.astype(np.float64)
    center = np.linalg.lstsq(np.column_stack((2 * p, np.ones(len(p)))), (p*p).sum(axis=1), rcond=None)[0][:3]
    radii = np.linalg.norm(p - center, axis=1)
    return {"centerM": center.tolist(), "radiusM": float(radii.mean()),
            "radialRmsM": float(radii.std()), "vertices": len(points),
            "method": "Algebraic least-squares sphere, all femoral-head cartilage vertices."}


def bone_shading(points, normals, faces):
    """Cosine-weighted, 96-ray hemispherical visibility of this bone alone."""
    tree = RayMeshIntersector(Trimesh(vertices=points, faces=faces, process=False))
    count = 96
    u = (np.arange(count) + 0.5) / count
    azimuth = np.arange(count) * np.pi * (3 - np.sqrt(5))
    disk_x = np.sqrt(u) * np.cos(azimuth)
    disk_y = np.sqrt(u) * np.sin(azimuth)
    disk_z = np.sqrt(1 - u)
    intensity = np.ones(len(points))
    for first in range(0, len(points), 512):
        p = points[first:first + 512]
        n = normals[first:first + 512]
        # Use a nonparallel reference axis to span each tangent plane.
        ref = np.where(np.abs(n[:, 0, None]) < 0.8, [1, 0, 0], [0, 0, 1])
        tx = np.cross(n, ref)
        tx /= np.linalg.norm(tx, axis=1, keepdims=True)
        ty = np.cross(n, tx)
        direction = tx[:, None] * disk_x[:, None] + ty[:, None] * disk_y[:, None] + n[:, None] * disk_z[:, None]
        origin = np.repeat(p + 0.00015 * n, count, axis=0)
        _, ray, hit = tree.intersects_id(origin, direction.reshape(-1, 3), multiple_hits=False, return_locations=True)
        obscuration = np.zeros(len(origin))
        if len(ray):
            distance = np.linalg.norm(hit - origin[ray], axis=1)
            obscuration[ray] = np.maximum(0, 1 - distance / 0.07)
        intensity[first:first + len(p)] -= 0.62 * obscuration.reshape(-1, count).mean(axis=1)
    return np.repeat(intensity[:, None], 3, axis=1).astype("<f4")


def export(source, output):
    doc, data = source_geometry(source)
    sorted_layers = list(COLORS)
    extras = {"source": PUBLISHER, "sourceGLBSHA256": SOURCE_HASH,
              "license": "https://creativecommons.org/licenses/by-sa/4.0/",
              "changes": "Selected regional rest surfaces in original coordinates, preserved UV seams and normals, applied planar proximal-thigh and upper-lumbar field boundaries and capped closed section contours. Replaced materials.",
              "boneShading": "Own-bone linear ambient accessibility, 96 cosine-weighted rays per vertex, 0.15 mm normal offset, 70 mm falloff, 0.62 maximum gain."}
    atlas = {"asset": {"version": "2.0", "generator": Path(__file__).name,
                       "copyright": "Open3D Project, Jan Kooloos, Eungyeol Lee and collaborators. CC BY-SA 4.0.", "extras": extras},
             "scene": 0, "scenes": [{"name":"Right hip atlas", "nodes":[]}], "nodes": [], "meshes": [],
             "materials": [{"name": k, "doubleSided": True, "pbrMetallicRoughness": {
                 "baseColorFactor": [*COLORS[k], 1], "roughnessFactor":0.78, "metallicFactor":0}}
                 for k in sorted_layers], "accessors": [], "bufferViews": []}
    parents = {child: group["name"] for group in doc["nodes"] if "children" in group for child in group["children"]}
    binary = bytearray()

    def append(values, element_type, kind, target, bounds=False):
        binary.extend(b"\x00" * (-len(binary) % 4))
        view_id = len(atlas["bufferViews"])
        atlas["bufferViews"].append({"buffer": 0, "byteLength": values.nbytes, "byteOffset": len(binary), "target": target})
        binary.extend(values.tobytes())
        acc = {"bufferView": view_id, "type": kind, "count": len(values), "componentType": element_type}
        if bounds:
            acc.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        index = len(atlas["accessors"])
        atlas["accessors"].append(acc)
        return index

    for region, entries in GROUPS.items():
        for source_id in entries:
            node = doc["nodes"][source_id]
            if any(k in node for k in ("matrix", "translation", "rotation", "scale")):
                raise ValueError(f"Unexpected coordinate transform on {node['name']}")
            primitive, = doc["meshes"][node["mesh"]]["primitives"]
            attrs, faces = unpack(doc, data, primitive)
            label = re.sub(r"\.r$", "", node["name"].replace("\u200b", "").strip()).strip()
            if source_id == 418:
                label = "Ischiofemoral ligament"
            group = parents[source_id]
            layer = LAYER[group]
            if source_id == 191:
                layer = "cartilage"
            if source_id == 265:
                layer = "tendons"
            meta = {"label": label, "regionGroup": region, "sourceNode": source_id,
                    "sourceLabel": node["name"], "sourceGroup": group,
                    "sourceMaterial": primitive["material"], "layer": layer}
            if source_id == 137:
                extras["femoralHeadSphere"] = fit_head(np.unique(attrs["POSITION"], axis=0))
            for y, above in ((CUT_Y_M, True), (SUPERIOR_Y_M, False)):
                if (attrs["POSITION"][:, 1].min() < y if above else attrs["POSITION"][:, 1].max() > y):
                    attrs, faces, caps = clipped_surface(attrs, faces, y, above)
                    if not len(faces):
                        raise ValueError(f"Region excludes all of {label}")
                    meta.setdefault("sections", []).append({"yM": y, "closedContoursCapped": caps})
            pos, nor = attrs["POSITION"], attrs["NORMAL"]
            indices = faces.reshape(-1)
            dtype, component = ("<u2", 5123) if indices.max() < 65536 else ("<u4", 5125)
            geometry = {"attributes": {}, "material": sorted_layers.index(layer),
                        "indices": append(indices.astype(dtype), component, "SCALAR", 34963)}
            if region == "bones":
                attrs["COLOR_0"] = bone_shading(pos, nor, faces)
            for key, values in attrs.items():
                geometry["attributes"][key] = append(values.astype("<f4"), 5126,
                    f"VEC{values.shape[1]}", 34962, key == "POSITION")
            atlas["scenes"][0]["nodes"].append(len(atlas["nodes"]))
            atlas["nodes"].append({"name": label, "mesh": len(atlas["meshes"]), "extras": meta})
            atlas["meshes"].append({"name": label, "primitives": [geometry]})

    attach_materials(atlas, binary, doc, data)
    binary.extend(b"\x00" * (-len(binary) % 4))
    atlas["buffers"] = [{"byteLength": len(binary)}]
    header = json.dumps(atlas, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 4)
    output.parent.mkdir(exist_ok=True, parents=True)
    with output.open("wb") as f:
        f.write(struct.pack("<4sIII4s", b"glTF", 2, 28 + len(header) + len(binary), len(header), b"JSON"))
        f.write(header)
        f.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        f.write(binary)
    print(output, extras["femoralHeadSphere"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Publisher ZIP or raw GLB (downloads pinned revision when omitted)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    export(args.source, args.output)
