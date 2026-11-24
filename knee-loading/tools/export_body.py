#!/usr/bin/env python3
"""Convert the full-body OpenSim display geometry for the GC4 gait study to GLB."""

from argparse import ArgumentParser
from array import array
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import struct
from subprocess import run
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote
from xml.etree import ElementTree


SOURCE_REVISION = "d9b05d470b1a481c222372c85b75772faf8f7792"
MODEL_REVISION = "00b6f94ee8bd2f60ae5b6f07a114f1aade089cb8"
MODEL_SOURCE_PATH = (
    "MuscleTendonLengthParameterizationReduction/"
    "JW All Body Model-scaled_forvisualization_leftright_v42.osim"
)
MODEL_ONLY = {"femoral_component", "tibial_tray"}
POSITION_TARGET = 34962
INDEX_TARGET = 34963


def read_piece(path):
    root = ElementTree.parse(path).getroot()
    piece = root.find("./PolyData/Piece")
    if piece is None or root.get("type") != "PolyData":
        raise ValueError(f"Expected VTK PolyData in {path}")
    return piece


def values(element, cast):
    if element is None or not element.text or element.get("format") != "ascii":
        raise ValueError("Expected an ASCII VTK data array")
    return [cast(token) for token in element.text.split()]


def load_mesh(path, scale, reflected=False):
    piece = read_piece(path)
    position_data = values(piece.find("./Points/DataArray"), float)
    vertices = list(zip(
        (x * scale[0] for x in position_data[0::3]),
        (y * scale[1] for y in position_data[1::3]),
        (z * scale[2] for z in position_data[2::3]),
        strict=True,
    ))
    polygon_vertices = values(piece.find("./Polys/DataArray[@Name='connectivity']"), int)
    polygon_ends = values(piece.find("./Polys/DataArray[@Name='offsets']"), int)
    if (len(vertices) != int(piece.get("NumberOfPoints"))
            or len(polygon_ends) != int(piece.get("NumberOfPolys"))
            or len(polygon_vertices) != polygon_ends[-1]):
        raise ValueError(f"Inconsistent mesh dimensions in {path}")
    if any(index < 0 or index >= len(vertices) for index in polygon_vertices):
        raise ValueError(f"Out-of-bounds polygon index in {path}")

    triangles = []
    start = 0
    for end in polygon_ends:
        polygon = polygon_vertices[start:end]
        if len(polygon) < 3:
            raise ValueError(f"Invalid polygon with {len(polygon)} vertices in {path}")
        triangles.extend((polygon[0], a, b) for a, b in zip(polygon[1:-1], polygon[2:], strict=True))
        start = end
    if reflected:
        vertices = [(x, y, -z) for x, y, z in vertices]
        triangles = [(a, c, b) for a, b, c in triangles]

    # These source meshes often duplicate a vertex at a polygon boundary.
    # Share normals at coincident points without deleting the source vertices.
    normal_sum = defaultdict(lambda: [0.0, 0.0, 0.0])
    for i, j, k in triangles:
        a, b, c = vertices[i], vertices[j], vertices[k]
        ab = [v - u for u, v in zip(a, b, strict=True)]
        ac = [v - u for u, v in zip(a, c, strict=True)]
        normal = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        for vertex in (a, b, c):
            summed = normal_sum[vertex]
            for axis, component in enumerate(normal):
                summed[axis] += component

    normals = []
    for vertex in vertices:
        summed = normal_sum[vertex]
        magnitude = math.hypot(*summed)
        normals.extend((component / magnitude for component in summed) if magnitude else (0, 1, 0))
    return vertices, normals, triangles


def little_endian(code, values):
    result = array(code, values)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def align(chunk, byte=b"\x00"):
    return chunk + byte * ((-len(chunk)) % 4)


def display_mesh_name(filename):
    names = {
        "hat_jaw": "Mandible",
        "hat_skull": "Skull",
        "hat_spine": "Spine",
        "hat_ribs_scap": "Rib cage and scapulae",
        "pat": "Right patella",
        "foot": "Right hindfoot and midfoot",
        "bofoot": "Right forefoot",
        "l_foot": "Left hindfoot and midfoot",
        "l_bofoot": "Left forefoot",
    }
    stem = Path(filename).stem
    if stem in names:
        return names[stem]
    suffix_start = stem.rfind("_")
    suffix = stem[suffix_start + 1:]
    if suffix in {"r", "rv", "rvs", "l", "lv", "lvs"}:
        return f"{'Right' if suffix.startswith('r') else 'Left'} {stem[:suffix_start].replace('_', ' ')}"
    return stem.replace("_", " ").capitalize()


def export_body(model_path, geometry_directory, output_path):
    document = {
        "asset": {
            "version": "2.0",
            "generator": "Human Simulator OpenSim VTP exporter",
            "extras": {
                "displayGeometry": f"https://github.com/opensim-org/opensim-models/tree/{SOURCE_REVISION}/Geometry",
                "bodyFrames": f"https://github.com/gilserrancoli/ReducePoly_forMSK/tree/{MODEL_REVISION}",
                "coordinates": "Meters in each OpenSim body frame. Root-level body nodes are posed by the application.",
                "pelvis": "The source hemipelvis is mirrored for the left side. The reference sacrum is attached to the pelvis.",
            },
        },
        "scene": 0,
        "scenes": [{"name": "GC4 full-body reference", "nodes": []}],
        "nodes": [],
        "meshes": [],
        "materials": [{
            "name": "Bone",
            "pbrMetallicRoughness": {"baseColorFactor": [0.82, 0.76, 0.63, 1], "metallicFactor": 0, "roughnessFactor": 0.85},
        }],
        "accessors": [],
        "bufferViews": [],
    }
    binary = bytearray()

    def accessor(data, component_type, count, kind, target, bounds=None):
        view = len(document["bufferViews"])
        index = len(document["accessors"])
        document["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target}
        )
        binary.extend(align(data))
        description = {"bufferView": view, "componentType": component_type, "count": count, "type": kind}
        if bounds:
            description["min"], description["max"] = bounds
        document["accessors"].append(description)
        return index

    model = ElementTree.parse(model_path).getroot()
    for body in model.findall("./Model/BodySet/objects/Body"):
        name = body.get("name")
        if name in MODEL_ONLY:
            continue
        body_node = {"name": name, "children": [], "extras": {"opensimBody": name}}
        document["scenes"][0]["nodes"].append(len(document["nodes"]))
        document["nodes"].append(body_node)
        attachments = []
        for mesh in body.findall("./attached_geometry/Mesh"):
            filename = mesh.findtext("mesh_file")
            if mesh.findtext("socket_frame") != ".." or not filename or Path(filename).name != filename:
                raise ValueError(f"Expected a local mesh attached directly to body {name}")
            scale = tuple(float(value) for value in mesh.findtext("scale_factors", "1 1 1").split())
            if len(scale) != 3 or any(not math.isfinite(value) or value <= 0 for value in scale):
                raise ValueError(f"Expected three positive scales for {filename}")
            if name == "pelvis" and filename == "pelvis.vtp":
                attachments.extend([
                    (filename, scale, "Right hemipelvis", False),
                    (filename, scale, "Left hemipelvis", True),
                    ("sacrum.vtp", scale, "Sacrum", False),
                ])
            else:
                attachments.append((filename, scale, display_mesh_name(filename), False))
        for filename, scale, label, reflected in attachments:
            vertices, normals, triangles = load_mesh(geometry_directory / filename, scale, reflected)
            bounds = ([min(axis) for axis in zip(*vertices)], [max(axis) for axis in zip(*vertices)])
            positions = accessor(
                little_endian("f", (value for point in vertices for value in point)), 5126, len(vertices), "VEC3",
                POSITION_TARGET, bounds,
            )
            normal_indices = accessor(little_endian("f", normals), 5126, len(vertices), "VEC3", POSITION_TARGET)
            wide = len(vertices) > 65535
            indices = accessor(
                little_endian("I" if wide else "H", (index for triangle in triangles for index in triangle)),
                5125 if wide else 5123, 3 * len(triangles), "SCALAR", INDEX_TARGET,
            )
            mesh_index = len(document["meshes"])
            document["meshes"].append({
                "name": label,
                "primitives": [{"attributes": {"POSITION": positions, "NORMAL": normal_indices}, "indices": indices, "material": 0}],
            })
            body_node["children"].append(len(document["nodes"]))
            node = {"name": label, "mesh": mesh_index, "extras": {"sourceMesh": filename}}
            if reflected:
                node["extras"]["reflectedAcrossSagittalPlane"] = True
            document["nodes"].append(node)
        if not body_node["children"]:
            raise ValueError(f"Missing display meshes for {name}")

    if not document["scenes"][0]["nodes"]:
        raise ValueError("No OpenSim body frames found")
    document["buffers"] = [{"byteLength": len(binary)}]
    metadata = align(json.dumps(document, separators=(",", ":"), allow_nan=False).encode(), b" ")
    length = 12 + 8 + len(metadata) + 8 + len(binary)
    output_path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<I4s", len(metadata), b"JSON") + metadata
        + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    )
    print(f"Exported {len(document['scenes'][0]['nodes'])} body frames and {len(document['meshes'])} anatomical meshes to {output_path}")


def fetch(source, target):
    run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "2", source, "--output", target], check=True)


def fetch_and_export(output):
    with TemporaryDirectory(prefix="whole-body-geometry-") as temporary:
        stage = Path(temporary)
        model = stage / "body.osim"
        fetch(
            f"https://raw.githubusercontent.com/gilserrancoli/ReducePoly_forMSK/{MODEL_REVISION}/{quote(MODEL_SOURCE_PATH)}",
            model,
        )
        source = f"https://raw.githubusercontent.com/opensim-org/opensim-models/{SOURCE_REVISION}/Geometry"
        bodies = ElementTree.parse(model).getroot().findall("./Model/BodySet/objects/Body")
        filenames = {
            mesh.text for body in bodies if body.get("name") not in MODEL_ONLY
            for mesh in body.findall("./attached_geometry/Mesh/mesh_file")
        }
        filenames.add("sacrum.vtp")
        if any(not name or Path(name).name != name for name in filenames):
            raise ValueError("Expected an OpenSim geometry filename")
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(lambda name: fetch(f"{source}/{quote(name)}", stage / name), sorted(filenames)))
        export_body(model, stage, output)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch the version-pinned source files from GitHub")
    parser.add_argument("model", type=Path, nargs="?", help="Harba et al. whole-body OpenSim .osim visualization model")
    parser.add_argument("geometry", type=Path, nargs="?", help="OpenSim Models Geometry directory containing ASCII VTP files")
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "assets/body.glb")
    arguments = parser.parse_args()
    if arguments.fetch:
        if arguments.model or arguments.geometry:
            parser.error("--fetch takes no model or geometry arguments")
        fetch_and_export(arguments.output)
    elif arguments.model and arguments.geometry:
        export_body(arguments.model, arguments.geometry, arguments.output)
    else:
        parser.error("provide a model and geometry directory, or use --fetch")
