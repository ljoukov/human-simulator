"""Build the texture-free, elbow-rigged Open3DModel upper limb.

Requires NumPy, SciPy, DracoPy, and libigl. Supply the Open3DModel upper-limb GLB or its archive:
  python tools/build_asset.py path/to/upper-limb-glb.zip assets/upper-limb.glb

Source geometry: Open3DModel – Upper limb, Open3D project (CC BY-SA 4.0).
Build output is CC BY-SA 4.0. Generated materials use plain original colors.
"""

import argparse
import json
import re
import struct
from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile

import DracoPy
import igl
import numpy as np
from scipy.spatial import cKDTree


PIVOT = np.array([-0.224, 1.104, -0.040])
AXIS = np.array([-0.969, 0.197, -0.150])
AXIS /= np.linalg.norm(AXIS)
RANGE_DEGREES = 125
NUM_BEND_JOINTS = 7
LIGAMENT_POSE_STEP = 5
MUSCLE_POSES = np.linspace(0.0, RANGE_DEGREES, 11)
CONTACT_POSES = np.linspace(0.0, RANGE_DEGREES, 26)
SOURCE = "https://anatomytool.org/open3dmodel"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
FASCIA_PATCH = "Proximal antebrachial fascia (cutaway)"
APONEUROSIS = "Bicipital aponeurosis"
BICEPS_TENDON = "Common tendon of biceps brachii"
TRICEPS_TENDON = "Common tendon of triceps brachii"
BICEPS_HEADS = ("Long head of biceps brachii", "Short head of biceps brachii")
FASCIA_SUPPORT = (
    "Palmaris longus muscle", "Flexor carpi radialis", "Humeral head of pronator teres",
)

# Centers of source ligament vertices nearest the humerus and the distal
# structure (annular ligament for radial collateral, ulna for ulnar collateral).
COLLATERAL_ANCHORS = {
    "Radial collateral ligament of elbow": (
        [-0.24668, 1.10813, -0.03952], [-0.24318, 1.08752, -0.03942],
    ),
    "Ulnar collateral ligament of elbow": (
        [-0.19290, 1.09573, -0.03737], [-0.21009, 1.08433, -0.03958],
    ),
}


class RigidBlend(NamedTuple):
    distal_limit: float
    proximal_limit: float
    falloff_power: float
    ease_toward_distal: bool = False
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)


# These medial forearm flexors are pulled back to their humeral origins.
# A blend between the two rigid frames keeps their distal bellies alongside
# the flexed forearm without swinging the intermediate tissue around a wide arc.
FLEXOR_ENVELOPES = {
    "flexor carpi radialis": RigidBlend(1.055, 1.117, 3, True),
    # Separate origin fields keep the ulnar patch mobile and the humeral
    # footprint fixed at the medial epicondyle.
    "flexor digitorum superficialis humero-ulnar head": RigidBlend(0.951, 0.979, 2, True,
                                                                 axis=(0.64, 1.0, 0.0)),
    "humeral head of flexor carpi ulnaris": RigidBlend(0.9841, 1.01452, 3.5,
                                                       axis=(0.45, 1.0, 0.45)),
    "humeral head of pronator teres": RigidBlend(0.99934, 1.14134, 1.4, True,
                                                 axis=(0.0, 1.0, -0.60)),
    "palmaris longus muscle": RigidBlend(1.047, 1.113, 3, True),
}

# Lateral origins stay on the humerus. The extensor bellies and their distal
# tendons follow the radius and ulna. Limit the blend to the proximal forearm.
LATERAL_ENVELOPES = {
    "brachioradialis muscle": RigidBlend(1.033, 1.193, 2, True,
                                         axis=(0.0, 1.0, 0.20)),
    "extensor carpi radialis brevis": RigidBlend(1.025, 1.140, 1.5, True),
    "extensor carpi radialis longus": RigidBlend(1.025, 1.154, 2, True),
    "extensor digiti minimi": RigidBlend(1.035, 1.111, 2, True),
    "extensor digitorum": RigidBlend(1.030, 1.114, 2, True),
    "humeral head of extensor carpi ulnaris": RigidBlend(1.050, 1.114, 1.5, True),
    "supinator": RigidBlend(1.025, 1.112, 3, True),
}

# Upper flexors shorten toward their attached distal points while proximal
# sections stay near the humerus or scapula. Both biceps heads and expansions
# use the same field so their common junction retains its continuity.
BICEPS_ROUTE = (1.068, 1.302, -0.236, -0.019, 0.030)
UPPER_FLEXOR_ROUTES = {
    "brachialis muscle": (1.068, 1.275, -0.218, -0.036, 0.0),
    "long head of biceps brachii": BICEPS_ROUTE,
    "short head of biceps brachii": BICEPS_ROUTE,
    "common tendon of biceps brachii": BICEPS_ROUTE,
    "bicipital aponeurosis": BICEPS_ROUTE,
}

# Correct named high-overlap pairs around the elbow, keeping the source rest
# anatomy and distal tendons. The nearest triangle test is geometric, not a
# constitutive muscle model. Pairs omitted here have no contact constraint.
CONTACT_PAIRS = {
    "Anconeus muscle": ("Humerus", "Articular cartilage of humerus distal end", "Ulna",
                        "Articular cartilage of ulna (proximal end)"),
    "Long head of biceps brachii": ("Brachialis muscle", "Humerus"),
    "Short head of biceps brachii": ("Brachialis muscle", "Humerus", "Long head of biceps brachii"),
    "Humeral head of pronator teres": (
        "Radius", "Articular cartilage of radius head", "Brachialis muscle", "Ulnar head of pronator teres",
    ),
    "Flexor digitorum superficialis humero-ulnar head": (
        "Ulna", "Articular cartilage of ulna (proximal end)", "Humeral head of pronator teres",
    ),
    "Flexor carpi radialis": ("Humerus", "Articular cartilage of humerus distal end",
                             "Flexor digitorum superficialis humero-ulnar head", "Brachialis muscle"),
    "Brachioradialis muscle": ("Humerus", "Articular cartilage of radius head",
                               "Extensor carpi radialis longus", "Extensor carpi radialis brevis",
                               "Anconeus muscle", "Lateral head of triceps brachii", "Brachialis muscle"),
    "Brachialis muscle": ("Humerus", "Articular cartilage of humerus distal end",
                          "Brachioradialis muscle", "Flexor carpi radialis"),
}

# Source meshes retained as individual selectable structures. Counterparts that
# duplicate the same tissue (such as the three separate deltoid heads) are omitted.
GROUPS = {
    "Pectoral girdle - bones": "bones",
    "Pectoral girdle - muscles": "muscles",
    "Arm - bones": "bones",
    "Arm - cartilages": "bones",
    "Arm - muscles": "muscles",
    "Arm - nerves": "nerves",
    "Forearm - bones": "bones",
    "Forearm - cartilages": "bones",
    "Forearm - muscles": "muscles",
    "Forearm - capsules, ligaments, fasciae": "ligaments",
    "Hand and wrist - bones": "bones",
    "Hand and wrist - muscles": "muscles",
    "Hand and wrist - nerves": "nerves",
}
SHOULDER_MUSCLES = (
    "deltoid muscle", "infraspinatus", "supraspinatus", "subscapularis",
    "teres minor", "teres major",
)
ARM_NERVES = (
    "median nerve", "ulnar nerve", "radial nerve", "radial nerve (deep branch)",
    "radial nerve (superficial br)", "radial nerve (posterior interosseus n)",
    "musculocutaneus nerve",
)
CROSSING_FOREARM_MUSCLES = (
    "anconeus", "brachioradialis", "extensor carpi radialis", "extensor digiti minimi",
    "extensor digitorum", "flexor carpi radialis", "flexor digitorum superficialis humero",
    "humeral head", "palmaris longus", "supinator",
)
# Original linear RGB palette for untextured PBR materials. Related shades
# help separate neighboring muscle bodies during selection and orbit.
COLORS = {
    "bone": (0.82, 0.69, 0.51),
    "cartilage": (0.33, 0.58, 0.55),
    "muscle": (0.43, 0.086, 0.065),
    "muscle warm": (0.51, 0.13, 0.092),
    "muscle dark": (0.34, 0.061, 0.053),
    "tendon": (0.73, 0.52, 0.35),
    "fascia": (0.47, 0.32, 0.25),
    "forearm fascia": (0.62, 0.47, 0.36),
    "nerve": (0.82, 0.48, 0.061),
    "ligament": (0.45, 0.58, 0.53),
}


def read_glb(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            data = archive.read("upper-limb.glb")
    else:
        data = path.read_bytes()
    magic, version, size = struct.unpack_from("<4sII", data)
    if (magic, version, size) != (b"glTF", 2, len(data)):
        raise ValueError("Expected an Open3DModel GLB 2 file")
    length, kind = struct.unpack_from("<I4s", data, 12)
    binary_size, binary_kind = struct.unpack_from("<I4s", data, 20 + length)
    if kind != b"JSON" or binary_kind != b"BIN\x00":
        raise ValueError("Expected JSON and BIN chunks")
    return json.loads(data[20:20 + length]), data[28 + length:28 + length + binary_size]


class Asset:
    def __init__(self):
        self.binary = bytearray()
        self.json = {
            "asset": {
                "version": "2.0",
                "generator": "Open3DModel upper limb preparation",
                "copyright": "Open3DModel — Upper limb, Open3D project contributors. CC BY-SA 4.0.",
                "extras": {
                    "source": SOURCE,
                    "license": LICENSE,
                    "changes": "Selected geometry, original materials, and contact-corrected elbow poses.",
                },
            },
            "scene": 0,
            "scenes": [{"name": "Right upper limb", "nodes": [0]}],
            "nodes": [],
            "meshes": [],
            "materials": [
                {
                    "name": name.title(),
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [*color, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.59 if name in ("bone", "cartilage") else 0.74,
                    },
                    **({"alphaMode": "BLEND", "doubleSided": True}
                       if name == "forearm fascia" else {}),
                }
                for name, color in COLORS.items()
            ],
            "accessors": [],
            "bufferViews": [],
            "buffers": [],
            "skins": [],
            "animations": [],
        }

    def node(self, name, parent=None, **properties):
        nodes = self.json["nodes"]
        index = len(nodes)
        nodes.append({"name": name, **properties})
        if parent is not None:
            nodes[parent].setdefault("children", []).append(index)
        return index

    def buffer_view(self, values, target=None):
        while len(self.binary) % 4:
            self.binary.append(0)
        view = {"buffer": 0, "byteOffset": len(self.binary), "byteLength": values.nbytes}
        if target:
            view["target"] = target
        self.binary.extend(values.tobytes())
        self.json["bufferViews"].append(view)
        return len(self.json["bufferViews"]) - 1

    def accessor(self, values, component, kind, target=None, bounds=False, normalized=False):
        accessor = {
            "bufferView": self.buffer_view(values, target),
            "componentType": component,
            "count": len(values),
            "type": kind,
        }
        if bounds:
            accessor["min"] = np.atleast_1d(values.min(axis=0)).tolist()
            accessor["max"] = np.atleast_1d(values.max(axis=0)).tolist()
        if normalized:
            accessor["normalized"] = True
        self.json["accessors"].append(accessor)
        return len(self.json["accessors"]) - 1

    def sparse_delta(self, delta, bounds=False):
        changed = np.flatnonzero(np.max(np.abs(delta), axis=1) > (5e-6 if bounds else 1e-4))
        if len(changed) >= len(delta) * 0.8:
            return self.accessor(delta.astype("<f4"), 5126, "VEC3", 34962, bounds=bounds)
        accessor = {"componentType": 5126, "count": len(delta), "type": "VEC3"}
        if len(changed):
            index_type, component = ("<u2", 5123) if len(delta) <= 65535 else ("<u4", 5125)
            accessor["sparse"] = {
                "count": len(changed),
                "indices": {
                    "bufferView": self.buffer_view(changed.astype(index_type)),
                    "componentType": component,
                },
                "values": {"bufferView": self.buffer_view(delta[changed].astype("<f4"))},
            }
        if bounds:
            accessor["min"] = np.minimum(delta.min(axis=0), 0.0).tolist()
            accessor["max"] = np.maximum(delta.max(axis=0), 0.0).tolist()
        self.json["accessors"].append(accessor)
        return len(self.json["accessors"]) - 1

    def write(self, path):
        self.json["buffers"] = [{"byteLength": len(self.binary)}]
        document = json.dumps(self.json, separators=(",", ":"), ensure_ascii=False).encode()
        document += b" " * (-len(document) % 4)
        self.binary.extend(b"\0" * (-len(self.binary) % 4))
        size = 28 + len(document) + len(self.binary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            struct.pack("<4sIII4s", b"glTF", 2, size, len(document), b"JSON")
            + document
            + struct.pack("<I4s", len(self.binary), b"BIN\0")
            + self.binary
        )


def selected_structures(source):
    nodes = source["nodes"]
    for index in source["scenes"][source["scene"]]["nodes"]:
        group = nodes[index]
        region = group["name"]
        if region not in GROUPS:
            continue
        for child in group["children"]:
            node = nodes[child]
            label = re.sub(r"\.r$", "", node["name"].replace("\u200b", "").strip()).strip()
            name = label.lower()
            if region == "Pectoral girdle - muscles" and not name.startswith(SHOULDER_MUSCLES):
                continue
            if region == "Arm - nerves" and name not in ARM_NERVES:
                continue
            if region == "Hand and wrist - muscles" and not name.startswith((
                "adductor", "abductor", "opponens", "deep head", "superficial head",
                "flexor", "extensor hood", "palmaris",
            )):
                continue
            if name == "long head of biceps brachii tendon sheath":
                continue
            layer = GROUPS[region]
            if name.startswith(("common tendon", "bicipital aponeurosis", "extensor hood")):
                layer = "tendons"
            if name == "antebrachial fascia":
                label = FASCIA_PATCH
                layer = "tendons"
            if name.startswith("art cart"):
                label = label.replace("Art cart of ", "Articular cartilage of ", 1)
            yield child, node, region, layer, label


def material_name(layer, label, ordinal):
    if layer == "bones":
        return "cartilage" if label.startswith("Articular cartilage") else "bone"
    if layer == "muscles":
        return ("muscle", "muscle warm", "muscle dark")[ordinal % 3]
    if label == APONEUROSIS:
        return "fascia"
    if label == FASCIA_PATCH:
        return "forearm fascia"
    return {"nerves": "nerve", "tendons": "tendon", "ligaments": "ligament"}[layer]


def binding(region, layer, label, points):
    """Return body attachment or transitional weight envelope.

    These envelopes are pose deformation, not an elastic or nerve model.
    Smooth angular interpolation preserves edge shape at larger flexion angles.
    """
    name = label.lower()
    if label == FASCIA_PATCH:
        return "surface"
    if region.startswith("Pectoral"):
        return "fixed"
    if region.startswith("Hand and wrist"):
        return "distal"
    if region.startswith("Arm"):
        if layer == "bones" or name.startswith("coracobrachialis"):
            return "fixed"
        if name in UPPER_FLEXOR_ROUTES:
            return "guided"
        if "triceps" in name:
            return RigidBlend(1.110, 1.280, 3)
        if layer == "nerves":
            return (1.052, 1.165)
        return (1.068, 1.191)
    if label in COLLATERAL_ANCHORS:
        return "anchored"
    if name.startswith("articular capsule of elbow"):
        high = float(points[:, 1].max()) - 0.002
        return (high - 0.067, high)
    if name in FLEXOR_ENVELOPES:
        return FLEXOR_ENVELOPES[name]
    if name in LATERAL_ENVELOPES:
        return LATERAL_ENVELOPES[name]
    if name == "anconeus muscle":
        return "anconeus"
    if layer == "muscles" and name.startswith(CROSSING_FOREARM_MUSCLES):
        high = float(points[:, 1].max()) - 0.003
        return (max(1.010, high - 0.091), high)
    return "distal"


def skin_influences(points, envelope):
    if isinstance(envelope, RigidBlend):
        low, high, power, ease_toward_distal, axis = envelope
        relative = np.clip((high - points @ axis) / (high - low), 0.0, 1.0)
        relative = (1.0 - (1.0 - relative) ** power if ease_toward_distal
                    else relative ** power)
        joint0, joint1 = 0, NUM_BEND_JOINTS - 1
        blend = np.rint(255 * relative).astype(np.uint8)
    else:
        low, high = envelope
        relative = np.clip((high - points[:, 1]) / (high - low), 0.0, 1.0)
        relative = relative * relative * (3.0 - 2.0 * relative)
        scaled = relative * (NUM_BEND_JOINTS - 1)
        joint0 = np.floor(scaled).astype(np.uint8)
        joint1 = np.minimum(joint0 + 1, NUM_BEND_JOINTS - 1)
        blend = np.rint(255 * (scaled - joint0)).astype(np.uint8)
    joints = np.zeros((len(points), 4), dtype=np.uint8)
    weights = np.zeros_like(joints)
    joints[:, 0], joints[:, 1] = joint0, joint1
    weights[:, 0], weights[:, 1] = 255 - blend, blend
    return joints, weights


def skin_attributes(asset, points, envelope):
    joints, weights = skin_influences(points, envelope)
    return {
        "JOINTS_0": asset.accessor(joints, 5121, "VEC4", 34962),
        "WEIGHTS_0": asset.accessor(weights, 5121, "VEC4", 34962, normalized=True),
    }


def rotate_vectors(vectors, axis, angle):
    return (vectors * np.cos(angle) + np.cross(axis, vectors) * np.sin(angle)
            + np.outer(vectors @ axis, axis) * (1.0 - np.cos(angle)))


def guided_flexor_pose(points, angle, route):
    low, high, attachment_x, attachment_z, anterior_offset = route
    length = high - low
    distal_anchor = np.array([attachment_x, low, attachment_z]) - PIVOT
    anchor_displacement = rotate_vectors(distal_anchor[None, :], AXIS, angle)[0] - distal_anchor
    up = np.array([0, 1, 0])
    distal_direction = rotate_vectors(up[None, :], AXIS, angle)[0]
    t = np.clip((points[:, 1] - low) / length, 0.0, 1.0)
    distal_fraction = (1.0 - t) ** 2 * (1.0 + 2.0 * t)
    cross_section = points.copy()
    cross_section[:, 0] = attachment_x
    cross_section[:, 2] = attachment_z
    radial_offset = points - cross_section
    center = cross_section + distal_fraction[:, None] * anchor_displacement
    tangent = up - (6.0 * (1.0 - t) * t / length)[:, None] * anchor_displacement
    # The superficial biceps can displace anteriorly as it shortens against
    # the deeper brachialis. The offset vanishes at both anatomical anchors.
    offset = anterior_offset * np.sin(angle / 2.0) ** 2
    center[:, 2] += np.sin(np.pi * t) ** 2 * offset
    tangent[:, 2] += np.sin(2.0 * np.pi * t) * (np.pi / length) * offset
    # Round the junction so the distal section meets the rigid distal frame.
    fillet = np.clip(t / 0.20, 0.0, 1.0)
    fillet = (fillet * fillet * (3.0 - 2.0 * fillet))[:, None]
    tangent = fillet * tangent + (1.0 - fillet) * distal_direction
    stretch = np.linalg.norm(tangent, axis=1, keepdims=True)
    direction = tangent / np.maximum(stretch, 1e-9)
    # Transport the section perpendicular to the local path. Offset its
    # change in axial length by scaling the two transverse directions.
    radial_offset /= np.sqrt(np.maximum(stretch, 0.23))
    turn_axis = np.cross(up, direction)
    skew = np.cross(turn_axis, radial_offset)
    rotated = radial_offset + skew + np.cross(turn_axis, skew) / np.maximum(1.0 + direction[:, 1:2], 1e-5)
    posed = center + rotated
    distal = points[:, 1] <= low
    posed[distal] = PIVOT + rotate_vectors(points[distal] - PIVOT, AXIS, angle)
    return posed


def unit_vectors(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)


def transferred_normals(points, normals, faces, posed):
    source_normals = igl.per_vertex_normals(points, faces)
    return unit_vectors(normals + igl.per_vertex_normals(posed, faces) - source_normals)


def shape_morph(asset, points, normals, faces, posed):
    """Preserve the source normal field while changing mesh geometry."""
    changed_normals = transferred_normals(points, normals, faces, posed)
    return {
        "POSITION": asset.sparse_delta(posed - points, bounds=True),
        "NORMAL": asset.sparse_delta(changed_normals - normals),
    }


def unpack_surface(source, data, primitive):
    compressed = primitive["extensions"]["KHR_draco_mesh_compression"]
    view = source["bufferViews"][compressed["bufferView"]]
    offset = view.get("byteOffset", 0)
    mesh = DracoPy.decode(data[offset:offset + view["byteLength"]])
    decoded = compressed["attributes"]
    points = mesh.get_attribute_by_unique_id(decoded["POSITION"])["data"]
    normals = mesh.get_attribute_by_unique_id(decoded["NORMAL"])["data"]
    return points, normals, mesh.faces


class ContactSurface:
    def __init__(self, points, faces, label, mode):
        self.points, self.faces = points.astype(float), faces.astype("i8")
        self.label, self.mode = label, mode
        _, _, inverse = igl.remove_duplicate_vertices(self.points, 1.0e-7)
        self.weld_groups = inverse.reshape(-1)
        self.weld_sizes = np.bincount(self.weld_groups)
        if isinstance(mode, tuple):
            self.joints, self.weights = skin_influences(self.points, mode)
            self.weights = self.weights / 255.0

    def weld_displacement(self, displacement):
        # glTF duplicates UV and normal seams. Their rendered attributes may
        # differ, but each copy must receive the same geometric displacement.
        mean = np.zeros((len(self.weld_sizes), 3))
        np.add.at(mean, self.weld_groups, displacement)
        return (mean / self.weld_sizes[:, None])[self.weld_groups]

    def attach_anconeus(self, humerus, ulna):
        """Fix the epicondylar origin and move the broad ulnar insertion ribbon."""
        _, first, inverse, welded_faces = igl.remove_duplicate_vertices(self.points, self.faces, 1e-7)
        welded_points = self.points[first]
        humeral_dist = np.sqrt(igl.point_mesh_squared_distance(
            welded_points, humerus.points, humerus.faces,
        )[0])
        ulnar_dist = np.sqrt(igl.point_mesh_squared_distance(
            welded_points, ulna.points, ulna.faces,
        )[0])
        humerus_origin = (humeral_dist < 0.0015) & (humeral_dist < ulnar_dist)
        ulna_insertion = (ulnar_dist < 0.0015) & (ulnar_dist < humeral_dist)
        pinned = humerus_origin | ulna_insertion
        welded_weights = igl.harmonic(welded_points, welded_faces, np.flatnonzero(pinned),
                                      ulna_insertion[pinned, None].astype(float), 1)[:, 0]
        self.humeral_origin = humerus_origin[inverse]
        self.ulnar_insertion = ulna_insertion[inverse]
        self.ulnar_weight = np.clip(welded_weights[inverse], 0.0, 1.0)

    def pose(self, angle, rotations):
        if self.mode == "guided":
            return guided_flexor_pose(self.points, angle, UPPER_FLEXOR_ROUTES[self.label.lower()]), None
        if self.mode == "distal":
            return (self.points - PIVOT) @ rotations[-1].T + PIVOT, None
        if self.mode == "anconeus":
            distal = (self.points - PIVOT) @ rotations[-1].T + PIVOT
            # The free fan curves around the lateral epicondyle as it flexes.
            weight = self.ulnar_weight[:, None]
            wrap = np.sin(np.pi * weight) * np.sin(angle / 2.0) ** 2
            return (self.points + weight * (distal - self.points)
                    + wrap * np.array([-0.020, 0.0, 0.006])), None
        if isinstance(self.mode, tuple):
            matrices = np.einsum("ni,nijk->njk", self.weights, rotations[self.joints])
            return np.einsum("nij,nj->ni", matrices, self.points - PIVOT) + PIVOT, matrices
        return self.points.copy(), None


def smooth_ramp(fraction):
    t = np.clip(fraction, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class SurfaceAttachment:
    """Freeze the correspondence to a particular source triangle and point."""
    def __init__(self, points, receiver):
        squared, self.face_ids, self.anchors = igl.point_mesh_squared_distance(
            points, receiver.points, receiver.faces,
        )
        self.distance = np.sqrt(squared)
        self.indices = receiver.faces[self.face_ids]
        triangles = receiver.points[self.indices]
        self.weights = np.nan_to_num(igl.barycentric_coordinates(
            self.anchors, triangles[:, 0], triangles[:, 1], triangles[:, 2],
        ))
        # Transfer the small separation of neighboring mesh sheets in a
        # triangle-local frame instead of treating the offset as world-fixed.
        frame = self.triangle_frames(triangles)
        self.offset = np.einsum("nki,ni->nk", frame, points - self.anchors)
        self.normals = frame[:, 2]

    @staticmethod
    def triangle_frames(triangles):
        first = unit_vectors(triangles[:, 1] - triangles[:, 0])
        normal = unit_vectors(np.cross(first, triangles[:, 2] - triangles[:, 0]))
        return np.stack((first, np.cross(normal, first), normal), axis=1)

    def follow(self, points, offset_scale=1.0):
        triangles = points[self.indices]
        anchor = np.einsum("ni,nij->nj", self.weights, triangles)
        offset = np.einsum("nki,nk->ni", self.triangle_frames(triangles), self.offset)
        return anchor + offset * offset_scale


class ElbowAttachments:
    """Couple the atlas's separate tendon, muscle and fascia meshes."""
    def __init__(self, surfaces):
        fascia = surfaces[FASCIA_PATCH]
        aponeurosis = surfaces[APONEUROSIS]
        tendon = surfaces[BICEPS_TENDON]
        self.surfaces = surfaces
        self.updated = (FASCIA_PATCH, APONEUROSIS, BICEPS_TENDON, TRICEPS_TENDON)
        self.apo_to_tendon = SurfaceAttachment(aponeurosis.points, tendon)
        self.apo_to_fascia = SurfaceAttachment(aponeurosis.points, fascia)
        self.fascia_receivers = {
            name: SurfaceAttachment(fascia.points, surfaces[name])
            for name in FASCIA_SUPPORT
        }
        distances = np.array([receiver.distance for receiver in self.fascia_receivers.values()])
        weights = np.exp(-(distances - distances.min(axis=0)) / 0.0025)
        self.fascia_weights = weights / weights.sum(axis=0)

        distances = np.array([
            SurfaceAttachment(aponeurosis.points, surfaces[name]).distance
            for name in FASCIA_SUPPORT
        ])
        flexor_distance = distances.min(axis=0)
        above_tendon = self.apo_to_tendon.distance
        self.fan_to_fascia = smooth_ramp((1.101 - aponeurosis.points[:, 1]) / 0.013)
        self.fan_to_fascia *= smooth_ramp((above_tendon - 0.003) / 0.007)
        self.fan_to_fascia *= smooth_ramp((0.010 - flexor_distance) / 0.005)
        self.fan_to_tendon = smooth_ramp((0.010 - above_tendon) / 0.008)
        fan_normals = igl.per_vertex_normals(aponeurosis.points, aponeurosis.faces)
        rim = ((self.fan_to_fascia > 0.8) & (aponeurosis.points[:, 1] < 1.090)
               & (np.abs(np.sum(fan_normals * self.apo_to_fascia.normals, axis=1)) < 0.55))
        rim_distance = cKDTree(aponeurosis.points[rim]).query(aponeurosis.points)[0]
        # Blend the serrated edge into the fascia and preserve the intervening
        # sheet above the forearm muscles.
        self.fan_offset = (1.0 - 0.90 * smooth_ramp((0.007 - rim_distance) / 0.007))[:, None]

        targets = self.apo_to_fascia.anchors[self.fan_to_fascia > 0.12]
        self.fascia_extent = cKDTree(targets)
        close_faces = self.fascia_extent.query(fascia.points[fascia.faces].mean(axis=1))[0] < 0.009
        close_faces[self.apo_to_fascia.face_ids[self.fan_to_fascia > 0.12]] = True
        self.fascia_indices, inverse = np.unique(fascia.faces[close_faces], return_inverse=True)
        self.fascia_faces = inverse.reshape(-1, 3)
        self.fascia_colors = np.ones((len(self.fascia_indices), 4), dtype="<f4")
        patch_distance = self.fascia_extent.query(fascia.points[self.fascia_indices])[0]
        self.fascia_colors[:, 3] = 0.52 * smooth_ramp((0.013 - patch_distance) / 0.008)

        self.tendon_to_head = {
            name: SurfaceAttachment(tendon.points, surfaces[name]) for name in BICEPS_HEADS
        }
        distances = np.array([anchor.distance for anchor in self.tendon_to_head.values()])
        self.tendon_head_index = distances.argmin(axis=0)
        self.tendon_head_strength = smooth_ramp((0.010 - distances.min(axis=0)) / 0.0085)
        self.bone_strength = {}
        for name, bone in ((BICEPS_TENDON, "Radius"), (TRICEPS_TENDON, "Ulna")):
            attachment = SurfaceAttachment(surfaces[name].points, surfaces[bone])
            self.bone_strength[name] = smooth_ramp((0.0045 - attachment.distance) / 0.003)

    def update(self, deformed, distal_rotation):
        tendon = self.surfaces[BICEPS_TENDON]
        candidates = np.array([
            attachment.follow(deformed[name]) for name, attachment in self.tendon_to_head.items()
        ])
        nearest_belly = candidates[self.tendon_head_index, np.arange(len(tendon.points))]
        deformed[BICEPS_TENDON] += (
            nearest_belly - deformed[BICEPS_TENDON]
        ) * self.tendon_head_strength[:, None]
        for name, strength in self.bone_strength.items():
            pinned = (self.surfaces[name].points - PIVOT) @ distal_rotation.T + PIVOT
            deformed[name] += (pinned - deformed[name]) * strength[:, None]

        follow_fascia = np.array([
            anchor.follow(deformed[name]) for name, anchor in self.fascia_receivers.items()
        ])
        deformed[FASCIA_PATCH] = np.einsum("ij,ijk->jk", self.fascia_weights, follow_fascia)
        fan = deformed[APONEUROSIS]
        fan += (self.apo_to_tendon.follow(deformed[BICEPS_TENDON]) - fan) * self.fan_to_tendon[:, None]
        woven_edge = self.apo_to_fascia.follow(deformed[FASCIA_PATCH], self.fan_offset)
        fan += (woven_edge - fan) * self.fan_to_fascia[:, None]


class ContactProjection:
    def __init__(self, surface):
        points, faces = surface.points, surface.faces
        self.movable = points[:, 1] > 1.021
        if isinstance(surface.mode, tuple):
            self.movable &= (surface.joints * surface.weights).sum(axis=1) > 0
        if surface.mode == "guided":
            route = UPPER_FLEXOR_ROUTES[surface.label.lower()]
            self.movable &= (points[:, 1] > route[0]) & (points[:, 1] < route[1])
        if surface.mode == "anconeus":
            self.movable &= ~(surface.humeral_origin | surface.ulnar_insertion)
        # A seam is pinned if any of its coincident copies is a footprint.
        movable_per_group = np.bincount(surface.weld_groups, self.movable,
                                        minlength=len(surface.weld_sizes))
        self.movable = (movable_per_group == surface.weld_sizes)[surface.weld_groups]
        vertices = np.flatnonzero(self.movable)
        triangles = faces[self.movable[faces].any(axis=1)]
        self.samples = np.vstack((np.tile(vertices[:, None], (1, 3)), np.tile(triangles, (4, 1))))
        self.factors = np.vstack((
            np.tile([1, 0, 0], (len(vertices), 1)),
            np.repeat([[1/3, 1/3, 1/3], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], len(triangles), axis=0),
        ))
        self.push_factors = self.factors * self.movable[self.samples]
        divisor = np.maximum(np.sum(self.factors * self.push_factors, axis=1), 0.25)
        self.push_factors /= divisor[:, None]
        self.neighbors = igl.adjacency_matrix(faces)
        self.neighbors.resize((len(points), len(points)))
        self.degree = np.maximum(np.asarray(self.neighbors.sum(axis=1)), 1)

    def locations(self, positions):
        return np.einsum("ij,ijk->ik", self.factors, positions[self.samples])

    def push(self, current, obstacles, reference):
        samples = self.locations(current)
        displacement = np.zeros_like(samples)
        largest = np.zeros(len(samples))
        for neighbor, obstacle_positions in obstacles:
            distance, _, near, _ = igl.signed_distance(
                samples, obstacle_positions, neighbor.faces,
                igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            )
            deficit = reference[neighbor.label] - distance
            active = np.isfinite(deficit) & (deficit > largest)
            offset = samples[active] - near[active]
            source_to_surface = offset * np.sign(distance[active, None])
            offset_norm = np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-9)
            displacement[active] = source_to_surface / offset_norm * np.minimum(deficit[active, None], 0.003)
            largest[active] = deficit[active]
        active = largest > 0.0
        indices = self.samples[active].reshape(-1)
        push_weights = self.push_factors[active]
        change = np.zeros_like(current)
        counts = np.zeros(len(current))
        np.add.at(change, indices, (displacement[active, None, :] * push_weights[:, :, None]).reshape(-1, 3))
        np.add.at(counts, indices, (push_weights > 0).reshape(-1))
        change /= np.maximum(counts[:, None], 1.0)
        change = 0.85 * change + 0.15 * (self.neighbors @ change) / self.degree
        change[~self.movable] = 0.0
        return change


def separate_lateral_contact(deformed, surfaces, degrees):
    """Constrain the facing brachialis and brachioradialis flanks in deep bend."""
    blend = np.clip((degrees - 100.0) / 22.5, 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    plane_origin = np.array([-0.228, 1.181, 0.0])
    normal = np.array([-1.0, 0.0, 0.60])
    normal /= np.linalg.norm(normal)
    for name, side, near, far in (
        ("Brachialis muscle", -1.0, 1.074, 1.250),
        ("Brachioradialis muscle", 1.0, 0.975, 1.130),
    ):
        surface = surfaces[name]
        start = surface.points[:, 1]
        posed = deformed[name]
        support = np.clip((start - near) / 0.030, 0.0, 1.0) * np.clip((far - start) / 0.030, 0.0, 1.0)
        world_support = np.clip((posed[:, 1] - 1.140) / 0.025, 0.0, 1.0)
        world_support *= np.clip((1.235 - posed[:, 1]) / 0.025, 0.0, 1.0)
        depth = (posed - plane_origin) @ normal * side
        push = 0.0002 + np.logaddexp(0, depth / 0.0018) * 0.0018 - depth
        step = normal * (side * blend * np.minimum(support, world_support) * push)[:, None]
        deformed[name] += surface.weld_displacement(step)
        if side > 0:
            # Redistribute the compressed brachioradialis belly into available
            # lateral space without moving it back across the separating plane.
            flexed = deformed[name]
            material_support = np.clip((start - 0.935) / 0.055, 0.0, 1.0)
            material_support *= np.clip((1.185 - start) / 0.070, 0.0, 1.0)
            pose_support = np.clip((flexed[:, 1] - 1.102) / 0.045, 0.0, 1.0)
            pose_support *= np.clip((1.304 - flexed[:, 1]) / 0.045, 0.0, 1.0)
            room = np.maximum((flexed - plane_origin) @ normal, 0.0)
            expansion = normal * (room * material_support * pose_support * blend * 1.05)[:, None]
            deformed[name] += surface.weld_displacement(expansion)


def bake_contacts(source, binary, selected):
    needed = set(CONTACT_PAIRS)
    needed.update(other for neighbors in CONTACT_PAIRS.values() for other in neighbors)
    needed.update(FASCIA_SUPPORT + BICEPS_HEADS)
    needed.update((FASCIA_PATCH, APONEUROSIS, BICEPS_TENDON, TRICEPS_TENDON))
    surfaces = {}
    for _, node, region, layer, label in selected:
        if label not in needed:
            continue
        primitive, = source["meshes"][node["mesh"]]["primitives"]
        points, _, faces = unpack_surface(source, binary, primitive)
        surfaces[label] = ContactSurface(points, faces, label, binding(region, layer, label, points))

    surfaces["Anconeus muscle"].attach_anconeus(surfaces["Humerus"], surfaces["Ulna"])
    attachments = ElbowAttachments(surfaces)
    projections = {name: ContactProjection(surfaces[name]) for name in CONTACT_PAIRS}
    reference = {}
    for name, neighbors in CONTACT_PAIRS.items():
        projection = projections[name]
        samples = projection.locations(surfaces[name].points)
        reference[name] = {}
        for other in neighbors:
            nearby = surfaces[other]
            distance, _, _, _ = igl.signed_distance(
                samples, nearby.points, nearby.faces, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER,
            )
            # Existing source intersections and on-bone attachments are not
            # corrected by moving their rest footprints.
            reference[name][other] = np.minimum(0.00015, distance) - 0.0001

    baked = {name: [] for name in (*CONTACT_PAIRS, *attachments.updated)}
    neutral = {name: surface.points.copy() for name, surface in surfaces.items()}
    attachments.update(neutral, np.eye(3))
    for degrees in CONTACT_POSES[1:]:
        angle = np.radians(degrees)
        rotations = np.array([
            rotate_vectors(np.eye(3), AXIS, turn).T
            for turn in np.linspace(0.0, angle, NUM_BEND_JOINTS)
        ])
        poses = {name: surface.pose(angle, rotations) for name, surface in surfaces.items()}
        deformed = {name: posed.copy() for name, (posed, _) in poses.items()}
        # Preserve neutral attachment locations when solving the posed contacts.
        max_displacement = min(degrees / 25.0, 1.0) * 0.009
        for _ in range(6):
            for name, neighbors in CONTACT_PAIRS.items():
                projection = projections[name]
                blockers = [(surfaces[other], deformed[other]) for other in neighbors]
                current = deformed[name]
                step = projection.push(current, blockers, reference[name])
                delta = surfaces[name].weld_displacement(current + step - poses[name][0])
                norm = np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1.0e-9)
                deformed[name] = poses[name][0] + delta * np.minimum(max_displacement / norm, 1.0)
        separate_lateral_contact(deformed, surfaces, degrees)
        attachments.update(deformed, rotations[-1])
        for name in baked:
            posed, matrices = poses[name]
            baked[name].append((posed, matrices, deformed[name]))
    return baked, neutral[APONEUROSIS], attachments


def contact_morph(asset, points, normals, faces, baked):
    posed, matrices, deformed = baked
    if matrices is None:
        return shape_morph(asset, points, normals, faces, deformed)
    inverse = np.linalg.inv(matrices)
    normal_change = igl.per_vertex_normals(deformed, faces) - igl.per_vertex_normals(posed, faces)
    return {
        "POSITION": asset.sparse_delta(np.einsum("nij,nj->ni", inverse, deformed - posed), bounds=True),
        "NORMAL": asset.sparse_delta(np.einsum("nij,nj->ni", inverse, normal_change)),
    }


def collateral_morphs(asset, position, normals, anchors):
    """Move each endpoint with its anatomy and retain a positive span scale."""
    proximal, distal = np.array(anchors)
    rest_span = distal - proximal
    rest_length = np.linalg.norm(rest_span)
    rest_direction = rest_span / rest_length
    offsets = position - proximal
    along = np.outer(offsets @ rest_direction, rest_direction)
    normals_along_span = np.outer(normals @ rest_direction, rest_direction)
    targets = []
    for degrees in range(LIGAMENT_POSE_STEP, RANGE_DEGREES + 1, LIGAMENT_POSE_STEP):
        posed_distal = PIVOT + rotate_vectors(
            (distal - PIVOT)[None, :], AXIS, np.radians(degrees),
        )[0]
        span = posed_distal - proximal
        length = np.linalg.norm(span)
        ratio = length / rest_length
        direction = span / length
        turn = np.cross(rest_direction, direction)
        sine = np.linalg.norm(turn)
        angle = np.arctan2(sine, rest_direction @ direction)
        turn /= sine
        deformed = proximal + rotate_vectors(offsets + along * (ratio - 1.0), turn, angle)
        posed_normals = rotate_vectors(normals + normals_along_span * (1.0 / ratio - 1.0),
                                      turn, angle)
        posed_normals /= np.linalg.norm(posed_normals, axis=1, keepdims=True)
        targets.append({
            "POSITION": asset.accessor((deformed - position).astype("<f4"), 5126, "VEC3",
                                       34962, bounds=True),
            "NORMAL": asset.accessor((posed_normals - normals).astype("<f4"), 5126,
                                     "VEC3", 34962),
        })
    return targets


def animate_morphs(asset, node, count):
    animation = asset.json["animations"][0]
    times = asset.accessor(np.linspace(0.0, 1.0, count + 1, dtype="<f4"), 5126, "SCALAR",
                           bounds=True)
    weights = np.eye(count + 1, dtype="<f4")[:, 1:].reshape(-1)
    output = asset.accessor(weights, 5126, "SCALAR")
    animation["channels"].append({
        "sampler": len(animation["samplers"]),
        "target": {"node": node, "path": "weights"},
    })
    animation["samplers"].append({"input": times, "output": output, "interpolation": "LINEAR"})


def make_rig(asset):
    root = asset.node("UpperLimb", extras={
        "side": "right", "metersPerUnit": 1, "anteriorAxis": [0, 0, 1],
        "flexionRangeDegrees": [0, RANGE_DEGREES],
    })
    joints = []
    for index in range(NUM_BEND_JOINTS):
        if index == 0:
            name = "HumerusFrame"
        elif index == NUM_BEND_JOINTS - 1:
            name = "ElbowPivot"
        else:
            name = f"ElbowBend{index}"
        joints.append(asset.node(name, root, translation=PIVOT.tolist(),
                                 extras={"axis": AXIS.tolist()}))
    pivot = joints[-1]
    # A 30 cm reference lever for the external load. This is a model parameter.
    lever = np.array([-0.025, -0.275, 0.061])
    lever *= 0.30 / np.linalg.norm(lever)
    asset.node("HandLoadCenter", pivot, translation=lever.tolist())
    offset = np.eye(4, dtype="<f4")
    offset[:3, 3] = -PIVOT
    bind = np.repeat(offset.T.reshape(1, 16), len(joints), axis=0)
    asset.json["skins"].append({
        "name": "Geometry-derived elbow deformation",
        "inverseBindMatrices": asset.accessor(bind, 5126, "MAT4"),
        "skeleton": root,
        "joints": joints,
    })
    times = asset.accessor(np.array([0.0, 1.0], dtype="<f4"), 5126, "SCALAR", bounds=True)
    animation = {"name": "Elbow flexion", "samplers": [], "channels": [],
                 "extras": {"degrees": [0, RANGE_DEGREES]}}
    for index, joint in enumerate(joints[1:], 1):
        half_angle = np.radians(RANGE_DEGREES * index / (len(joints) - 1)) / 2
        poses = np.array([[0, 0, 0, 1], [*(AXIS * np.sin(half_angle)), np.cos(half_angle)]],
                         dtype="<f4")
        rotations = asset.accessor(poses, 5126, "VEC4")
        animation["channels"].append({
            "sampler": len(animation["samplers"]),
            "target": {"node": joint, "path": "rotation"},
        })
        animation["samplers"].append({
            "input": times, "output": rotations, "interpolation": "LINEAR",
        })
    asset.json["animations"].append(animation)
    return root, pivot


def build(input_path, output_path):
    source, data = read_glb(input_path)
    selected = list(selected_structures(source))
    contact_bakes, neutral_fan, attachments = bake_contacts(source, data, selected)
    asset = Asset()
    root, pivot = make_rig(asset)
    materials = {name: index for index, name in enumerate(COLORS)}

    for ordinal, node, region, layer, label in selected:
        source_mesh = source["meshes"][node["mesh"]]
        mesh_index = len(asset.json["meshes"])
        primitives = []
        metadata = {}
        morph_count = 0
        parent = root
        for primitive in source_mesh["primitives"]:
            points, normals, faces = unpack_surface(source, data, primitive)
            if label == APONEUROSIS:
                normals = transferred_normals(points.astype(float), normals, faces.astype("i8"), neutral_fan)
                points = neutral_fan
            if label == FASCIA_PATCH:
                points = points[attachments.fascia_indices]
                normals = normals[attachments.fascia_indices]
                faces = attachments.fascia_faces
            points, normals = points.astype("<f4"), normals.astype("<f4")
            mode = binding(region, layer, label, points)
            attributes = {
                "POSITION": asset.accessor(points, 5126, "VEC3", 34962, bounds=True),
                "NORMAL": asset.accessor(normals, 5126, "VEC3", 34962),
            }
            if label == FASCIA_PATCH:
                attributes["COLOR_0"] = asset.accessor(attachments.fascia_colors, 5126, "VEC4", 34962)
            if mode == "distal":
                metadata = {"translation": (-PIVOT).tolist()}
                parent = pivot
            elif isinstance(mode, tuple):
                metadata = {"skin": 0}
                attributes.update(skin_attributes(asset, points, mode))
            index_type, component = ("<u2", 5123) if len(points) <= 65535 else ("<u4", 5125)
            indices = asset.accessor(faces.reshape(-1).astype(index_type), component,
                                     "SCALAR", 34963)
            converted = {
                "attributes": attributes,
                "indices": indices,
                "material": materials[material_name(layer, label, ordinal)],
            }
            if mode == "anchored":
                converted["targets"] = collateral_morphs(asset, points, normals,
                                                         COLLATERAL_ANCHORS[label])
                morph_count = len(converted["targets"])
            if label in contact_bakes:
                poses = contact_bakes[label]
                if label == FASCIA_PATCH:
                    poses = [
                        (posed[attachments.fascia_indices], None, deformed[attachments.fascia_indices])
                        for posed, _, deformed in poses
                    ]
                converted["targets"] = [
                    contact_morph(asset, points.astype(float), normals, faces.astype("i8"), result)
                    for result in poses
                ]
                morph_count = len(converted["targets"])
            elif mode == "guided":
                route = UPPER_FLEXOR_ROUTES[label.lower()]
                muscle_points = points.astype(np.float64)
                muscle_faces = faces.astype(np.int64)
                converted["targets"] = [
                    shape_morph(asset, muscle_points, normals, muscle_faces,
                                guided_flexor_pose(muscle_points, np.radians(deg), route))
                    for deg in MUSCLE_POSES[1:]
                ]
                morph_count = len(converted["targets"])
            primitives.append(converted)
        mesh_properties = {"name": label, "primitives": primitives}
        if morph_count:
            mesh_properties["weights"] = [0.0] * morph_count
        asset.json["meshes"].append(mesh_properties)
        output_node = asset.node(label, parent, mesh=mesh_index, **metadata, extras={
            "label": label, "layer": layer, "region": region.split(" - ", 1)[0],
        })
        if morph_count:
            animate_morphs(asset, output_node, morph_count)

    asset.write(output_path)
    print(f"Wrote {len(asset.json['meshes'])} named structures to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)
