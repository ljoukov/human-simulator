"""Build the Open3DModel right leg with a landmark-registered knee-flexion clip.

Usage: python tools/build_asset.py lower-limb-glb.zip assets/lower-limb.glb
Requires NumPy, SciPy, DracoPy and libigl. The publisher mesh is in metres.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import struct
from pathlib import Path
from zipfile import ZipFile

import igl
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
# Reuse the project's glTF serialization and source-Draco decoder.
spec = importlib.util.spec_from_file_location(
    "upper_limb_export", HERE.parents[1] / "upper-limb/tools/build_asset.py"
)
glb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(glb)
SOURCE = "https://caskanatomy.info/open3dmodelfiles/lower-limb/lower-limb-glb.zip"
SOURCE_HASH = "5ecb1ef4c07a2f6aa17021ee6f6d0bb0587595821467ec8c6a14f24ee1091173"
PIVOT = np.array([-0.074042, 0.462321, -0.035407])
LATERAL = np.array([-0.115778, 0.462059, -0.035820])
MEDIAL = np.array([-0.032306, 0.462582, -0.034994])
HIP = np.array([-0.083822, 0.860117, -0.005327])
TIP = np.array([-0.0868, 0.4243, 0.0030])
TUBEROSITY = np.array([-0.0839, 0.3846, 0.0016])
ANGLES = np.arange(0, 121, 10)
NERVE_IDS = {
    326,
    327,
    328,
    329,
    330,
    331,
    332,
    333,
    334,
    342,
    348,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    357,
    358,
    359,
    360,
    365,
    366,
    367,
    368,
    369,
    378,
    379,
    380,
    381,
    383,
    384,
    387,
    388,
    390,
}
RETINACULA = {419, 420, 422, 423}
QUADRICEPS = {307, 320, 321, 322}
PES_MUSCLES = {281, 308, 311}
HAMSTRINGS = {281: 296, 287: 264, 308: 296, 310: 309, 311: 296, 312: 264}
COLORS = {
    "plantaris belly and tendon": (1.0, 1.0, 1.0),
    "bone": (0.81, 0.73, 0.56),
    "cartilage": (0.45, 0.69, 0.7),
    "ligament": (0.63, 0.64, 0.46),
    "tendon": (0.85, 0.73, 0.52),
    "muscle": (0.49, 0.105, 0.095),
    "muscle warm": (0.62, 0.157, 0.105),
    "muscle dark": (0.4, 0.075, 0.08),
    "nerve": (0.96, 0.70, 0.17),
    "fascia": (0.71, 0.69, 0.58),
    "sheet": (0.63, 0.72, 0.68),
}


def unit(v):
    return v / np.linalg.norm(v)


def ramp(a, b, x):
    s = np.clip((x - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def basis():
    lateral = unit(LATERAL - MEDIAL)
    superior = HIP - PIVOT
    superior = unit(superior - np.dot(superior, lateral) * lateral)
    return np.column_stack((np.cross(superior, lateral), superior, lateral))


def surface_category(group, node, label):
    if group == "Bones":
        return "bones"
    if group == "Cartilages":
        return "cartilage"
    if group == "Fascia":
        return "fascia"
    if group == "Ligaments":
        return "cartilage" if node == 191 else "ligaments"
    if group == "Muscles":
        if node == 315 or node == 284:
            return None
        return (
            "tendons"
            if "tendon" in label.lower() or "aponeurosis" in label.lower()
            else "muscles"
        )
    if group == "Nerves" and node in NERVE_IDS:
        return "nerves"
    if group == "Overlays" and (node in RETINACULA or node in {421, 424}):
        return "ligaments"
    return None


class Surface:
    def __init__(self, node, name, category, group, points, normals, faces):
        self.node, self.name, self.category, self.group = node, name, category, group
        self.p, _, inv, self.f = igl.remove_duplicate_vertices(
            points.astype(float), faces.astype("i8"), 1e-7
        )
        self.p, self.f = np.ascontiguousarray(self.p), np.ascontiguousarray(self.f)
        self.n = np.zeros_like(self.p)
        np.add.at(self.n, inv, normals)
        lengths = np.linalg.norm(self.n, axis=1, keepdims=True)
        self.n /= np.maximum(lengths, 1e-12)
        bad = lengths[:, 0] < 1e-6
        self.n[bad] = igl.per_vertex_normals(self.p, self.f)[bad]
        self.tree = igl.AABB()
        self.tree.init(self.p, self.f)
        self.face_normals = igl.per_face_normals(self.p, self.f, np.zeros(3))
        a, b, c = self.p[self.f].transpose(1, 0, 2)
        if np.einsum("ij,ij->", a, np.cross(b, c)) < 0:
            self.face_normals *= -1

    def closest(self, points):
        return self.tree.squared_distance(self.p, self.f, np.ascontiguousarray(points))

    def distance(self, points):
        return np.sqrt(self.closest(points)[0])

    def signed(self, points):
        square, face, close = self.closest(points)
        direction = np.einsum("ij,ij->i", points - close, self.face_normals[face])
        return np.sqrt(square) * np.where(direction < 0, -1, 1)


def read_surfaces(path):
    if path.suffix == ".zip":
        with ZipFile(path) as z:
            content = z.read("lower-limb.glb")
    else:
        content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_HASH:
        raise ValueError("Atlas revision differs from the checked footprint table")
    n = struct.unpack_from("<I", content, 12)[0]
    doc, binary = json.loads(content[20 : 20 + n]), content[28 + n :]
    out = {}
    for parent in doc["scenes"][doc["scene"]]["nodes"]:
        group = doc["nodes"][parent]
        for node in group["children"]:
            obj = doc["nodes"][node]
            label = re.sub(
                r"\.r$", "", obj["name"].replace("\u200b", "").strip()
            ).strip()
            if node == 351:
                label = "Lateral femoral cutaneous nerve"
            layer = surface_category(group["name"], node, label)
            if not layer:
                continue
            (prim,) = doc["meshes"][obj["mesh"]]["primitives"]
            out[node] = Surface(
                node,
                label,
                layer,
                group["name"],
                *glb.unpack_surface(doc, binary, prim),
            )
    return out


def transformed(points, pose):
    rotation, translation = pose
    return points @ rotation.T + translation


def inverse_points(points, pose):
    rotation, translation = pose
    return (points - translation) @ rotation


def reference_poses(surfaces):
    frame = basis()
    scale = np.linalg.norm(HIP - PIVOT) / 0.404
    model_knee = np.array([0.0, -0.404, 0.0])
    samples = json.loads((HERE.parent / "data/knee-motion.json").read_text())["samples"]
    poses = {"tibia": [], "patella": []}
    for name, samples_for_body in poses.items():
        ref = np.array(samples[0][name])
        for a in ANGLES:
            mat = np.array(samples[a][name])
            r = frame @ mat[:, :3] @ ref[:, :3].T @ frame.T
            if name == "tibia":
                origin0 = PIVOT + scale * frame @ (ref[:, 3] - model_knee)
            else:
                origin0 = TIP
            moved = origin0 + scale * frame @ (mat[:, 3] - ref[:, 3])
            if name == "tibia":
                # Seat the atlas cartilage/menisci against its own femoral condyles.
                # These measured offsets are computed along the moving tibia from the
                # neutral bidirectional surface clearances of its three articular pairs.
                seat = (
                    np.interp(
                        a,
                        [0, 10, 20, 30, 45, 60, 75, 90, 105, 120],
                        [0, 1.13, 1.54, 1.81, 2.27, 2.84, 3.26, 3.84, 4.75, 5.24],
                    )
                    * 0.001
                )
                moved += seat * (r @ frame[:, 1])
            samples_for_body.append((r, moved - r @ origin0))
    # Set the patella against the trochlea while preserving the measured native
    # apex-to-tuberosity span; the atlas and reference do not share knee shape.
    femur = surfaces[55]
    cartilage = surfaces[145].p[::3]
    baseline = np.percentile(femur.signed(cartilage), [1, 10])
    span = TIP - TUBEROSITY
    lateral = frame[:, 2]
    axial = float(span @ lateral)
    arc_radius = np.linalg.norm(span - axial * lateral)

    def seat_patella(r, t, tibial_foot, previous_angle):
        def candidate(alpha):
            direction = np.cos(alpha) * frame[:, 1] + np.sin(alpha) * frame[:, 0]
            apex = tibial_foot + axial * lateral + arc_radius * direction
            return apex - r @ TIP

        def cost(alpha):
            adjusted = candidate(alpha)
            distances = np.percentile(femur.signed(cartilage @ r.T + adjusted), [1, 10])
            errors = (distances - baseline) * 1000.0
            # Remain on the same gliding branch relative to the preceding pose.
            return (
                np.dot(errors, errors)
                + 0.18 * (alpha - previous_angle) ** 2
                + 0.0003 * np.sum(((adjusted - t) * 1000.0) ** 2)
            )

        result = minimize_scalar(
            cost,
            bounds=(previous_angle - 0.70, previous_angle + 0.70),
            method="bounded",
        )
        return result.x, candidate(result.x)

    previous_angle = 0.0
    for i in range(1, len(ANGLES)):
        r, t = poses["patella"][i]
        tibial_foot = transformed(TUBEROSITY, poses["tibia"][i])
        previous_angle, t = seat_patella(r, t, tibial_foot, previous_angle)
        poses["patella"][i] = r, t
    return poses


def bone_group(s):
    i, y = s.node, s.p[:, 1]
    if i in {74, 145}:
        return "patella"
    if s.category == "bones":
        return "fixed" if i in {48, 55, 60, 63, 64, 65, 66, 67, 80, 85} else "tibia"
    if s.group == "Cartilages" and i != 162:
        return (
            "fixed"
            if i
            in {
                127,
                128,
                129,
                130,
                131,
                132,
                133,
                136,
                137,
                140,
                146,
                147,
                148,
                149,
                161,
            }
            else "tibia"
        )
    if i in {264, 296, 309, 216, 221, 313, 159, 160, 180, 181, 235, 236, 249}:
        return "tibia"
    if i in {265, 260, 191, 172, 173, 278, 279, 280}:
        return "fixed"
    if y.max() < 0.423:
        return "tibia"
    if y.min() > 0.595:
        return "fixed"
    return "dynamic"


def anchor(surfaces, p, other, inner=0.0016, outer=0.013):
    return 1 - ramp(inner, outer, surfaces[other].distance(p))


def weights(s, surfaces):
    p, i = s.p, s.node
    y = p[:, 1]
    pat = np.zeros(len(p))
    # Regional field shared by neighboring material and nerve branches.
    tib = 1 - ramp(0.397, 0.530, y)
    if s.category == "nerves":
        # The sciatic branches course over the posterior knee and share the
        # distal frame through their division at the neck of the fibula.
        tib = 1 - ramp(0.419, 0.550, y)
        if i == 390:
            original = 1 - ramp(0.397, 0.530, y)
            tib = original + (1 - original) * (1 - ramp(0.420, 0.470, y))
    elif i in QUADRICEPS or i == 306 or i in RETINACULA:
        pat = 1 - ramp(0.464, 0.599, y)
        tib = 1 - ramp(0.394, 0.431, y)
        if i == 306:
            on_patella = anchor(surfaces, p, 74, 0.0016, 0.008)
            tib = tib * (1 - on_patella)
            tib += (1 - tib) * anchor(surfaces, p, 86, 0.0016, 0.007)
        if i in QUADRICEPS:
            near_origin = anchor(surfaces, p, 55, 0.0016, 0.013) * ramp(0.505, 0.555, y)
            pat *= 1 - near_origin
        if i in RETINACULA:
            pat *= 1 - anchor(surfaces, p, 55, 0.001, 0.012)
    elif i in HAMSTRINGS:
        tib = 1 - ramp(0.425, 0.630, y)
        tib += (1 - tib) * anchor(surfaces, p, HAMSTRINGS[i], 0.002, 0.016)
        if i == 312:
            tib *= 1 - anchor(surfaces, p, 55, 0.0016, 0.012) * ramp(0.485, 0.55, y)
    elif i in {286, 290, 300}:
        tib = 1 - ramp(0.325, 0.485, y)
        tib *= 1 - anchor(surfaces, p, 55, 0.0018, 0.015)
    elif i in {301, 179, 234, 210, 250, 421}:
        # Intra-articular and collateral structures carry two distinct footprints.
        a = surfaces[55].distance(p)
        distal = 56 if i == 210 else 86
        b = surfaces[distal].distance(p)
        tib = np.maximum(a - 0.0008, 0.0) / np.maximum(
            np.maximum(a - 0.0008, 0.0) + np.maximum(b - 0.0008, 0.0), 1e-9
        )
    elif i in {169, 165, 166, 187, 162, 424}:
        near_shank = anchor(surfaces, p, 86, 0.0016, 0.013)
        tib += (1 - tib) * near_shank
        tib *= 1 - anchor(surfaces, p, 55, 0.0016, 0.012)
        pat = anchor(surfaces, p, 74, 0.0015, 0.018)
    return tib, pat * (1 - tib)


def bend(points, pose, fraction):
    # Interpolate rotation about a common anatomical centre. This preserves the
    # thickness of a cross-section more faithfully than linear matrix blending.
    r, t = pose
    rotvec = Rotation.from_matrix(r).as_rotvec()
    turned = Rotation.from_rotvec(fraction[:, None] * rotvec).apply(points - PIVOT)
    shift = r @ PIVOT + t - PIVOT
    return turned + PIVOT + fraction[:, None] * shift


def deformed(points, tib, pat, pose_t, pose_p):
    if not np.any(pat):
        return bend(points, pose_t, tib)
    rt, tt = pose_t
    rp, tp = pose_p
    turns = (
        tib[:, None] * Rotation.from_matrix(rt).as_rotvec()
        + pat[:, None] * Rotation.from_matrix(rp).as_rotvec()
    )
    return (
        Rotation.from_rotvec(turns).apply(points - PIVOT)
        + PIVOT
        + tib[:, None] * (rt @ PIVOT + tt - PIVOT)
        + pat[:, None] * (rp @ PIVOT + tp - PIVOT)
    )


# Fit near-joint soft tissue on a graph of surface edges and cross-sectional
# chords. Local rotations resist shear; an explicit closed-surface volume
# constraint prevents collision corrections from flattening a muscle.
FITTED = QUADRICEPS | set(HAMSTRINGS) | {286, 290, 300, 301, 306, 169, 261}
BONE_TARGETS = (55, 86, 56, 74)


def volume_and_gradient(p, f):
    a, b, c = p[f].transpose(1, 0, 2)
    grad = np.zeros_like(p)
    for j, term in enumerate([np.cross(b, c), np.cross(c, a), np.cross(a, b)]):
        np.add.at(grad, f[:, j], term / 6)
    return np.einsum("ij,ij->", a, np.cross(b, c)) / 6, grad


class TissueFit:
    def __init__(self, s, surfaces):
        self.s = s
        p, f = s.p, s.f
        edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
        # Neighbour chords keep the two faces of a muscle section together.
        _, neighbors = cKDTree(p).query(p, k=min(15, len(p)))
        chords = np.column_stack(
            (
                np.repeat(np.arange(len(p)), neighbors.shape[1] - 1),
                neighbors[:, 1:].ravel(),
            )
        )
        edges = np.unique(np.sort(np.vstack((edges, chords)), axis=1), axis=0)
        a, b = edges.T
        self.a = np.r_[a, b]
        self.b = np.r_[b, a]
        self.e = p[self.a] - p[self.b]
        self.w = np.clip(
            0.008 / np.maximum(np.linalg.norm(self.e, axis=1), 0.0004), 0.2, 5.0
        )
        self.degree = np.bincount(self.a, weights=self.w, minlength=len(p))
        self.restvol, _ = volume_and_gradient(p, f)
        self.near = {i: surfaces[i].signed(p) for i in BONE_TARGETS}
        self.bones = {i: surfaces[i] for i in BONE_TARGETS}
        # Keep broad tissue patches, including the entire distal myotendinous seam.
        y = p[:, 1]
        if s.node in QUADRICEPS:
            lock = anchor(surfaces, p, 306, 0.0025, 0.010)
            home = 60 if s.node == 307 else 55
            lock = np.maximum(
                lock, anchor(surfaces, p, home, 0.0025, 0.009) * ramp(0.508, 0.55, y)
            )
        elif s.node in HAMSTRINGS:
            home = {287: 265, 311: 265, 310: 60, 281: 60, 308: 60, 312: 55}[s.node]
            lock = np.maximum(
                anchor(surfaces, p, HAMSTRINGS[s.node], 0.0025, 0.012),
                anchor(surfaces, p, home, 0.0018, 0.008) * ramp(0.47, 0.52, y),
            )
        elif s.node in {286, 290, 300}:
            lock = np.maximum(
                anchor(surfaces, p, 55, 0.002, 0.011),
                anchor(surfaces, p, 262, 0.0025, 0.012),
            )
        elif s.node == 301:
            lock = np.maximum(
                anchor(surfaces, p, 55, 0.0018, 0.006),
                anchor(surfaces, p, 86, 0.0018, 0.006),
            )
        elif s.node == 306:
            lock = np.maximum.reduce(
                [anchor(surfaces, p, i, 0.002, 0.008) for i in (74, 86, *QUADRICEPS)]
            )
        elif s.node == 169:
            lock = np.maximum(anchor(surfaces, p, 86, 0.002, 0.01), ramp(0.70, 0.81, y))
        else:
            lock = np.maximum(
                anchor(surfaces, p, 55, 0.0018, 0.007), 1 - ramp(0.46, 0.48, y)
            )
        if s.node in {286, 290, 300}:
            lock = np.maximum(lock, 1 - ramp(0.335, 0.407, y))
        elif s.node in QUADRICEPS or s.node in (HAMSTRINGS.keys() - PES_MUSCLES):
            lock = np.maximum(lock, ramp(0.59, 0.665, y))
        self.guard = []
        if s.node in {286, 290}:
            self.make_guards(surfaces, [86, 56, 313], 1 - ramp(0.412, 0.466, y))
        elif s.node in HAMSTRINGS:
            self.make_guards(surfaces, [55], ramp(0.404, 0.469, y))
        self.line = None
        ends = {
            286: (55, 262),
            290: (55, 262),
            312: (55, 264),
            287: (265, 264),
            310: (60, 309),
            311: (265, 296),
            281: (60, 296),
            308: (60, 296),
            301: (55, 86),
        }
        if s.node in ends:
            upper, lower = ends[s.node]
            du = surfaces[upper].distance(p)
            dl = surfaces[lower].distance(p)
            prox = p[du < 0.0018].mean(0)
            distal = p[dl < 0.0018].mean(0)
            along = np.clip(
                ((p - prox) @ (distal - prox)) / np.sum((distal - prox) ** 2), 0, 1
            )
            hold_upper = np.maximum(
                1 - ramp(0.002, 0.014, du), 1 - ramp(0.0, 0.13, along)
            )
            hold_lower = np.maximum(1 - ramp(0.002, 0.014, dl), ramp(0.84, 0.98, along))
            if s.node in PES_MUSCLES:
                # The three long muscles of pes anserinus span most of the
                # thigh. Their unanchored shaft changes orientation gradually
                # along its own material length between hip and calf.
                hold_upper = np.maximum(hold_upper, 1 - ramp(0.12, 0.85, along))
            elif s.node in HAMSTRINGS:
                hold_upper = np.maximum(hold_upper, ramp(0.605, 0.675, y))
            if s.node in {286, 290}:
                hold_lower = np.maximum(hold_lower, 1 - ramp(0.245, 0.33, y))
            self.line = (prox, distal, hold_upper, hold_lower)
            if s.node in {286, 290}:
                lock = np.maximum(
                    anchor(surfaces, p, 55, 0.002, 0.011),
                    anchor(surfaces, p, 262, 0.0025, 0.012),
                )
        self.lock = lock
        self.guide = self.degree * (0.05 + 18 * lock**2)
        self.contact = self.degree * 0.48
        K = (
            diags(self.degree + self.guide + self.contact)
            - coo_matrix((self.w, (self.a, self.b)), shape=(len(p), len(p))).tocsr()
        )
        self.solve = factorized(K.tocsc())

    def make_guards(self, surfaces, ids, strength):
        # A surface can wrap around a bone even with few of its own vertices
        # inside it. Keep the muscle on the original side of the moving
        # shank/soleus, using the rest triangle and its outward-facing plane.
        for bone_id in ids:
            other = surfaces[bone_id]
            _, faces, near = other.closest(self.s.p)
            restvec = self.s.p - near
            outward = other.face_normals[faces].copy()
            lengths = np.linalg.norm(restvec, axis=1)
            away = lengths > 0.002
            outward[away] = restvec[away] / lengths[away, None]
            gap = np.einsum("ij,ij->i", restvec, outward)
            self.guard.append(
                (bone_id, near, outward, np.clip(gap, -0.001, 0.006), strength)
            )

    def contacts(self, x, pose_t, pose_p):
        push = np.zeros_like(x)
        cutoff = 0.00015
        for i, pose in ((55, None), (86, pose_t), (56, pose_t), (74, pose_p)):
            bone = self.bones[i]
            q = x if pose is None else inverse_points(x, pose)
            in_box = ((q >= bone.p.min(0) - 0.001) & (q <= bone.p.max(0) + 0.001)).all(
                axis=1
            )
            idx = np.flatnonzero(in_box)
            if not len(idx):
                continue
            square, face, near = bone.closest(q[idx])
            diff = q[idx] - near
            n = bone.face_normals[face]
            sign = np.einsum("ij,ij->i", n, diff)
            d = np.where(sign < 0, -1.0, 1.0) * np.sqrt(square)
            # Source footprint overlaps are preserved; limit newly introduced depth.
            target = np.minimum(self.near[i][idx], cutoff)
            bad = d < target
            if bad.any():
                direction = n[bad]
                if pose is not None:
                    direction = direction @ pose[0].T
                push[idx[bad]] += (target[bad] - d[bad])[:, None] * direction
        for i, near, n, minimum, strength in self.guard:
            pose = None if i == 55 else pose_t
            q = x if pose is None else inverse_points(x, pose)
            g = np.einsum("ij,ij->i", q - near, n)
            correction = np.maximum(0.0, minimum - g) * strength
            direction = n if pose is None else n @ pose[0].T
            push += direction * correction[:, None]
        return push

    def fit(self, initial, previous, pose_t, pose_p, steps):
        p, f = self.s.p, self.s.f
        if self.line is not None:
            origin, distal, hold_upper, hold_lower = self.line
            ds = distal - origin
            de = transformed(distal, pose_t) - origin
            axis = np.cross(unit(ds), unit(de))
            angle = np.arctan2(np.linalg.norm(axis), unit(ds) @ unit(de))
            r = Rotation.from_rotvec(unit(axis) * angle).as_matrix()
            ratio = np.linalg.norm(de) / np.linalg.norm(ds)
            off = p - origin
            axial = np.outer(off @ unit(ds), unit(ds))
            guide = origin + (ratio * axial + (off - axial) / np.sqrt(ratio)) @ r.T
            fixed = guide + (p - guide) * hold_upper[:, None]
            guide = fixed + (transformed(p, pose_t) - fixed) * hold_lower[:, None]
            # Retain the measured material footprints rigidly in their parent frame.
            initial = guide + (initial - guide) * self.lock[:, None] ** 3
        x = initial.copy() if previous is None else 0.96 * initial + 0.04 * previous
        x[self.lock > 0.98] = initial[self.lock > 0.98]
        for _ in range(steps):
            dx = x[self.a] - x[self.b]
            products = (
                self.w[:, None, None] * dx[:, :, None] * self.e[:, None, :]
            ).reshape(-1, 9)
            cov = np.stack(
                [
                    np.bincount(self.a, weights=row, minlength=len(p))
                    for row in products.T
                ],
                axis=-1,
            )
            u, _, v = np.linalg.svd(cov.reshape(-1, 3, 3))
            u[:, :, -1] *= np.sign(np.linalg.det(u @ v))[:, None]
            rot = u @ v
            edge = (
                np.einsum("nij,nj->ni", (rot[self.a] + rot[self.b]) * 0.5, self.e)
                * self.w[:, None]
            )
            rhs = np.stack(
                [np.bincount(self.a, weights=col, minlength=len(p)) for col in edge.T],
                axis=-1,
            )
            contact = self.contacts(x, pose_t, pose_p)
            contact *= np.minimum(
                1.0,
                0.012
                / np.maximum(np.linalg.norm(contact, axis=1, keepdims=True), 1e-12),
            )
            projected = x + 2.1 * contact
            rhs += self.guide[:, None] * initial + self.contact[:, None] * projected
            trial = self.solve(rhs)
            vol, grad = volume_and_gradient(trial, f)
            response = self.solve(grad)
            scale = (self.restvol - vol) / max(np.sum(grad * response), 1e-16)
            delta = np.clip(scale, -1e8, 1e8) * response
            trial += delta * np.minimum(
                1.0,
                0.003 / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9),
            )
            x = trial
        # Exact source-matched seam and enthesis constraints.
        return x + (initial - x) * self.lock[:, None] ** 4


class SurfaceGuide:
    """Carry small nerve branches beside their local source muscle/tendon surfaces."""

    def __init__(self, nerve, surfaces, carrier_ids):
        self.entries = []
        dists = []
        for i in carrier_ids:
            carrier = surfaces[i]
            d, ids, near = carrier.closest(nerve.p)
            faces = carrier.f[ids]
            a, b, c = carrier.p[faces].transpose(1, 0, 2)
            bary = igl.barycentric_coordinates(near, a, b, c)
            # Store the offset in an orthonormal frame at the triangle.
            lead = b - a
            lead /= np.maximum(np.linalg.norm(lead, axis=1, keepdims=True), 1e-12)
            normal = carrier.face_normals[ids].copy()
            transverse = np.cross(normal, lead)
            frame = np.stack((lead, transverse, normal), axis=1)
            local = np.einsum("nij,nj->ni", frame, nerve.p - near)
            sign = np.sign(np.einsum("ij,ij->i", normal, np.cross(b - a, c - a)))
            self.entries.append((i, faces, bary, local, sign))
            dists.append(np.sqrt(d))
        distances = np.stack(dists, axis=1)
        # For localized branches, a broad two-surface blend avoids switches
        # between the medial gastrocnemius, semitendinosus and sartorius.
        w = np.exp(-(distances - distances.min(axis=1, keepdims=True)) / 0.0045)
        self.w = w / np.sum(w, axis=1, keepdims=True)
        y = nerve.p[:, 1]
        self.strength = ramp(0.31, 0.38, y) * (1 - ramp(0.535, 0.625, y))

    def pose(self, original, k, all_poses, surfaces):
        result = np.zeros_like(original)
        for j, (i, faces, bary, local, sign) in enumerate(self.entries):
            carrier = all_poses[i][k]
            a, b, c = carrier[faces].transpose(1, 0, 2)
            near = np.einsum("ni,nij->nj", bary, carrier[faces])
            lead = b - a
            normal = np.cross(lead, c - a) * sign[:, None]
            lead /= np.maximum(np.linalg.norm(lead, axis=1, keepdims=True), 1e-12)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            transverse = np.cross(normal, lead)
            frame = np.stack((lead, transverse, normal), axis=1)
            transported = near + np.einsum("nji,nj->ni", frame, local)
            result += transported * self.w[:, j, None]
        return original + (result - original) * self.strength[:, None]


def affine_ligament(s, pose_t, surfaces):
    """Transport each ligament in the changing span between its own footprints."""
    p = s.p
    a = surfaces[55].distance(p)
    b = surfaces[86].distance(p)
    proximal = p[(a < 0.002) & (a < b)].mean(0)
    distal = p[(b < 0.002) & (b < a)].mean(0)
    rest = distal - proximal
    span = transformed(distal, pose_t) - proximal
    cross = np.cross(unit(rest), unit(span))
    angle = np.arctan2(np.linalg.norm(cross), unit(rest) @ unit(span))
    r = Rotation.from_rotvec(unit(cross) * angle).as_matrix()
    off = p - proximal
    axial = np.outer(off @ unit(rest), unit(rest))
    scale = np.linalg.norm(span) / np.linalg.norm(rest)
    return proximal + (scale * axial + off - axial) @ r.T


def settle_interfaces(surfaces, all_poses, fits):
    """Separate overlapping soft tissues in the final pose with both sides mobile.

    Frozen source overlap is the material-coordinate baseline. Embedded muscle
    origins and all paired tendon footprints keep their already fitted motions.
    """
    pairs = [
        (287, 286),
        (312, 286),
        (287, 312),
        (312, 55),
        (310, 308),
        (310, 311),
        (287, 310),
        (311, 308),
        (311, 281),
        (308, 281),
        (290, 313),
        (286, 313),
        (290, 86),
        (286, 56),
        (312, 55),
        (321, 287),
        (322, 310),
        (301, 86),
        (290, 55),
        (286, 55),
        (290, 286),
        (287, 55),
        (310, 55),
        (312, 321),
        (311, 290),
        (310, 290),
    ]
    pairs = list(dict.fromkeys(tuple(sorted(pair)) for pair in pairs))
    sources = {i for pair in pairs for i in pair}
    limits = {}
    from itertools import permutations

    for pair in pairs:
        for a, b in permutations(pair):
            limits[a, b] = np.maximum(0.0, -surfaces[b].signed(surfaces[a].p))
    for k in range(1, len(ANGLES)):
        current = {i: all_poses[i][k] for i in sources}
        for _ in range(12):
            shifts = {i: np.zeros_like(current[i]) for i in sources if i in fits}
            hits = {i: np.zeros(len(current[i])) for i in shifts}
            for left, right in pairs:
                for a, b in ((left, right), (right, left)):
                    v, other = current[a], current[b]
                    mask = ((v >= other.min(0)) & (v <= other.max(0))).all(axis=1)
                    idx = np.flatnonzero(mask)
                    if not len(idx):
                        continue
                    winding = np.abs(
                        igl.fast_winding_number(other, surfaces[b].f, v[idx])
                    )
                    idx = idx[winding > 0.52]
                    if not len(idx):
                        continue
                    ds, face, near = igl.point_mesh_squared_distance(
                        v[idx], other, surfaces[b].f
                    )
                    d = ds**0.5
                    depth = np.maximum(0.0, d - limits[a, b][idx] - 0.00020)
                    move = (
                        np.minimum(0.003, depth)[:, None]
                        * (near - v[idx])
                        / np.maximum(d[:, None], 1e-10)
                    )
                    if a in fits:
                        shifts[a][idx] += move
                        hits[a][idx] += depth > 0
                    if b in fits:
                        contact_faces = surfaces[b].f[face]
                        # The other object's surface can be completely enclosed.
                        # Its material samples then push back on the enclosing triangle.
                        for col in range(3):
                            np.add.at(shifts[b], contact_faces[:, col], -move / 3)
                            np.add.at(hits[b], contact_faces[:, col], (depth > 0) / 3)
            for i, delta in shifts.items():
                fit = fits[i]
                v = current[i]
                mask = (1 - fit.lock) ** 2
                delta /= np.maximum(hits[i], 1)[:, None]
                for _smooth in range(2):
                    adjacent = np.stack(
                        [
                            np.bincount(
                                fit.a, weights=fit.w * col[fit.b], minlength=len(v)
                            )
                            / fit.degree
                            for col in delta.T
                        ],
                        axis=-1,
                    )
                    delta = 0.55 * delta + 0.45 * adjacent
                delta *= mask[:, None]
                v += delta
                volume, grad = volume_and_gradient(v, surfaces[i].f)
                grad *= mask[:, None] * (hits[i] < 0.1)[:, None]
                addition = (fit.restvol - volume) / max(1e-12, np.sum(grad**2)) * grad
                addition *= np.minimum(
                    1.0,
                    0.0006
                    / np.maximum(
                        np.linalg.norm(addition, axis=1, keepdims=True), 1e-10
                    ),
                )
                v += addition


def seat_plantaris(surfaces, all_poses, fits, poses):
    """Route the belly deep to the lateral gastrocnemius and keep the long distal tendon."""
    plantaris = surfaces[300]
    p, y = plantaris.p, plantaris.p[:, 1]
    guide = SurfaceGuide(plantaris, surfaces, (286, 290, 313))
    lateral = ramp(0.355, 0.389, y)
    distal = guide.w[:, 1:].copy()
    distal /= np.maximum(distal.sum(axis=1, keepdims=True), 1e-12)
    guide.w = np.column_stack((lateral, (1 - lateral[:, None]) * distal))
    guide.strength = ramp(0.322, 0.359, y) * (1 - ramp(0.448, 0.481, y))
    # Align each transverse slice with its routed neighbour, retaining a
    # coherent orientation across the small rounded muscle belly.
    kernel = np.exp(-0.5 * ((y[:, None] - y[None, :]) / 0.008) ** 2)
    kernel /= kernel.sum(axis=1, keepdims=True)
    rest_mean = kernel @ p
    for k in range(1, len(ANGLES)):
        raw = all_poses[300][k]
        target = guide.pose(raw, k, all_poses, surfaces)
        moved_mean = kernel @ target
        cross_moments = kernel @ (target[:, :, None] * p[:, None, :]).reshape(-1, 9)
        covariance = (
            cross_moments.reshape(-1, 3, 3)
            - moved_mean[:, :, None] * rest_mean[:, None, :]
        )
        u, _, v = np.linalg.svd(covariance)
        u[:, :, -1] *= np.sign(np.linalg.det(u @ v))[:, None]
        fitted = moved_mean + np.einsum("nij,nj->ni", u @ v, p - rest_mean)
        blend = ramp(0.32, 0.345, y) * (1 - ramp(0.459, 0.48, y))
        fitted = raw + blend[:, None] * (fitted - raw)
        smoothed = fits[300].fit(
            fitted, None, poses["tibia"][k], poses["patella"][k], 12
        )
        all_poses[300][k] = smoothed


def prebake(surfaces, poses):
    all_poses = {}
    driver_weights = {}
    fits = {}
    for i, s in surfaces.items():
        group = bone_group(s)
        if group != "dynamic":
            if group in poses:
                all_poses[i] = [transformed(s.p, p) for p in poses[group]]
            else:
                all_poses[i] = [s.p] * len(ANGLES)
            continue
        tib, pat = weights(s, surfaces)
        driver_weights[i] = (tib, pat)
        fit = TissueFit(s, surfaces) if i in FITTED else None
        if fit:
            fits[i] = fit
        working = [s.p]
        for k in range(1, len(ANGLES)):
            pt, pp = poses["tibia"][k], poses["patella"][k]
            target = deformed(s.p, tib, pat, pt, pp)
            if fit is not None:
                target = fit.fit(target, working[-1] if k > 1 else None, pt, pp, 18)
            if i in {179, 234}:
                target = affine_ligament(s, pt, surfaces)
            working.append(target)
        all_poses[i] = working
    settle_interfaces(surfaces, all_poses, fits)
    seat_plantaris(surfaces, all_poses, fits, poses)
    guide_ids = {332, 333, 342, 351, 353, 360, 365, 366, 367, 369, 379, 380, 387, 388}
    carrier_ids = (
        281,
        286,
        287,
        290,
        296,
        301,
        306,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        320,
        321,
        322,
        264,
    )
    # Superficial cutaneous trunks lie in fascia. The quadriceps sheet and
    # iliotibial tract provide the closest underlying moving surfaces at the knee.
    local_carriers = {353: (169, 166, 286)}
    for i in guide_ids:
        s = surfaces[i]
        if i not in driver_weights:
            continue
        guide = SurfaceGuide(s, surfaces, local_carriers.get(i, carrier_ids))
        for k in range(1, len(ANGLES)):
            all_poses[i][k] = guide.pose(all_poses[i][k], k, all_poses, surfaces)
    # The main fibular trunk glides on the lateral side of the femoral
    # condyle. Carry its nearby lateral sural branch junction with it.
    branch_point = np.array([-0.1027, 0.4895, -0.0479])
    for i in (326, 353):
        p = surfaces[i].p
        local = ramp(0.440, 0.463, p[:, 1]) * (1 - ramp(0.523, 0.548, p[:, 1]))
        if i == 353:
            local *= 1 - ramp(0.008, 0.030, np.linalg.norm(p - branch_point, axis=1))
        correction = 0.008 * local[:, None] * np.array([-1.0, 0.2, 0.15])
        for k in range(1, len(ANGLES)):
            all_poses[i][k] += ramp(12.0, 92.0, ANGLES[k]) * correction
    # A distal branch must keep the same material junction even where the
    # free parts of the two nerves follow different superficial tissues.
    junctions = [(353, 326), (360, 390), (380, 379), (342, 380)]
    for child, parent in junctions:
        s = surfaces[child]
        distance = surfaces[parent].distance(s.p)
        center = s.p[distance < 0.0004].mean(0)
        guide = SurfaceGuide(s, surfaces, (parent,))
        guide.strength = 1 - ramp(0.008, 0.060, np.linalg.norm(s.p - center, axis=1))
        for k in range(1, len(ANGLES)):
            all_poses[child][k] = guide.pose(
                all_poses[child][k], k, all_poses, surfaces
            )
    return all_poses, driver_weights


def initialize_asset():
    asset = glb.Asset()
    asset.json["asset"] = {
        "version": "2.0",
        "generator": "Open3DModel lower limb preparation",
        "copyright": "Open3D project, Jan Kooloos, Eungyeol Lee and collaborators. CC BY-SA 4.0.",
        "extras": {
            "source": SOURCE,
            "license": "https://creativecommons.org/licenses/by-sa/4.0/",
            "changes": "Selected geometry, original materials, welded surfaces and fitted knee motion.",
        },
    }
    asset.json["scenes"][0]["name"] = "Right lower limb"
    asset.json["materials"] = []
    for name, rgb in COLORS.items():
        sheet = name == "sheet"
        asset.json["materials"].append(
            {
                "name": name,
                "doubleSided": name in {"sheet", "fascia"},
                **({"alphaMode": "BLEND"} if sheet else {}),
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*rgb, 0.13 if sheet else 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.69,
                },
            }
        )
    root = asset.node(
        "LowerLimb",
        extras={
            "side": "right",
            "metersPerUnit": 1,
            "anteriorAxis": [0, 0, 1],
            "flexionRangeDegrees": [0, 120],
        },
    )
    nodes = {"fixed": root}
    nodes["tibia"] = asset.node("KneePivot", root, translation=PIVOT.tolist())
    asset.node(
        "FootLoadCenter",
        nodes["tibia"],
        translation=(0.40 * unit(np.array([-0.077, 0.082, -0.04]) - PIVOT)).tolist(),
        extras={"idealLeverMetres": 0.40},
    )
    nodes["patella"] = asset.node("PatellaFrame", root, translation=TIP.tolist())
    anim = {"name": "Knee flexion", "channels": [], "samplers": []}
    asset.json["animations"].append(anim)
    return asset, nodes, anim


def channel(asset, anim, node, path, times, data, kind):
    out = asset.accessor(data.astype("<f4"), 5126, kind)
    anim["channels"].append(
        {"sampler": len(anim["samplers"]), "target": {"node": node, "path": path}}
    )
    anim["samplers"].append({"input": times, "output": out, "interpolation": "LINEAR"})


def palette(s):
    if s.node == 300:
        return "plantaris belly and tendon"
    if s.category == "muscles":
        return ("muscle", "muscle warm", "muscle dark")[s.node % 3]
    if s.category == "fascia":
        return "sheet" if s.node in {165, 166} else "fascia"
    return {
        "bones": "bone",
        "nerves": "nerve",
        "tendons": "tendon",
        "ligaments": "ligament",
        "cartilage": "cartilage",
    }[s.category]


def build(source, output):
    surfaces = read_surfaces(source)
    poses = reference_poses(surfaces)
    all_poses, driver_weights = prebake(surfaces, poses)
    asset, nodes, anim = initialize_asset()
    times = asset.accessor((ANGLES / 120).astype("<f4"), 5126, "SCALAR", bounds=True)
    for name, center in (("tibia", PIVOT), ("patella", TIP)):
        frame = poses[name]
        rotations = Rotation.from_matrix(np.stack([r for r, _ in frame])).as_quat()
        for k in range(1, len(rotations)):
            if np.dot(rotations[k], rotations[k - 1]) < 0:
                rotations[k] *= -1
        channel(asset, anim, nodes[name], "rotation", times, rotations, "VEC4")
        channel(
            asset,
            anim,
            nodes[name],
            "translation",
            times,
            np.array([r @ center + t for r, t in frame]),
            "VEC3",
        )
    morph_output = asset.accessor(
        np.eye(len(ANGLES), dtype="<f4")[:, 1:].reshape(-1), 5126, "SCALAR"
    )
    # Single full-strength influence per vertex supplies an exact local frame for
    # the morph residual. Vertices on the rigid shin need no repeated target data.
    centers = np.stack((np.zeros(3), PIVOT, TIP))
    inv = np.tile(np.eye(4), (3, 1, 1))
    inv[:, :3, 3] = -centers
    asset.json["skins"].append(
        {
            "joints": [nodes[n] for n in ("fixed", "tibia", "patella")],
            "inverseBindMatrices": asset.accessor(
                inv.transpose(0, 2, 1).reshape(3, 16).astype("<f4"), 5126, "MAT4"
            ),
        }
    )
    for i, s in surfaces.items():
        group = bone_group(s)
        center = (
            PIVOT if group == "tibia" else TIP if group == "patella" else np.zeros(3)
        )
        attributes = {
            "POSITION": asset.accessor(
                (s.p - center).astype("<f4"), 5126, "VEC3", 34962, bounds=True
            ),
            "NORMAL": asset.accessor(s.n.astype("<f4"), 5126, "VEC3", 34962),
        }
        if s.node == 300:
            # The source combines muscle and long tendon in one closed surface.
            # Its cylindrical tendon widens into a spindle around Y=0.36 m.
            muscle = ramp(0.344, 0.375, s.p[:, 1])
            rgb = np.outer(1 - muscle, COLORS["tendon"]) + np.outer(
                muscle, COLORS["muscle"]
            )
            attributes["COLOR_0"] = asset.accessor(
                rgb.astype("<f4"), 5126, "VEC3", 34962
            )
        prim = {
            "attributes": attributes,
            "indices": asset.accessor(s.f.ravel().astype("<u2"), 5123, "SCALAR", 34963),
            "material": list(COLORS).index(palette(s)),
        }
        mesh = {"name": s.name, "primitives": [prim]}
        parent = nodes.get(group, nodes["fixed"])
        extra = {}
        if group == "dynamic":
            tib, pat = driver_weights[i]
            joints = np.argmax(np.stack([1 - tib - pat, tib, pat], axis=1), axis=1)
            attrs = np.zeros((len(s.p), 4), dtype=np.uint8)
            attrs[:, 0] = joints
            attributes["JOINTS_0"] = asset.accessor(attrs, 5121, "VEC4", 34962)
            attrs[:, 0] = 255
            attributes["WEIGHTS_0"] = asset.accessor(
                attrs, 5121, "VEC4", 34962, normalized=True
            )
            prim["targets"] = []
            neutral_normals = igl.per_vertex_normals(s.p, s.f)
            for k in range(1, len(ANGLES)):
                pt, pp = poses["tibia"][k], poses["patella"][k]
                posed = all_poses[i][k].copy()
                # Carry the source shading offset through each coherent local rotation.
                normals = igl.per_vertex_normals(posed, s.f)
                offset = s.n - neutral_normals
                normals += Rotation.from_rotvec(
                    tib[:, None] * Rotation.from_matrix(pt[0]).as_rotvec()
                ).apply(
                    Rotation.from_rotvec(
                        pat[:, None] * Rotation.from_matrix(pp[0]).as_rotvec()
                    ).apply(offset)
                )
                normals /= np.maximum(
                    np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
                )
                for joint, pose in ((1, pt), (2, pp)):
                    sel = joints == joint
                    posed[sel] = inverse_points(posed[sel], pose)
                    normals[sel] = normals[sel] @ pose[0]
                prim["targets"].append(
                    {
                        "POSITION": asset.sparse_delta(posed - s.p, bounds=True),
                        "NORMAL": asset.sparse_delta(normals - s.n),
                    }
                )
            mesh["weights"] = [0.0] * (len(ANGLES) - 1)
            extra["skin"] = 0
        index = len(asset.json["meshes"])
        asset.json["meshes"].append(mesh)
        node = asset.node(
            s.name,
            parent,
            mesh=index,
            **extra,
            extras={
                "label": s.name,
                "layer": s.category,
                "sourceNode": i,
                "sourceGroup": s.group,
            },
        )
        if group == "dynamic":
            anim["channels"].append(
                {
                    "sampler": len(anim["samplers"]),
                    "target": {"node": node, "path": "weights"},
                }
            )
            anim["samplers"].append(
                {"input": times, "output": morph_output, "interpolation": "LINEAR"}
            )
    asset.write(output)
    print(
        f"Wrote {len(surfaces)} structures, {output.stat().st_size / 1e6:.1f} MB to {output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
