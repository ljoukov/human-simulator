"""Build and validate the native / WebAssembly iliotibial tract contact model."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
from scipy.spatial.transform import Rotation

from prepare_hip_deformation import ATLAS, DATA, ROOT, UNIT, deformation, load_meshes

BAND = "Iliotibial tract"
OBSTACLES = [
    "Tensor fasciae latae",
    "Gluteus medius muscle",
    "Gluteus minimus muscle",
    "Gluteus maximus muscle",
    "Vastus lateralis muscle",
    "Femur",
    "Hip bone",
    "Trochanteric bursa of gluteus maximus",
]
C_SOURCE = ROOT / "hip-study/contact/contact.c"
WASM = ROOT / "upper-limb/hip-study/assets/hip-contact.wasm"
MODEL = DATA / "hip-contact.bin"


def read_weights(name, keys, manifest, binary):
    rec = manifest["surfaces"][name]
    if "constant" in rec:
        return np.full(len(keys), rec["constant"])
    stride = rec.get("recordBytes", manifest["recordBytes"])
    field = np.dtype(
        dict(
            names=["coords", "value"],
            formats=[("<i4", (3,)), "<u2"],
            offsets=[0, 12],
            itemsize=stride,
        )
    )
    rows = np.frombuffer(
        binary, dtype=field, count=len(keys), offset=rec["offsetBytes"]
    )
    if not np.array_equal(rows["coords"], keys):
        raise ValueError(name + " vertex-order mismatch")
    return rows["value"] / 65535.0


def first_hit(points, mesh, direction=1):
    """Extreme neutral X on a surface along a ray through each query's Y/Z."""
    origins = points.copy()
    origins[:, 0] = -direction
    axis = np.zeros_like(points)
    axis[:, 0] = direction
    _, row, hit = RayMeshIntersector(mesh).intersects_id(
        origins, axis, multiple_hits=False, return_locations=True
    )
    depth = np.full(len(points), np.nan)
    depth[row] = hit[:, 0]
    return depth


def reference_interfaces(verts, faces):
    # All point coordinates here are local to the neutral femoral head centre.
    band = verts[0][:, :3]
    w = verts[0][:, 3]
    tri = faces[0]
    sheet = trimesh.Trimesh(band, tri, process=False)
    tfl = trimesh.Trimesh(verts[1][:, :3], faces[1] - len(band), process=False)
    samples = (
        np.array(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [0.7, 0.15, 0.15],
                [0.15, 0.7, 0.15],
                [0.15, 0.15, 0.7],
            ]
        )
        @ band[tri]
    ).reshape(-1, 3)
    front = first_hit(samples, sheet)
    outer = (abs(front - samples[:, 0]) <= 0.00005).reshape(-1, 4)
    xgap = (first_hit(samples, tfl) - samples[:, 0]).reshape(-1, 4)
    finite = np.isfinite(xgap)
    tfl_cover = (
        (sheet.face_normals[:, 0] < -0.15)
        & outer[:, 0]
        & (outer.sum(axis=1) >= 3)
        & np.all(~finite | (xgap >= -0.00005), axis=1)
    )
    # The tract encloses part of TFL. Material on its medial sheet must retain
    # this side instead of being lifted through the TFL belly.
    xnear = first_hit(band, tfl)
    xfar = first_hit(band, tfl, -1)
    medial = (
        np.isfinite(xnear)
        & np.isfinite(xfar)
        & (band[:, 0] > (xnear + xfar) / 2 + 0.0005)
        & (abs(band[:, 0] - xfar) < 0.008)
    )
    ends = (w == 0) | (w == 1)
    fixed = ends | medial
    # The band can pass into the attachment bone in the neutral dissection.
    # Preserve those true insertions. Other band faces still contact bone.
    crest = np.max(w[tri], axis=1) <= 0.035
    masks = np.full(len(tri), (1 << len(OBSTACLES)) - 1, dtype="<u4")
    masks[~tfl_cover] &= np.uint32(
        ~(1 << OBSTACLES.index("Tensor fasciae latae")) & 0xFFFFFFFF
    )
    masks[crest] &= np.uint32(~(1 << OBSTACLES.index("Hip bone")) & 0xFFFFFFFF)
    # A material metric, connected along the segmented band surface; directly
    # couple nearby skin points across a thin lamina but never shortcut through
    # the TFL belly from its superficial to its deep investing sheet.
    edge = np.unique(
        np.sort(
            np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1
        ),
        axis=0,
    )
    d = cdist(band, band)
    u, v = np.where((d < 0.0035) & (d > 0))
    linked = np.unique(
        np.sort(np.vstack((edge, np.column_stack((u, v)))), axis=1), axis=0
    )
    u, v = linked.T
    du = band[u] - band[v]
    dist = np.linalg.norm(du, axis=1)
    thin = dist < 0.0035
    dist[thin] = np.linalg.norm(du[thin] * [0.25, 1, 1], axis=1)
    graph = csr_matrix((np.r_[dist, dist], (np.r_[u, v], np.r_[v, u])), shape=d.shape)
    metric = shortest_path(graph, directed=False).astype("<f4")
    # The largest displacement allowed by pinned endpoints and the slope bound.
    cap = 0.28 * metric[:, fixed].min(axis=1)
    if not np.isfinite(metric).all():
        raise ValueError("Disconnected tract mesh")
    return (
        metric,
        masks,
        cap,
        dict(
            tflCoverFaceIds=np.flatnonzero(tfl_cover).tolist(),
            tflMedialVertexIds=np.flatnonzero(medial).tolist(),
            attachmentVertexIds=np.flatnonzero(ends).tolist(),
            iliacAttachmentFaces=np.flatnonzero(crest).tolist(),
        ),
    )


def prepare():
    meshes, center, sha = load_meshes(ATLAS)
    fields = json.loads((DATA / "hip-deformation.json").read_text())
    weights = (DATA / "hip-attachments.bin").read_bytes()
    if (
        fields["sourceAtlasSha256"] != sha
        or hashlib.sha256(weights).hexdigest() != fields["weightsSha256"]
    ):
        raise ValueError("Regenerate atlas attachment fields first.")
    verts = []
    faces = []
    records = []
    nv = 0
    for name in [BAND, *OBSTACLES]:
        keys, tri, _ = meshes[name]
        xyz = keys / UNIT - center
        w = read_weights(name, keys, fields, weights)
        verts.append(np.column_stack((xyz, w)))
        faces.append(tri + nv)
        records.append(
            dict(label=name, vertexStart=nv, vertices=len(xyz), triangles=len(tri))
        )
        nv += len(xyz)
    band = verts[0]
    metric, masks, cap, interfaces = reference_interfaces(verts, faces)
    tags = np.concatenate(
        [np.full(len(f), 1 << i, dtype="<u4") for i, f in enumerate(faces[1:])]
    )
    header = np.array(
        [nv, len(band), len(faces[0]), sum(map(len, faces[1:]))], dtype="<u4"
    )
    payload = b"".join(
        a.tobytes()
        for a in [
            header,
            np.concatenate(verts).astype("<f4"),
            np.concatenate(faces).astype("<u4"),
            metric,
            cap.astype("<f4"),
            masks,
            tags,
        ]
    )
    MODEL.write_bytes(payload)
    return dict(
        centerM=center.tolist(),
        models=records,
        atlasSha256=sha,
        attachmentsSha256=fields["weightsSha256"],
        solverSourceSha256=hashlib.sha256(C_SOURCE.read_bytes()).hexdigest(),
        contactInputSha256=hashlib.sha256(payload).hexdigest(),
        clearanceMm=0.25,
        maxAddedDisplacementSlope=0.28,
        localThroughThicknessMetricScale=0.25,
        interfaces=interfaces,
    )


def compile_wasm():
    clang = (
        os.environ.get("WASM_CC")
        or shutil.which("clang", path="/opt/homebrew/opt/llvm/bin")
        or shutil.which("clang")
    )
    ld = os.environ.get("WASM_LD") or shutil.which("wasm-ld")
    if not ld:
        raise RuntimeError("Install LLVM and wasm-ld to build the browser solver")
    args = [
        clang,
        "--target=wasm32",
        "-O3",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fno-builtin",
        "-nostdlib",
        "-fuse-ld=" + ld,
        str(C_SOURCE),
        "-Wl,--no-entry",
        "-Wl,--allow-undefined",
        "-Wl,--export-memory",
        "-Wl,--initial-memory=16777216",
        "-Wl,--max-memory=16777216",
    ]
    names = [
        "contact_input",
        "contact_initialize",
        "contact_solve",
        "contact_output",
        "contact_offsets",
        "contact_metrics",
    ]
    subprocess.run(
        args + ["-Wl,--export=" + k for k in names] + ["-o", str(WASM)], check=True
    )


class HipContact:
    """positions([flex,add,twist]) returns sorted welded ITB XYZ, in atlas metres."""

    def __init__(self):
        binary = MODEL.read_bytes()
        self.scratch = tempfile.TemporaryDirectory()
        lib = Path(self.scratch.name) / "contact.dylib"
        subprocess.run(
            [
                os.environ.get("CC", "clang"),
                "-O3",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-shared",
                "-fPIC",
                str(C_SOURCE),
                "-o",
                str(lib),
            ],
            check=True,
        )
        self.lib = ctypes.CDLL(str(lib))
        self.lib.contact_input.restype = ctypes.c_void_p
        initialize = self.lib.contact_initialize
        initialize.argtypes = [ctypes.c_uint]
        ctypes.memmove(self.lib.contact_input(), binary, len(binary))
        if initialize(len(binary)):
            raise RuntimeError("Incompatible input data")
        self.lib.contact_solve.argtypes = [ctypes.c_double] * 4
        self.lib.contact_solve.restype = ctypes.c_int
        counts = np.frombuffer(binary, "<u4", count=4)
        rest = np.frombuffer(
            binary, "<f4", count=int(4 * counts[0]), offset=16
        ).reshape(-1, 4)
        self.n = int(counts[1])
        self.rest = rest[: self.n, :3].astype(float)
        self.weights = rest[: self.n, 3].astype(float)
        self.triangles = np.frombuffer(
            binary, "<u4", count=int(3 * counts[2]), offset=16 + 16 * int(counts[0])
        ).reshape(-1, 3)
        self.center = load_meshes(ATLAS)[1]
        for func, attr, size in [
            ("contact_output", "vertices", 3 * self.n),
            ("contact_offsets", "offsets", self.n),
            ("contact_metrics", "metrics", 32),
        ]:
            getter = getattr(self.lib, func)
            getter.restype = ctypes.POINTER(ctypes.c_double)
            setattr(self, attr, np.ctypeslib.as_array(getter(), shape=(size,)))

    def positions(self, pose):
        flex, add, twist = pose
        rv = Rotation.from_euler("ZXY", [add, -flex, twist], degrees=True).as_rotvec()
        a = np.linalg.norm(rv)
        axis = rv / a if a else [0.0, 1.0, 0.0]
        status = self.lib.contact_solve(*axis, a)
        if status:
            raise RuntimeError(f"Contact solver status {status} at {pose}")
        return self.vertices.reshape(-1, 3).copy() + self.center


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    result = prepare()
    compile_wasm()
    result["wasmSha256"] = hashlib.sha256(WASM.read_bytes()).hexdigest()
    (DATA / "hip-contact.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Ready", MODEL.stat().st_size, WASM.stat().st_size)
