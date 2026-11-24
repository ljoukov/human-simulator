"""Embed surface detail in the hip atlas while preserving the publisher's UV charts.

The publisher's normal maps describe individual bones, the capsule and ligaments.
The muscle and connective-tissue illustrations are generated independently here.
The latter describe the appearance of bundles on a teaching model.
"""
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image


def periodic_field(u, v, rng, frequencies):
    field = np.zeros_like(u)
    for across, along, weight in frequencies:
        phase = rng.uniform(0, 2 * np.pi)
        field += weight * np.cos(2 * np.pi * (across * u + along * v) + phase)
    return field


def tissue_maps(connective, size=768):
    """Illustrate gently divergent fascicle bundles in longitudinal atlas coordinates."""
    y, x = np.mgrid[0:size, 0:size] / size
    rng = np.random.default_rng(79 if connective else 303)
    bends = periodic_field(x, y, rng, [(0, 1, .011), (1, 2, .009), (3, -1, .005), (6, 2, .001)])
    cross = x + bends
    broad = periodic_field(x, y, rng, [(1, 0, .42), (3, 1, .28), (7, -2, .18), (11, 1, .12)])

    def oriented_noise(across, along, displacement):
        # A periodic, anisotropic random field gives bundles irregular spacings
        # and lengths. Warp the field along the atlas chart without UV seams.
        fx = np.fft.rfftfreq(size)[None, :] * size
        fy = np.fft.fftfreq(size)[:, None] * size
        weight = np.exp(-.5 * ((fx / across) ** 2 + (fy / along) ** 2))
        spectrum = np.fft.rfft2(rng.standard_normal(x.shape))
        field = np.fft.irfft2(spectrum * weight, s=x.shape, axes=(0, 1))
        field = (field - field.mean()) / field.std()
        at = (displacement % 1) * size
        left = np.floor(at).astype(int) % size
        blend = at - np.floor(at)
        rows = np.arange(size)[:, None]
        return field[rows, left] * (1 - blend) + field[rows, (left + 1) % size] * blend

    bundle = oriented_noise(30, 2.4, cross)
    fine = oriented_noise(86, 5.5, cross + .0025 * broad)
    mottling = oriented_noise(6, 2.5, x)
    septa = np.clip((bundle - .55) * .8, 0, 1) ** 1.4
    grain = rng.normal(0, 1.3, x.shape)
    if connective:
        base, relief = [211, 194, 173], .19
        color = np.asarray(base) + (5 * broad + 4 * bundle + 3 * fine + 9 * septa + grain)[..., None]
    else:
        base, relief = [153, 72, 66], .33
        lighting = 6 * mottling + 5 * broad + 3 * fine + 2.5 * bundle + grain
        color = np.asarray(base) + lighting[..., None] + septa[..., None] * [21, 27, 22]
    height = relief * (.95 * bundle + .41 * fine + .24 * broad)
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * .5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * .5
    normals = np.stack((-dx, dy, np.ones_like(dx)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    def encode(values, fmt):
        stream = BytesIO()
        image = Image.fromarray(np.rint(np.clip(values, 0, 255)).astype(np.uint8))
        image.save(stream, format=fmt, **({"quality": 88, "subsampling": 0} if fmt == "JPEG" else {}))
        return stream.getvalue()

    return encode(color, "JPEG"), encode((normals + 1) * 127.5, "PNG")


def attach_materials(atlas, binary, original_doc, original_binary):
    """Add embedded textures to a glTF asset in-place. Call before writing its buffer length."""
    atlas["textures"], atlas["images"] = [], []
    atlas["samplers"] = [{"minFilter": 9987, "magFilter": 9729, "wrapS": 10497, "wrapT": 10497}]
    maps, materials = {}, {}
    normal_sources = []

    def embed(name, mime, content):
        binary.extend(bytes(-len(binary) % 4))
        offset = len(binary)
        binary.extend(content)
        view = len(atlas["bufferViews"])
        atlas["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(content)})
        idx = len(atlas["images"])
        atlas["images"].append({"name": name, "mimeType": mime, "bufferView": view})
        atlas["textures"].append({"name": name, "sampler": 0, "source": idx})
        return idx

    def illustration(kind):
        if kind not in maps:
            color, normal = tissue_maps(connective=kind == "connective tissue")
            maps[kind] = (embed(kind + " colour", "image/jpeg", color),
                          embed(kind + " relief", "image/png", normal))
        return maps[kind]

    def source_normal(index):
        image_idx = original_doc["textures"][index]["source"]
        if image_idx not in maps:
            image = original_doc["images"][image_idx]
            view = original_doc["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            maps[image_idx] = embed(image["name"], image["mimeType"], original_binary[start:start + view["byteLength"]])
            normal_sources.append(image["name"])
        return maps[image_idx]

    for node in atlas["nodes"]:
        if "mesh" not in node:
            continue
        extras = node.get("extras", {})
        source_material = extras.get("sourceMaterial")
        if source_material is None and "sourceNode" in extras:
            raw_node = original_doc["nodes"][extras["sourceNode"]]
            source_material = original_doc["meshes"][raw_node["mesh"]]["primitives"][0].get("material")
        if source_material is None:
            continue
        source = original_doc["materials"][source_material]
        color_ref = source.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        normal_ref = source.get("normalTexture")
        kind = None
        if color_ref is not None:
            source_image = original_doc["images"][original_doc["textures"][color_ref["index"]]["source"]]["name"]
            if source_image in {"Muscle tiles plain", "Muscle long tendons", "Muscle tiles"}:
                kind = "muscle"
            elif source_image == "Tendon only":
                kind = "connective tissue"
        if not (kind or normal_ref):
            continue
        for prim in atlas["meshes"][node["mesh"]]["primitives"]:
            if "TEXCOORD_0" not in prim["attributes"]:
                continue
            key = (source_material, prim.get("material"))
            if key not in materials:
                mat = deepcopy(atlas["materials"][key[1]]) if key[1] is not None else {}
                mat["name"] = source["name"] + " · hip atlas"
                pbr = mat.setdefault("pbrMetallicRoughness", {})
                if kind:
                    albedo, normal = illustration(kind)
                    pbr.update(baseColorFactor=[1, 1, 1, 1], baseColorTexture={"index": albedo},
                               roughnessFactor=.68, metallicFactor=0)
                    mat["normalTexture"] = {"index": normal, "scale": .85}
                if normal_ref:
                    mat["normalTexture"] = {"index": source_normal(normal_ref["index"]),
                                            "scale": min(float(normal_ref.get("scale", 1)), 1.)}
                materials[key] = len(atlas["materials"])
                atlas["materials"].append(mat)
            prim["material"] = materials[key]
    atlas["asset"].setdefault("extras", {})["surfaceIllustration"] = {
        "sourceNormalMaps": normal_sources,
        "muscleAppearance": "Original procedural colour and normal maps follow the source UV charts. "
                            "Fascicles and connective septa illustrate surface appearance.",
    }
