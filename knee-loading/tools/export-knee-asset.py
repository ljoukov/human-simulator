#!/usr/bin/env python3
"""Export the knee tissues and their fitted flexion clip from the lower-limb GLB."""

import argparse
import copy
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lower-limb/assets/lower-limb.glb"
OUTPUT = ROOT / "knee-loading/assets/knee.glb"
TISSUES = {
    "Femur",
    "Tibia",
    "Fibula",
    "Patella",
    "Art cart of femur distal end",
    "Art cart of tibia proximal end",
    "Art cart of patella",
    "Art cart of fibula proximal tibiofibular joint",
    "Art cart of tibia proximal tibiofibular joint",
    "Lateral meniscus",
    "Medial meniscus",
    "Anterior horn of Lateral meniscus",
    "Anterior horn of Medial meniscus",
    "Posterior horn of Lateral meniscus",
    "Posterior horn of Medial meniscus",
    "Anterior cruciate ligament",
    "Posterior cruciate ligament",
    "Fibular collateral ligament",
    "Medial collatertal ligament",
    "Posterior meniscofemoral ligament",
    "Transverse ligament of knee",
    "Oblique popliteal ligament",
    "Lateral patellar retinaculum (horizontal part)",
    "Lateral patellar retinaculum (vertical part)",
    "Medial patellar retinaculum (horizontal part)",
    "Medial patellar retinaculum (vertical part)",
    "Quadriceps common tendon and patellar ligament",
    "Common tendon of biceps femoris",
    "Rectus femoris",
    "Vastus lateralis muscle",
    "Vastus medialis muscle",
    "Long head of biceps femoris",
    "Medial head of gastrocnemius",
    "Lateral head of gastrocnemius",
    "Popliteus muscle",
}
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def read_glb(path):
    contents = path.read_bytes()
    magic, version, size = struct.unpack_from("<4sII", contents)
    if magic != b"glTF" or version != 2 or size != len(contents):
        raise ValueError("Expected a complete glTF 2.0 binary")
    json_length, json_type = struct.unpack_from("<II", contents, 12)
    binary_header = 20 + json_length
    bin_length, bin_type = struct.unpack_from("<II", contents, binary_header)
    if json_type != JSON_CHUNK or bin_type != BIN_CHUNK or binary_header + 8 + bin_length != size:
        raise ValueError("Expected one JSON chunk and one binary chunk")
    document = json.loads(contents[20:binary_header])
    if len(document["buffers"]) != 1 or document.get("images") or document.get("extensionsUsed"):
        raise ValueError("Expected the untextured lower-limb GLB with one buffer")
    return document, memoryview(contents)[binary_header + 8:]


def pack_glb(document, binary):
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    size = 28 + len(encoded) + len(binary)
    return (
        struct.pack("<4sIIII", b"glTF", 2, size, len(encoded), JSON_CHUNK)
        + encoded
        + struct.pack("<II", len(binary), BIN_CHUNK)
        + binary
    )


def export_knee(source):
    original, source_binary = read_glb(source)
    assets = original["nodes"]
    selected = {index for index, node in enumerate(assets) if node.get("name") in TISSUES}
    missing = TISSUES - {assets[index]["name"] for index in selected}
    if missing:
        raise ValueError(f"The source GLB is missing knee tissue meshes {sorted(missing)}")
    parents = {child: index for index, node in enumerate(assets) for child in node.get("children", [])}
    skin_ids = sorted({assets[index]["skin"] for index in selected if "skin" in assets[index]})
    selected.update(joint for skin in skin_ids for joint in original["skins"][skin]["joints"])
    for node in tuple(selected):
        while node in parents:
            node = parents[node]
            selected.add(node)

    node_ids = {old: new for new, old in enumerate(sorted(selected))}
    skin_map = {old: new for new, old in enumerate(skin_ids)}
    document = {key: copy.deepcopy(original[key]) for key in ("asset", "scene", "scenes", "materials")}
    document["asset"]["extras"]["changes"] += " Knee tissues and their animation extracted without decimation."
    for scene in document["scenes"]:
        scene["name"] = "Knee tissues"
        scene["nodes"] = [node_ids[index] for index in scene["nodes"] if index in node_ids]
    document.update(nodes=[], meshes=[], accessors=[], bufferViews=[], animations=[], skins=[])
    binary = bytearray()
    accessors = {}
    views = {}

    def buffer_view(index):
        if index not in views:
            entry = copy.deepcopy(original["bufferViews"][index])
            begin = entry.get("byteOffset", 0)
            end = begin + entry["byteLength"]
            binary.extend(b"\0" * (-len(binary) % 4))
            entry["byteOffset"] = len(binary)
            binary.extend(source_binary[begin:end])
            views[index] = len(document["bufferViews"])
            document["bufferViews"].append(entry)
        return views[index]

    def accessor(index):
        if index not in accessors:
            entry = copy.deepcopy(original["accessors"][index])
            if "bufferView" in entry:
                entry["bufferView"] = buffer_view(entry["bufferView"])
            if "sparse" in entry:
                for part in ("indices", "values"):
                    sparse = entry["sparse"][part]
                    sparse["bufferView"] = buffer_view(sparse["bufferView"])
            accessors[index] = len(document["accessors"])
            document["accessors"].append(entry)
        return accessors[index]

    for index in node_ids:
        node = copy.deepcopy(assets[index])
        if "children" in node:
            node["children"] = [node_ids[child] for child in node["children"] if child in node_ids]
        if "mesh" in node:
            mesh = copy.deepcopy(original["meshes"][node["mesh"]])
            for primitive in mesh["primitives"]:
                primitive["attributes"] = {key: accessor(value) for key, value in primitive["attributes"].items()}
                primitive["indices"] = accessor(primitive["indices"])
                if "targets" in primitive:
                    primitive["targets"] = [
                        {key: accessor(value) for key, value in target.items()} for target in primitive["targets"]
                    ]
            node["mesh"] = len(document["meshes"])
            document["meshes"].append(mesh)
        if "skin" in node:
            node["skin"] = skin_map[node["skin"]]
        document["nodes"].append(node)

    for index in skin_ids:
        skin = copy.deepcopy(original["skins"][index])
        skin["joints"] = [node_ids[joint] for joint in skin["joints"]]
        if "skeleton" in skin:
            skin["skeleton"] = node_ids[skin["skeleton"]]
        skin["inverseBindMatrices"] = accessor(skin["inverseBindMatrices"])
        document["skins"].append(skin)

    for animation in original["animations"]:
        channels = []
        samplers = []
        for channel in animation["channels"]:
            target = channel["target"]
            if target["node"] not in node_ids:
                continue
            sampler = copy.deepcopy(animation["samplers"][channel["sampler"]])
            sampler["input"] = accessor(sampler["input"])
            sampler["output"] = accessor(sampler["output"])
            channels.append({"sampler": len(samplers), "target": {**target, "node": node_ids[target["node"]]}})
            samplers.append(sampler)
        if channels:
            document["animations"].append({"name": animation["name"], "channels": channels, "samplers": samplers})

    document["buffers"] = [{"byteLength": len(binary)}]
    return pack_glb(document, binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    options = parser.parse_args()
    result = export_knee(options.source)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_bytes(result)
    print(f"Wrote {len(result):,} bytes to {options.output}")


if __name__ == "__main__":
    main()
