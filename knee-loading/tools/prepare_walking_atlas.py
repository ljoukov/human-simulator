"""Prepare the registered BodyParts3D walking atlas from the 20110915 OBJ99 ZIP.

Requires numpy, trimesh and fast-simplification. The accompanying source note
links both inputs. Output uses OrganUI atlas coordinates rotated to +X anterior.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import struct
from zipfile import ZipFile

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
CENTER = np.array([-0.667999267578125, -109.74665069580078, 817.0244853496552])
SEGMENTS = ['pelvis', 'torso', 'head'] + [
    side + bone for side in ('right', 'left')
    for bone in ('Thigh', 'Shin', 'Foot', 'Toe', 'Arm', 'Forearm', 'ForearmMid', 'ForearmDistal', 'Hand')
]
SKIN_ID = 'FMA7163'

# Names are matched against the publisher's English FMA labels. Their OBJ99
# collection contains separate bilateral structures, including individual heads.
ARM = re.compile(r'deltoid|brachi|triceps|anconeus|pronator|supinator|palmaris|carpi|digitorum (superficialis|profundus)|pollicis|digiti minimi of hand|opponens|extensor indicis|extensor digitorum$|of (right |left )?hand|flexor retinaculum of .*wrist|extensor digiti minimi')
LEG = re.compile(r'iliacus|psoas|gluteus|fasciae latae|piriformis|obturator|gemellus|quadratus femoris|sartorius|vastus|rectus femoris|pectineus|adductor (magnus|minimus|longus|brevis)|gracilis|biceps femoris|semitendinosus|semimembranosus|tibialis|hallucis|fibularis|gastrocnemius|soleus|plantaris|popliteus|digitorum (longus|brevis)|calcaneal tendon|iliotibial tract|of (right |left )?foot|flexor accessorius')
TRUNK = re.compile(r'trapezius|latissimus|rhomboid|levator scapulae|iliocostalis|longissimus|spinalis|splenius|multifidus|pectoralis|serratus|intercostal muscle|subclavius|rectus abdominis|external oblique|internal oblique|transversus abdominis|linea alba|quadratus lumborum|infraspinatus|supraspinatus|teres (major|minor)|subscapularis|sternocleidomastoid|scalenus|platysma|sternohyoid|omohyoid|thyrohyoid')
FACE = re.compile(r'frontal belly|occipital belly|epicranius|orbicularis|nasalis|procerus|buccinator|masseter|temporalis|zygomaticus|corrugator|risorius|mentalis|depressor .*oris|levator .*oris|labii|nasal cartilage|auricularis|ear cartilage|cartilage of .*auricle|tooth|eyeball')
TENDON = re.compile(r'tendon|aponeurosis|tract|linea alba|retinaculum|cartilage|tooth|eyeball')


def smooth(x, a, b):
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def weights(points, region, name, rigid=None, arm_override=None):
    """Local joint blends keep muscle groups separated where nearby limbs cross."""
    x, y, z = points.T
    w = np.zeros((len(points), len(SEGMENTS)))
    if rigid:
        w[:, SEGMENTS.index(rigid)] = 1
    else:
        # At the axilla the shoulder cap spans the thorax and humerus. The
        # pectorals and latissimus attach nearer the humerus than the obliques.
        edge = np.interp(y, [-.45, 0, .55, .98, 1.20, 1.38, 1.48], [.52, .47, .44, .39, .32, .34, .50])
        arm = smooth(abs(z), edge-.028, edge+.04) * smooth(y, -.53, -.42) * (1-smooth(y, 1.36, 1.48))
        if arm_override is not None:
            arm = arm_override
        elif region == 'arm':
            shoulder = smooth(abs(z), .31, .43)
            arm = 1 - smooth(y, 1.08, 1.32) * (1-shoulder)
        elif region == 'leg' or region == 'head':
            arm[:] = 0
        elif region == 'trunk' and not re.search(r'pectoralis|latissimus|teres|spinatus|subscapularis', name):
            arm[:] = 0
        torso = smooth(y, .20, .61)
        head = smooth(y, 1.48, 1.68)
        thigh = 1-smooth(y, -.18, .16)
        shin = 1-smooth(y, -1.16, -.92)
        foot = 1-smooth(y, -1.99, -1.81)
        toes = smooth(x, .095, .23)
        if region == 'head':
            head[:] = 1
            torso[:] = 1
        w[:, 0] = (1-arm)*(1-thigh)*(1-torso)
        w[:, 1] = (1-arm)*(1-thigh)*torso*(1-head)
        w[:, 2] = (1-arm)*(1-thigh)*torso*head
        forearm = 1-smooth(y, .48, .66)
        hand = 1-smooth(y, -.075, .075)
        for s, side in enumerate(('right', 'left')):
            mask = z >= 0 if side == 'right' else z < 0
            off = 3+s*9
            leg = mask*(1-arm)*thigh
            w[:, off] = leg*(1-shin)
            w[:, off+1] = leg*shin*(1-foot)
            w[:, off+2] = leg*shin*foot*(1-toes)
            w[:, off+3] = leg*shin*foot*toes
            w[:, off+4] = mask*arm*(1-forearm)
            fore = mask*arm*forearm*(1-hand)
            # Pronation is distributed from the elbow along the forearm.
            # Two short angular interpolation intervals preserve its volume.
            mid = 1-smooth(y, .24, .54)
            distal = 1-smooth(y, .01, .26)
            w[:, off+5] = fore*(1-mid)
            w[:, off+6] = fore*mid*(1-distal)
            w[:, off+7] = fore*mid*distal
            w[:, off+8] = mask*arm*forearm*hand
    idx = np.argsort(-w, axis=1)[:, :4]
    top = np.take_along_axis(w, idx, axis=1)
    quant = np.rint(255 * top / top.sum(axis=1)[:, None]).astype(int)
    quant[:, 0] += 255-quant.sum(axis=1)
    return idx.astype('u1'), quant.astype('u1')


def bone_region(name):
    s = 'right' if name.startswith('right') else 'left'
    for pattern, segment in [(r'femur|patella', 'Thigh'), (r'tibia$|fibula$', 'Shin'), (r'toe-', 'Toe'),
                             (r'talus|calcaneus|navicular|cuboid|cuneiform|metatarsal|of-the-foot', 'Foot'),
                             (r'humerus', 'Arm'), (r'ulna', 'Forearm'), (r'radius', 'ForearmDistal'),
                             (r'carpal|finger|thumb|scaphoid|lunate|triquetral|pisiform|trapez|capitate|hamate', 'Hand')]:
        if re.search(pattern, name):
            return s+segment
    if re.search(r'hip-bone|sacrum', name):
        return 'pelvis'
    if re.search(r'vertebr|sternum|rib|costal|clavicle|scapula|xiphoid|atlas-c1|axis-c2', name):
        return None
    return 'head'


def load_obj(archive, path, limit):
    mesh = trimesh.load(io.BytesIO(archive.read(path)), file_type='obj', force='mesh', process=False)
    # BodyParts3D includes normals at seams. Merge coincident vertices before
    # decimation and recalculate smooth vertex normals from the remaining faces.
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if path.endswith(SKIN_ID+'.obj'):
        # The scan models dermis as a two-sided organ with distinct inner
        # and outer walls. Keep only its continuous exterior for display.
        adjacency = vertex_graph(mesh)
        _, labels = connected_components(adjacency, directed=False)
        exterior = np.bincount(labels).argmax()
        mesh.update_faces(labels[mesh.faces[:,0]] == exterior)
        mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > limit:
        mesh = mesh.simplify_quadric_decimation(face_count=limit)
    p = (mesh.vertices-CENTER)*.0025
    mesh.vertices = p[:, [1, 2, 0]] * [-1, 1, -1]
    return mesh


def vertex_graph(mesh):
    i, j = mesh.edges_unique.T
    w = 1/np.maximum(np.linalg.norm(mesh.vertices[i]-mesh.vertices[j], axis=1), 1e-6)
    return coo_matrix((np.r_[w,w],(np.r_[i,j],np.r_[j,i])),shape=(len(mesh.vertices),len(mesh.vertices))).tocsr()


def skin_arm_weight(mesh):
    """Use skin connectivity to separate medial elbows from the flanks."""
    p = mesh.vertices
    y, lateral = p[:,1], np.abs(p[:,2])
    graph = vertex_graph(mesh)
    below = np.flatnonzero(y < .75)
    count, labels = connected_components(graph[below][:,below], directed=False)
    arm = np.zeros(len(p))
    for i in range(count):
        component = below[labels == i]
        if lateral[component].mean() > .49:
            arm[component] = 1.
    above = y >= 1.25
    arm[above] = smooth(lateral[above],.30,.43) * (1-smooth(y[above],1.36,1.48))
    free = (y >= .75) & ~above & (lateral > .28) & (lateral < .56)
    arm[(y >= .75) & ~above & (lateral >= .56)] = 1.
    lap = diags(np.asarray(graph.sum(axis=1)).ravel()) - graph
    arm[free] = spsolve(lap[free][:,free].tocsc(), graph[free][:,~free] @ arm[~free])
    return np.clip(arm,0,1)


def neutralize_pelvic_surface(mesh):
    """Span the anterior pelvic region with a smooth patch sharing its boundary.

    A discrete Dirichlet solve replaces the interior vertex positions with
    harmonic coordinates of the surrounding pelvis/upper thigh boundary.
    Back, legs, hands and the remainder of the scanned surface are untouched.
    """
    p = mesh.vertices.copy()
    x, y, z = p.T
    region = ((z/.175)**2 + ((y+.185)/.25)**2 < 1) & (x > -.105)
    # Positive inverse-edge-length weights maintain a smooth surface even at
    # the different triangle densities of adjacent scanned structures.
    adj = vertex_graph(mesh)
    lap = diags(np.asarray(adj.sum(axis=1)).ravel()) - adj
    free = np.flatnonzero(region)
    boundary = np.flatnonzero(~region)
    p[free] = spsolve(lap[free][:,free].tocsc(), adj[free][:,boundary] @ p[boundary])
    mesh.vertices = p
    return mesh


def colors_and_uv(points, name, kind):
    # Stable bilateral coloring and mild procedural fiber detail are for
    # anatomical illustration. They carry no measured activation information.
    key = re.sub(r'\b(right|left)\b', '', name)
    variant = hashlib.sha256(key.encode()).digest()[0] / 255
    if kind == 'Bones':
        rgb = np.array([210, 200, 174]) if 'cartilage' not in name and 'disc' not in name else np.array([168, 184, 169])
    elif kind == 'Skin':
        rgb = np.array([173, 171, 159])
    elif TENDON.search(name):
        rgb = np.array([208, 183, 153])
    else:
        rgb = np.array([153+35*variant, 63+23*variant, 57+16*variant])
    linear = ((rgb/255+.055)/1.055)**2.4
    color = np.tile(np.rint(255*linear).astype('u1'), (len(points), 1))
    if kind != 'Muscles' or TENDON.search(name):
        return color, np.zeros((len(points), 2), dtype='<f4')
    # Phase in the plane normal to the structure's long axis gives
    # continuous longitudinal shading from any camera angle.
    centered = points-points.mean(axis=0)
    _, directions = np.linalg.eigh(centered.T @ centered)
    across = centered @ (directions[:, 0]*.74 + directions[:, 1]*.67)
    along = centered @ directions[:, 2]
    return color, np.column_stack((across, along)).astype('<f4')


def export(archive_path, labels_path, output):
    bone_asset = (ROOT/'hip-study/assets/body.glb').read_bytes()
    bone_json = json.loads(bone_asset[20:20+struct.unpack_from('<I', bone_asset, 12)[0]])
    bone_labels = {node['extras']['fma']:node['name'] for node in bone_json['nodes']}
    labels = dict(line.strip().split('\t', 1) for line in labels_path.read_text().splitlines() if '\t' in line)
    doc = {'asset': {'version':'2.0','generator':'prepare_walking_atlas.py',
           'copyright':'BodyParts3D © The Database Center for Life Science, CC BY-SA 2.1 Japan'},
           'scene':0, 'scenes':[{'nodes':[0]}], 'nodes':[{'name':'Walking atlas','children':[]}],
           'buffers':[], 'bufferViews':[], 'accessors':[], 'meshes':[], 'materials':[], 'skins':[]}
    binary = bytearray()

    def attribute(arr, typ, component, target, normalized=False):
        data = np.ascontiguousarray(arr).tobytes()
        binary.extend(b'\x00'*(-len(binary)%4))
        doc['bufferViews'].append({'buffer':0,'byteOffset':len(binary),'byteLength':len(data),'target':target})
        binary.extend(data)
        acc = {'bufferView':len(doc['bufferViews'])-1, 'type':typ, 'componentType':component, 'count':len(arr)}
        if normalized:
            acc['normalized'] = True
        if typ == 'VEC3' and component == 5126:
            acc.update(min=arr.min(axis=0).tolist(), max=arr.max(axis=0).tolist())
        doc['accessors'].append(acc)
        return len(doc['accessors'])-1

    for seg in SEGMENTS:
        doc['nodes'][0]['children'].append(len(doc['nodes']))
        doc['nodes'].append({'name':seg})
    doc['skins'].append({'name':'Atlas segments','joints':list(range(1, len(SEGMENTS)+1))})

    with ZipFile(archive_path) as archive:
        files = {Path(f.replace(chr(92), '/')).stem:f for f in archive.namelist() if f.lower().endswith('.obj') and not f.startswith('__')}
        for kind in ['Muscles','Bones','Skin']:
            arrays = [[] for _ in range(7)]
            total = 0
            structures = []
            for fma, path in sorted(files.items()):
                name = labels.get(fma, '')
                region = next((region for region, reg in [('leg', LEG), ('arm', ARM), ('trunk', TRUNK), ('head', FACE)] if reg.search(name)), None)
                if kind == 'Bones' and fma not in bone_labels: continue
                if kind == 'Skin' and fma != SKIN_ID: continue
                if kind == 'Muscles' and (not region or fma in bone_labels or fma == SKIN_ID or re.search(r'artery|vein|colliculus|\bbone\b', name)): continue
                limit = 850 if kind == 'Muscles' else 480
                if kind == 'Muscles' and re.search(r'vastus|gluteus|trapezius|external oblique|pectoralis|latissimus|rectus femoris|rectus abdominis|gastrocnemius|deltoid|soleus', name):
                    limit = 1850
                if kind == 'Bones' and re.search(r'femur|tibia|hip-bone|frontal-bone|mandible|scapula', bone_labels[fma]): limit = 1500
                if kind == 'Skin': limit = 26000
                mesh = load_obj(archive, path, limit)
                if kind == 'Skin': mesh = neutralize_pelvic_surface(mesh)
                p = mesh.vertices.astype('<f4')
                n = mesh.vertex_normals.astype('<f4')
                if kind == 'Bones': name = bone_labels[fma]
                ids, w = weights(p, region if kind == 'Muscles' else 'skin', name, bone_region(name) if kind == 'Bones' else None,
                                 skin_arm_weight(mesh) if kind == 'Skin' else None)
                c, uv = colors_and_uv(p, name, kind)
                indices = mesh.faces.astype('<u4').ravel() + total
                for store, data in zip(arrays, [p,n,c,uv,ids,w,indices]): store.append(data)
                structures.append({'id':fma,'name':name,'triangles':len(indices)//3})
                total += len(p)
            if not total: continue
            p,n,c,uv,ids,w,indices = [np.concatenate(a) for a in arrays]
            primitive = {'attributes':{
                'POSITION':attribute(p, 'VEC3', 5126, 34962),
                'NORMAL':attribute(n, 'VEC3', 5126, 34962),
                'COLOR_0':attribute(np.column_stack((c, np.full(len(c),255,'u1'))),'VEC4',5121,34962,True),
                'JOINTS_0':attribute(ids,'VEC4',5121,34962),
                'WEIGHTS_0':attribute(w,'VEC4',5121,34962,True)},
                'indices':attribute(indices.astype('<u2' if total<65536 else '<u4'), 'SCALAR',5123 if total<65536 else 5125,34963),
                'material':len(doc['materials'])}
            if kind == 'Muscles': primitive['attributes']['TEXCOORD_0'] = attribute(uv,'VEC2',5126,34962)
            doc['materials'].append({'name':kind,'pbrMetallicRoughness':{'metallicFactor':0,'roughnessFactor':.7}})
            doc['nodes'][0]['children'].append(len(doc['nodes']))
            doc['nodes'].append({'name':kind,'mesh':len(doc['meshes']),'skin':0,'extras':{'structures':structures}})
            doc['meshes'].append({'name':kind,'primitives':[primitive]})
            print(kind, len(structures), 'structures,', len(indices)//3, 'triangles', flush=True)
    binary.extend(b'\x00'*(-len(binary)%4))
    doc['buffers'].append({'byteLength':len(binary)})
    header=json.dumps(doc,separators=(',',':')).encode()
    header += b' '*(-len(header)%4)
    output.write_bytes(struct.pack('<4sIII4s',b'glTF',2,28+len(header)+len(binary),len(header),b'JSON')+header+struct.pack('<I4s',len(binary),b'BIN\0')+binary)


def retain_credit(path):
    """gltfpack 0.24 omits the standard copyright field when packing."""
    data = path.read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20+size])
    doc['asset']['copyright'] = 'BodyParts3D © The Database Center for Life Science, CC BY-SA 2.1 Japan'
    doc['asset']['extras'] = {
        'source': 'https://dbarchive.biosciencedbc.jp/data/bodyparts3d/20110915/',
        'license': 'https://creativecommons.org/licenses/by-sa/2.1/jp/deed.en',
        'modifications': 'Selected and simplified surfaces, converted anatomical axes, added skin weights and illustrative colors, reconstructed the anterior pelvic surface, and applied meshoptimizer compression.',
    }
    header = json.dumps(doc, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 4)
    tail = data[20+size:]
    path.write_bytes(struct.pack('<4sIII4s', b'glTF', 2, 20+len(header)+len(tail), len(header), b'JSON') + header + tail)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'knee-loading/assets/walking-atlas.glb')
    parser.add_argument('--gltfpack', default='gltfpack', help='Path to the gltfpack 0.24.0 executable')
    options=parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        raw = Path(directory)/'atlas.glb'
        export(options.archive,options.labels,raw)
        subprocess.run([options.gltfpack, '-i', str(raw), '-o', str(options.output),
                        '-cc','-vpf','-vtf','-vp','16','-vt','16','-kn','-km','-ke','-kv'], check=True)
        retain_credit(options.output)
