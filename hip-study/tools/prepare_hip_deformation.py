"""Derive attachment weights and check hip motion against segmented atlas surfaces.

Run after prepare_hip_atlas.py. Requires NumPy, SciPy and trimesh.
Weights attach to neutral anatomical coordinates, so UV seams share motion.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, dijkstra, laplacian
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from trimesh import Trimesh
from trimesh.proximity import closest_point_naive
from trimesh.triangles import points_to_barycentric

ROOT = Path(__file__).resolve().parents[2]
ATLAS = ROOT / 'upper-limb/hip-study/assets/hip-atlas.glb'
DATA = ROOT / 'hip-study/data'
SPACING = .002
UNIT = 1_000_000  # Neutral positions in micrometres, shared across material/UV seams.
POSES = [[0,0,0], [-30,0,0], [75,0,0], [0,-25,0], [0,30,0], [0,0,-30], [0,0,30],
         [-30,-25,-30], [75,30,30]]  # [flexion, adduction, internal rotation], degrees.


def load_meshes(path):
    buf = path.read_bytes()
    size = struct.unpack_from('<I',buf,12)[0]
    doc = json.loads(buf[20:20+size])
    binary = memoryview(buf)[28+size:]
    def read(idx):
        ac = doc['accessors'][idx]
        bv = doc['bufferViews'][ac['bufferView']]
        width = {'SCALAR':1,'VEC3':3}[ac['type']]
        dtype = np.dtype({5123:'<u2',5125:'<u4',5126:'<f4'}[ac['componentType']])
        at = bv.get('byteOffset',0)+ac.get('byteOffset',0)
        return np.ndarray((ac['count'],width),dtype,buffer=binary,offset=at,
                          strides=(bv.get('byteStride',width*dtype.itemsize),dtype.itemsize))
    surfaces = {}
    for node in doc['nodes']:
        if 'mesh' not in node:
            continue
        if any(k in node for k in ('matrix','scale','rotation','translation')):
            raise ValueError('Expected neutral world-coordinate atlas')
        points,faces = [],[]
        count = 0
        for primitive in doc['meshes'][node['mesh']]['primitives']:
            p = read(primitive['attributes']['POSITION'])
            t = read(primitive['indices']).reshape(-1,3).astype(np.int64)
            points.append(p)
            faces.append(t+count)
            count += len(p)
        # Match JS rounding, including negative half coordinates.
        coords = np.floor(np.concatenate(points).astype(np.float64)*UNIT+.5).astype('<i4')
        key,inv = np.unique(coords,axis=0,return_inverse=True)
        tri = inv[np.concatenate(faces)]
        tri = tri[(tri[:,0]!=tri[:,1])&(tri[:,1]!=tri[:,2])&(tri[:,0]!=tri[:,2])]
        label = node.get('extras',{}).get('label',node['name'])
        surfaces[label] = (key,tri,node.get('extras',{}).get('regionGroup',''))
    center = np.array(doc['asset']['extras']['femoralHeadSphere']['centerM'])
    return surfaces,center,hashlib.sha256(buf).hexdigest()


def graph_edges(x,t):
    edge = np.unique(np.sort(np.concatenate((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])),axis=1),axis=0)
    u,v = edge.T
    length = np.linalg.norm(x[u]-x[v],axis=1)
    graph = csr_matrix((np.r_[length,length],(np.r_[u,v],np.r_[v,u])),shape=(len(x),len(x)))
    return graph


def bounded_samples(x,t):
    # Barycentric grids with maximum triangle edge spacing 2 mm give a
    # bounded nearest-sample approximation to each original triangle surface.
    tri = x[t]
    edge = np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2).max(axis=1)
    divisions = np.maximum(1,np.ceil(edge/SPACING).astype(int))
    patches = []
    for n in np.unique(divisions):
        ij = np.array([(i/n,j/n) for i in range(n+1) for j in range(n+1-i)])
        tau = np.column_stack((1-ij.sum(axis=1),ij))
        patches.append((tau@tri[divisions==n]).reshape(-1,3))
    return np.concatenate(patches)


class AttachmentSurfaces:
    def __init__(self, meshes, center):
        self.center = center
        self.samples,self.trees = {},{}
        for name in ['Hip bone','Sacrum','Femur','Acetabular labrum','Obturator membrane',
                     'Transverse acetabular ligament','Lumbar vertebra (L5)']:
            if name in meshes:
                key,t,_ = meshes[name]
                self.samples[name] = bounded_samples(key/UNIT,t)
        p,f = self.samples['Hip bone'], self.samples['Femur']
        cx,cy,_ = center
        self.samples.update({
            'pelvic ring':np.vstack((p,self.samples['Sacrum'])),
            'iliac wing':p[p[:,1]>cy+.02], 'iliac crest':p[p[:,1]>cy+.065],
            'ischium/pubis':p[p[:,1]<cy+.018],
            'trochanter':f[(f[:,0]<cx-.033)&(f[:,1]>cy-.064)],
            'femoral neck/shaft':f[(f[:,0]<cx-.024)|(f[:,1]<cy-.022)],
            'lesser trochanter vicinity':f[(f[:,1]<cy-.024)&(f[:,1]>cy-.09)&(f[:,0]>cx-.041)],
            'head':f[np.linalg.norm(f-center,axis=1)<.026],
        })
        for name, cloud in self.samples.items():
            if len(cloud):
                self.trees[name]=cKDTree(cloud)

    def distance(self,name,x):
        return self.trees[name].query(x)[0]


def strategy(label,group):
    s = label.lower()
    if label=='Femur' or 'cart of femur head' in s or 'vastus' in s:
        return 'femur'
    if 'burs' in s:
        if 'trochant' in s:
            return 'femur'
        return 'transfer'
    if s in ('gluteal aponeurosis',):
        return 'transfer'
    if label=='Ligament of head of femur':
        return ('Transverse acetabular ligament','head')
    if label=='Hip joint capsule':
        return ('Acetabular labrum','femoral neck/shaft')
    if any(k in s for k in ('iliofemoral ligament','pubofemoral ligament','ischiofemoral ligament')):
        return ('Hip bone','femoral neck/shaft')
    if 'zona orbicularis' in s:
        return 'transfer'
    if label in ('Gluteus medius muscle','Gluteus minimus muscle'):
        return ('iliac wing','trochanter')
    if 'maximus' in s:
        return ('pelvic ring','femoral neck/shaft')
    if label=='Piriformis muscle':
        return ('Sacrum','trochanter')
    if any(k in s for k in ('gemell','obturator internus','obturator externus','quadratus femoris')):
        return ('ischium/pubis','trochanter')
    if label=='Psoas major':
        return ('lumbar end','lesser trochanter vicinity')
    if label.startswith('Common tendon of Semitendinosus'):
        return 'transfer'
    if 'iliacus' in s:
        return ('iliac wing','lesser trochanter vicinity')
    if any(k in s for k in ('tensor fasciae','fascia lata','iliotibial tract','sartorius')):
        return ('iliac crest','distal section')
    if any(k in s for k in ('adductor','gracilis','pectineus','biceps femoris','semitendinosus',
                             'semimembranosus')):
        return ('ischium/pubis','distal section')
    if 'rectus femoris' in s:
        return ('Hip bone','distal section')
    if any(k in s for k in ('nerve','artery','vein','plexus','vessel')):
        return 'transfer'
    if (group in ('ligaments','context','bone','cartilage') or 'ligament' in s or
        'cartilage' in s or 'vertebra' in s or 'disc' in s or 'labrum' in s or
        label in ('Hip bone','Sacrum','Obturator membrane') or 'coccy' in s):
        return 'pelvis'
    raise ValueError(f'Assign an explicit motion model for {label!r} ({group}).')


def geodesic(graph,seeds):
    return dijkstra(graph,indices=np.flatnonzero(seeds),min_only=True)

def local_field(source, values, targets, radius):
    # Density-normalized convolution on a 2.5-mm 3D lattice. Opposing surfaces
    # of thin fascia and UV/tube seams sample the same continuous spatial field.
    step = .0025
    lo=np.minimum(source.min(axis=0),targets.min(axis=0))-step*2
    hi=np.maximum(source.max(axis=0),targets.max(axis=0))+step*2
    shape=np.ceil((hi-lo)/step).astype(int)+1
    ij = tuple(np.rint((source-lo)/step).astype(int).T)
    mass = np.zeros(shape); total=np.zeros(shape)
    np.add.at(mass,ij,1)
    np.add.at(total,ij,values)
    occupied = mass>0
    total[occupied] /= mass[occupied]
    mass[occupied]=1
    q=((targets-lo)/step).T
    def filtered(field):
        return map_coordinates(gaussian_filter(field,radius/step,mode='constant'),q,order=1)
    norm=filtered(mass)
    output = np.empty(len(targets))
    valid = norm>1e-9
    output[valid]=filtered(total)[valid]/norm[valid]
    if (~valid).any():
        output[~valid]=values[cKDTree(source).query(targets[~valid])[1]]
    return output

def bind_surface(x,t,refs,names):
    prox,dist = names
    # Some muscle insertions reach a fascia or continue below the atlas cut.
    # The local distal patch follows the thigh, explicitly a kinematic proxy.
    if prox=='lumbar end':
        y=x[:,1]; cap=x[y>=y.max()-.012]
        dp = cKDTree(cap).query(x)[0]
        if 'Lumbar vertebra (L5)' in refs.trees:
            dp = np.minimum(dp,refs.distance('Lumbar vertebra (L5)',x))
    else:
        dp = refs.distance(prox if prox in refs.trees else 'Hip bone',x)
    if dist=='distal section':
        y=x[:,1]; cap=x[y<=y.min()+min(.008,.12*np.ptp(y))]
        df = cKDTree(cap).query(x)[0]
    else:
        df = refs.distance(dist,x)
    ratio = dp/(dp+df+1e-12)
    # Pin near target tissue where the competing target is much farther.
    # A small relative-distance quantile also captures the closest footprint
    # when the source dissection leaves a gap between modelled surfaces.
    pin_p = (ratio<=np.quantile(ratio,.04)) | ((dp<.004)&(ratio<.18))
    pin_f = (ratio>=np.quantile(ratio,.96)) | ((df<.004)&(ratio>.82))
    pin_f &= ~pin_p
    if not pin_p.any() or not pin_f.any():
        raise ValueError(f'Missing attachment patch in {names}')
    graph = graph_edges(x,t)
    gp,gf = geodesic(graph,pin_p),geodesic(graph,pin_f)
    w = ratio.copy()
    valid = np.isfinite(gp)&np.isfinite(gf)
    w[valid] = gp[valid]/(gp[valid]+gf[valid])
    # Disconnected anatomical fragments with only one or neither pinned end
    # inherit the nearest connected anatomical field in 3D.
    if valid.any() and (~valid).any():
        d,ids=cKDTree(x[valid]).query(x[~valid],k=min(4,int(valid.sum())))
        if d.ndim==1: d,ids=d[:,None],ids[:,None]
        coeff=1/np.maximum(d,1e-6)**2
        w[~valid]=np.sum(coeff*w[valid][ids],axis=1)/coeff.sum(axis=1)
    # Unite the two faces of thin tissue while preserving its geodesic trend.
    # A smooth distance factor pins both attachment patches exactly.
    w = w*w*(3-2*w)
    dist_p=cKDTree(x[pin_p]).query(x)[0]
    dist_f=cKDTree(x[pin_f]).query(x)[0]
    scale=np.clip(.12*min(np.median(dist_p[pin_f]),np.median(dist_f[pin_p])),.0025,.012)
    w=local_field(x,w,x,scale)
    ap=1-np.exp(-.5*(dist_p/scale)**2)
    af=1-np.exp(-.5*(dist_f/scale)**2)
    w=(ap*af*w+ap*(1-af))/(ap+af-ap*af)
    w[pin_p],w[pin_f] = 0.,1.
    return w,pin_p,pin_f,graph,dict(proximalTarget=prox,distalTarget=dist,
        proximalSeeds=int(pin_p.sum()),distalSeeds=int(pin_f.sum()),
        bothEndsConnectedVertices=int(valid.sum()),components=int(connected_components(graph)[0]))


def representative_path(x,graph,a,b):
    # Mid-patch to mid-patch geodesic in one connected surface component.
    components = connected_components(graph)[1]
    common = np.intersect1d(components[a],components[b])
    if not len(common):
        return []
    comp = max(common,key=lambda c:np.count_nonzero(components==c))
    def middle(mask):
        ids=np.flatnonzero(mask & (components==comp))
        return int(ids[np.argmin(np.linalg.norm(x[ids]-x[ids].mean(axis=0),axis=1))])
    start,end = middle(a),middle(b)
    _,prev = dijkstra(graph,indices=start,return_predecessors=True)
    path=[end]
    while path[-1]!=start:
        path.append(int(prev[path[-1]]))
    return path[::-1]


def bind_tensor_to_tract(x, graph, geodesic_weights, proximal, distal, band, faces, band_weights):
    """Attach the tensor fasciae latae insertion to the tract carrying it."""
    host = Trimesh(band, faces, process=False)
    # Exact closest surface points. Small batches bound temporary triangle arrays.
    samples = [closest_point_naive(host, x[i:i + 32]) for i in range(0, len(x), 32)]
    closest, distance, face = [np.concatenate(values) for values in zip(*samples)]
    ids = faces[face]
    bary = points_to_barycentric(band[ids], closest)
    host_w = (bary * band_weights[ids]).sum(axis=1)
    # Retain the muscle's own geodesic progression, scaled to the distal tract.
    prior = geodesic_weights * host_w[distal].mean()
    # Only the immediately adjacent investing lamina contributes soft constraints.
    fade = np.clip((distance - .001) / .002, 0., 1.)
    contact = 4. * (1. - fade * fade * (3. - 2. * fade))
    # Smooth the departure from the geodesic prior across connected muscle tissue.
    conductance = graph.copy()
    conductance.data = 1. / np.maximum(conductance.data, .0005)
    conductance.data /= np.median(conductance.data)
    smooth = laplacian(conductance)
    matrix = (smooth + diags(contact + .25)).tocsr()
    rhs = smooth @ prior + contact * host_w + .25 * prior
    weights = np.zeros_like(prior)
    fixed = proximal | distal
    weights[distal] = host_w[distal]
    free = ~fixed
    weights[free] = spsolve(matrix[free][:, free], (rhs - matrix @ weights)[free])
    # Validate against the physical host triangle (including the neutral tissue
    # separation) rather than a rigid rotation of the entire thigh.
    insertion_host = (band[ids[distal]], band_weights[ids[distal]], bary[distal],
                      x[distal] - closest[distal], host_w[distal])
    detail = dict(
        distalTarget='Iliotibial tract', mode='tract insertion and surface geodesic',
        distalHostRestDistanceMm=np.round(1000 * np.array([distance[distal].min(),distance[distal].max()]), 4).tolist(),
        distalHostWeightRange=np.round([host_w[distal].min(),host_w[distal].max()], 6).tolist(),
        laminaSoftCouplingVertices=int(np.count_nonzero((contact > 0) & free)),
        couplingMethod='Hold proximal anchors at the pelvis. Assign distal anchors the barycentric weights of the nearest tract triangles. Retain a scaled geodesic muscle prior, with soft fascial coupling tapered between 1 and 3 mm and a graph-smoothed correction. The spatial scale and relative penalties are modeling assumptions.',
        anatomyReference='https://doi.org/10.1002/ca.22248',
    )
    return np.clip(weights, 0., 1.), insertion_host, detail


def tract_insertion_pose(host, center, pose):
    points, weights, bary, separation, local_weight = host
    posed = deformation((points - center).reshape(-1, 3), weights.ravel(), pose)
    return (bary[:, :, None] * posed.reshape(points.shape)).sum(axis=1) + deformation(separation, local_weight, pose)


def deformation(x,w,pose):
    if not np.any(pose):
        return x.copy()
    flex,add,twist = pose
    vector = Rotation.from_euler('ZXY',[add,-flex,twist],degrees=True).as_rotvec()
    return Rotation.from_rotvec(w[:,None]*vector).apply(x)


def validate(x,t,center,w,path,pin_p,pin_f,replays,insertion_host=None):
    p=x-center
    areas = np.linalg.norm(np.cross(p[t[:,1]]-p[t[:,0]],p[t[:,2]]-p[t[:,0]]),axis=1)
    use = areas > 1e-9
    ir,fr = np.zeros(2),np.zeros(2)
    lengths=[]; area_lows=[]; area_highs=[]
    for pose in POSES:
        q = deformation(p,w,pose)
        rigid = deformation(p,np.ones(len(p)),pose)
        if len(path):
            lengths.append(round(float(np.linalg.norm(np.diff(q[path],axis=0),axis=1).sum()*1000),3))
        distal_expected = rigid[pin_f] if insertion_host is None else tract_insertion_pose(insertion_host,center,pose)
        for mask,truth,metric in [(pin_p,p[pin_p],ir),(pin_f,distal_expected,fr)]:
            if mask.any(): metric[0]=max(metric[0],np.linalg.norm(q[mask]-truth,axis=1).max())
        if use.any():
            after = np.linalg.norm(np.cross(q[t[:,1]]-q[t[:,0]],q[t[:,2]]-q[t[:,0]]),axis=1)
            area_lows.append(np.quantile(after[use]/areas[use],.01))
            area_highs.append(np.quantile(after[use]/areas[use],.99))
    ranges={}
    if len(path):
        for clip,angles in replays.items():
            axis=Rotation.from_euler('ZXY',angles[:,[1,0,2]]*[1,-1,1],degrees=True).as_rotvec()
            rotations=Rotation.from_rotvec((axis[:,None,:]*w[None,path,None]).reshape(-1,3))
            coords=rotations.apply(np.tile(p[path],(len(angles),1))).reshape(len(angles),len(path),3)
            mm=1000*np.linalg.norm(np.diff(coords,axis=1),axis=2).sum(axis=1)
            ranges[clip]=np.round([mm.min(),mm.max()],3).tolist()
    result = dict(neutralMaxDisplacementMm=float(np.linalg.norm(deformation(p,w,POSES[0])-p,axis=1).max())*1000,
                proximalMaxSlipMm=float(ir[0])*1000 if pin_p.any() else None,
                distalMaxSlipMm=float(fr[0])*1000 if pin_f.any() else None,
                replayEdgePathRangeMm=ranges,
                edgePathLengthsMm=lengths,areaRatioWorstPoseP01=round(float(min(area_lows,default=1)),4),
                areaRatioWorstPoseP99=round(float(max(area_highs,default=1)),4))
    if insertion_host is not None:
        edges = np.unique(np.sort(np.concatenate((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])),axis=1),axis=0)
        a,b = edges.T
        rest_lengths = np.linalg.norm(p[a]-p[b],axis=1)
        checks = {}
        for name,poses in replays.items():
            slip, relative_change = [], []
            for pose in poses:
                q = deformation(p,w,pose)
                expected = tract_insertion_pose(insertion_host,center,pose)
                slip.append(1000*np.linalg.norm(q[pin_f]-expected,axis=1).max())
                change = np.abs(np.linalg.norm(q[a]-q[b],axis=1)/rest_lengths-1.)
                relative_change.append(np.quantile(change,[.95,.99,1.]))
            checks[name] = dict(frames=len(poses),maxTractInsertionSlipMm=round(float(max(slip)),4),
                                maxWithinFrameAbsEdgeChangeP95P99Max=np.round(np.max(relative_change,axis=0),4).tolist())
        result['tractInsertionReplays'] = checks
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas',type=Path,default=ATLAS)
    parser.add_argument('--output',type=Path,help='Directory for hip-attachments.bin and hip-deformation.json')
    parser.add_argument('--uncoupled-tfl',action='store_true',
                        help='Reproduce the previous distal-thigh TFL proxy for comparison. The default output is data/uncoupled-tfl.')
    args=parser.parse_args()
    meshes,center,atlas_hash=load_meshes(args.atlas)
    refs=AttachmentSurfaces(meshes,center)
    clips=json.loads((DATA/'motion.json').read_text())['clips']
    replays={c['id']:np.array([[f.get(k,0) for k in ('hipFlexionDeg','hipAdductionDeg','hipRotationDeg')]
                               for f in c['frames']]) for c in clips}
    records,fields={},{}
    binary=bytearray()
    modes={name:strategy(name,group) for name,(_,_,group) in meshes.items()}
    # The TFL ends on the iliotibial tract above the femoral section. Build its
    # host first while retaining the original ordering of the serialized fields.
    band_name = 'Iliotibial tract'
    band_key,band_faces,_ = meshes[band_name]
    band = band_key/UNIT
    band_binding = bind_surface(band,band_faces,refs,modes[band_name])
    band_weights = np.rint(np.clip(band_binding[0],0.,1.)*65535)/65535.
    for name in sorted(meshes,key=lambda k:modes[k]=='transfer'):
        key,t,_=meshes[name];x=key/UNIT;mode=modes[name]
        insertion=None
        insertion_host=None
        pin_p,pin_f=np.zeros(len(key),bool),np.zeros(len(key),bool)
        path=[]
        if mode=='pelvis' or mode=='femur':
            value=int(mode=='femur')
            records[name]=dict(mode=mode,constant=value,vertices=len(x))
            continue
        if mode=='transfer':
            label=name.lower()
            if 'common tendon' in label:
                donors=['Semitendinosus muscle','Long head of biceps femoris']
            elif 'zona orbicularis' in label or 'neck vessels' in label:
                donors=['Hip joint capsule']
            elif 'obturator' in label:
                donors=['Obturator internus']
            elif 'piriformis' in label:
                donors=['Piriformis muscle']
            elif 'iliacus' in label or 'iliopectineal' in label:
                donors=['Iliacus muscle','Psoas major']
            elif 'gluteal' in label or 'gluteus' in label:
                donors=['Gluteus medius muscle','Gluteus minimus muscle','Gluteus maximus muscle']
            elif any(k in label for k in ['iliopect','iliopsoas','femoral','anterior']):
                donors=['Iliacus muscle','Psoas major','Rectus femoris','Pectineus muscle']
            else:
                donors=['Gluteus maximus muscle','Obturator internus','Piriformis muscle',
                        'Quadratus femoris muscle','Pectineus muscle','Iliacus muscle','Tensor fasciae latae']
            donors=[k for k in donors if k in fields]
            if not donors: raise ValueError('Missing host tissue for '+name)
            xyz=np.concatenate([fields[k][0] for k in donors]); values=np.concatenate([fields[k][1] for k in donors])
            d=cKDTree(xyz).query(x)[0]
            w=local_field(xyz,values,x,.012)
            meta=dict(mode='adjacent tissue interpolation',donors=donors,medianHostDistanceMm=round(float(np.median(d)*1000),2))
        else:
            w,pin_p,pin_f,graph,meta=band_binding if name==band_name else bind_surface(x,t,refs,mode)
            path=representative_path(x,graph,pin_p,pin_f)
            meta['mode']='surface geodesic'
            if name=='Tensor fasciae latae' and not args.uncoupled_tfl:
                w,insertion_host,detail=bind_tensor_to_tract(x,graph,w,pin_p,pin_f,band,band_faces,band_weights)
                meta.update(detail)
            if name in ('Gluteus medius muscle','Gluteus minimus muscle'):
                bone_distance=refs.distance('trochanter',x)
                seed=pin_f & (bone_distance<=np.quantile(bone_distance[pin_f],.25))
                distance_mm=1000*geodesic(graph,seed)
                if not np.isfinite(distance_mm).all() or distance_mm.max()>=655.35:
                    raise ValueError('Invalid gluteal insertion mask distance')
                insertion=np.rint(100*distance_mm).astype('<u2')
                meta.update(insertionSeeds=int(seed.sum()),recordBytes=16,
                            insertionGeodesicMmRange=np.round([distance_mm.min(),distance_mm.max()],3).tolist())
        quantized=np.rint(np.clip(w,0,1)*65535).astype('<u2')
        w=quantized.astype(float)/65535
        fields[name]=(x,w)
        offset=len(binary)
        for row,(i,j,k,v) in enumerate(zip(*key.T,quantized)):
            binary.extend(struct.pack('<iiiH',i,j,k,v))
            if insertion is not None:
                binary.extend(struct.pack('<H',insertion[row]))
        records[name]=dict(meta,vertices=len(x),offsetBytes=offset,validation=validate(x,t,center,w,path,pin_p,pin_f,replays,insertion_host))
    out=args.output or (DATA/'uncoupled-tfl' if args.uncoupled_tfl else DATA)
    out.mkdir(parents=True,exist_ok=True)
    (out/'hip-attachments.bin').write_bytes(binary)
    manifest=dict(version=2,unitsPerMetre=UNIT,recordBytes=14,sourceAtlasSha256=atlas_hash,
        weightsSha256=hashlib.sha256(binary).hexdigest(),surfaces=records,poseDegrees=POSES,
        replayFrames={k:len(v) for k,v in replays.items()},replayMotionSha256=hashlib.sha256((DATA/'motion.json').read_bytes()).hexdigest(),
        insertionRegion=dict(distanceUnitMm=.01,fullThroughMm=5,endsAtMm=35,
            method='For each of the two gluteal muscles, the nearest quarter of the inferred femur pins to the trochanter surface forms an illustrative insertion footprint. Measure shortest distance along atlas edges from that footprint. Cubic easing from 5 to 35 mm draws the distal region. These widths are display assumptions, and tendon was not independently segmented.'),
        poseConvention='[flexion, adduction, internal rotation]; rotate about fitted hip centre using ZXY(add,-flex,rotation).',
        method='Proximity to anatomically selected bone surfaces sampled at ≤2 mm spacing or to atlas section patches defines approximate anchors. Blended tissues use mesh-geodesic distance between anchor patches with cubic easing and normalized spatial convolution; a smooth distance constraint fixes both anchor patches. Vertex positions move on a spherical arc about the fitted hip centre. Nearby fascial/neurovascular surfaces can inherit neighboring tissue motion. The truncated distal thigh follows the femur. Lumbar attachments travel with the pelvis.',
        interpretation='Kinematic illustration. Pins are inferred from a neutral dissection. The computed surface-edge lengths and area ratios check mesh behavior; they do not measure individual muscle fibers, tendon strain, pressure, contact, or inflammation.')
    (out/'hip-deformation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Surfaces',len(records),'fields',len(fields),'bytes',len(binary),'sha256',manifest['weightsSha256'])
    for name,s in records.items():
        if s.get('validation',{}).get('edgePathLengthsMm'):
            print(name,s['proximalSeeds'],s['distalSeeds'],s['validation']['edgePathLengthsMm'])


if __name__=='__main__':
    main()
