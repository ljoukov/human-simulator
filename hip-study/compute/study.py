"""Frontal-plane hip load reconstruction and participant-level validation.

Run directly with Python 3, NumPy, and a C++ compiler. The computation uses
OrthoLoad implant force only in the training loss and in final evaluation.
"""
import argparse
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
SOURCE = Path(__file__).with_name('search.cu')
PARAMETERS = [
    ('glutealArmScale', 0.72, 1.32),
    ('itbArmM', 0.025, 0.060),
    ('adductorArmM', 0.020, 0.050),
    ('itbMomentShare', 0.03, 0.35),
    ('coactivationBwPerGrfBw', 0.0, 0.35),
    ('glutealLineTiltDeg', 5.0, 32.0),
    ('neutralWrapHalfAngleDeg', 4.0, 22.0),
    ('wrapSlope', 0.20, 0.70),
]


def dump(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')


def sample(count, seed):
    # Independent uniform scenario ranges, stratified per axis. No disease priors.
    rng = np.random.default_rng(seed)
    raw = np.empty((count, len(PARAMETERS)), dtype=np.float64)
    for k, (_, low, high) in enumerate(PARAMETERS):
        raw[:, k] = low + (high-low)*(rng.permutation(count)+rng.random(count))/count
    arm_scale, itb_arm, add_arm, share, coact, tilt, wrap, slope = raw.T
    packed = np.column_stack(((1-share)/arm_scale, share/itb_arm, add_arm, 1/add_arm,
                              coact, np.sin(np.radians(tilt)), np.cos(np.radians(tilt)),
                              np.sin(np.radians(wrap)), slope)).astype(np.float32)
    return raw, packed


def interpolate(flexion, adduction, geometry, field_values):
    # Fixed 0-degree internal rotation since marker-derived axial rotation is unavailable.
    axes = [np.asarray(geometry['axes'][k], dtype=float) for k in ('flexionDeg','adductionDeg','rotationDeg')]
    field = np.asarray(field_values, dtype=float).reshape(tuple(map(len, axes)))
    if not np.any(axes[2] == 0.0):
        raise ValueError('Expected a neutral axial rotation slice.')
    field = field[:, :, np.flatnonzero(axes[2]==0.0)[0]]
    idx, fraction = [], []
    for axis, value in zip(axes[:2], [flexion, adduction]):
        value = np.clip(value, axis[0], axis[-1])
        i = np.clip(np.searchsorted(axis,value,side='right')-1,0,len(axis)-2)
        idx.append(i)
        fraction.append((value-axis[i])/(axis[i+1]-axis[i]))
    i,j = idx
    a,b = fraction
    result = (1-a)*((1-b)*field[i,j]+b*field[i,j+1]) + a*((1-b)*field[i+1,j]+b*field[i+1,j+1])
    if not np.all(np.isfinite(result)):
        raise ValueError('Non-finite OpenSim lookup value.')
    return result


def encode(trial, geometry):
    frames = trial['frames']
    def channel(key):
        return np.array([f[key] for f in frames], dtype=np.float64)
    m, add = channel('adductionMomentBwM'), channel('hipAdductionDeg')
    arm = interpolate(channel('hipFlexionDeg'), add, geometry, geometry['effectiveAbductionArmM'])
    if np.any(arm<=0):
        raise ValueError('Invalid abductor lever arm.')
    x = np.column_stack((channel('grfBw'),channel('horizontalBw'),np.maximum(m,0),
                         np.maximum(-m,0),1/arm,channel('measuredBw'),
                         np.full(len(frames),1/len(frames)),np.sin(np.radians(add))))
    if not np.all(np.isfinite(x)) or np.min(x[:,0]) < 0:
        raise ValueError('Invalid trial inputs: '+trial['id'])
    return x.astype(np.float32)


def combine(trials, xs):
    subjects = {t['subject'] for t in trials}
    rows = []
    for t in trials:
        x = xs[t['id']].copy()
        n_trials = sum(v['subject']==t['subject'] for v in trials)
        x[:,6] /= len(subjects)*n_trials
        rows.append(x)
    return np.concatenate(rows)


def simulate(x, parameters, itb_coefficient=None):
    """Independent float64 reference and branch-resolved internal forces, all in BW."""
    v,h,positive,negative,inv_arm,_,_,sin_add = x.astype(np.float64).T
    coeff,band_coeff,add_arm,inv_add,coact,tilt_sin,tilt_cos,neutral,slope = parameters.astype(np.float64).T[:,:,None]
    co = coact*v
    moment = positive + add_arm*co
    tendon = moment*inv_arm*coeff
    band = moment*band_coeff
    add = negative*inv_add+co
    reaction = np.hypot(v+tendon*tilt_cos+band+add*0.85, h-tendon*tilt_sin-add*0.52678269)
    # Equivalent cable against a smooth guide, resultant normal force = 2T sin(phi/2).
    # The pose-dependent bend is a scenario assumption. This is a regional load
    # proxy in BW, not pressure or a measurement of either individual tendon.
    compression = (band*itb_coefficient if itb_coefficient is not None else
                   2*band*np.clip(neutral+slope*sin_add, 0, 1))
    return reaction, tendon, compression


def compile_cpu(scratch):
    library = scratch / ('libhip.dylib' if platform.system()=='Darwin' else 'libhip.so')
    compiler = os.environ.get('CXX', 'clang++')
    cmd = [compiler,'-x','c++','-std=c++17','-O3','-march=native','-ffp-contract=off',
           '-fno-math-errno','-shared','-fPIC','-pthread',str(SOURCE),'-o',str(library)]
    subprocess.run(cmd,check=True)
    fun = ctypes.CDLL(str(library)).hip_cpu
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    fun.argtypes = [floats,floats,floats,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    fun.restype = None
    return fun, ' '.join(cmd[:9]), subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]


def score(fun,x,p,threads):
    out = np.empty(len(p),dtype=np.float32)
    fun(x,p,out,len(x),len(p),threads)
    return out


def rmse(a,b):
    return float(np.sqrt(np.mean((a-b)**2)))


def baseline_design(x):
    return np.column_stack((np.ones(len(x)),x[:,0],x[:,2]+x[:,3]))


def fit_baseline(x):
    w = np.sqrt(x[:,6])
    return np.linalg.lstsq(baseline_design(x)*w[:,None], x[:,5]*w, rcond=None)[0]


def avg_subject(trials, field):
    subjects = sorted({t['subject'] for t in trials})
    return float(np.mean([np.mean([t[field] for t in trials if t['subject']==s]) for s in subjects]))


def measure(t,x,p,beta,store_frames,itb_coefficient=None):
    contact,tension,compression = simulate(x,p,itb_coefficient)
    pred = np.median(contact,axis=0)
    base = baseline_design(x) @ beta
    lo,hi = np.quantile(contact,[.05,.95],axis=0)
    y = x[:,5]
    result = {k:t[k] for k in ('id','subject','split','activity','durationS','side')}
    result.update(rmseBw=rmse(pred,y),baselineRmseBw=rmse(base,y),
                  verticalOnlyRmseBw=rmse(x[:,0],y),
                  coverage90=float(np.mean((lo<=y)&(y<=hi))))
    if store_frames:
        channels = [y,pred,base,lo,hi,np.median(tension,axis=0),np.median(compression,axis=0)]
        names = ['measuredBw','predictedBw','baselineBw','loBw','hiBw','tendonBw','compressionBw']
        result['frames'] = [dict(phase=f['phase'],**{k:round(float(a[i]),5) for k,a in zip(names,channels)})
                            for i,f in enumerate(t['frames'])]
    return result


def ambiguity(trial,x,indices,p,raw,itb_coefficient=None):
    contact,tendon,compression = simulate(x,p[indices],itb_coefficient)
    # Prespecified signal: largest interface load gap at peak vertical GRF
    # with contact RMS discrepancy <= 0.05 BW over the entire sampled phase.
    frame = int(np.argmax(x[:,0]))
    squared = np.mean(contact**2,axis=1)
    distance = np.sqrt(np.maximum(0,squared[:,None]+squared[None,:]-2*(contact@contact.T)/len(x)))
    gap = compression[:,frame][None,:]-compression[:,frame][:,None]
    allowed = np.triu(np.ones_like(distance,dtype=bool),1)&(distance<=0.05)
    if not np.any(allowed):
        return {'trialId':trial['id'],'available':False,'reason':'No distinct retained pair within 0.05 BW RMS.'}
    a,b = np.unravel_index(np.argmax(np.where(allowed,np.abs(gap),-1)),gap.shape)
    if gap[a,b]<0: a,b=b,a
    def scenario(i):
        return dict(candidateId=int(indices[i]),parameters={desc[0]:round(float(v),5) for desc,v in zip(PARAMETERS,raw[indices[i]])},
                    frames=[{'phase':f['phase'],'contactBw':round(float(contact[i,j]),5),
                             'tendonBw':round(float(tendon[i,j]),5),'compressionBw':round(float(compression[i,j]),5)}
                            for j,f in enumerate(trial['frames'])])
    return dict(available=True,trialId=trial['id'],frameIndex=frame,phase=trial['frames'][frame]['phase'],
                contactDifferenceBw=float(abs(contact[a,frame]-contact[b,frame])),contactRmsDifferenceBw=float(distance[a,b]),
                compressionDifferenceBw=float(abs(gap[a,b])),tendonDifferenceBw=float(abs(tendon[a,frame]-tendon[b,frame])),
                low=scenario(a),high=scenario(b))


def convergence(trial, x, scores, p, coefficient):
    """Nested prefix search under fixed tolerances and a common reference waveform."""
    best = int(np.argmin(scores))
    anchor = simulate(x,p[[best]],coefficient)[0][0]
    peak = int(np.argmax(x[:,0]))
    threshold = math.sqrt(float(scores[best]))+0.02
    rows = []
    for count in sorted({n for n in [4096,65536,262144,len(scores)] if n<=len(scores)}):
        eligible = np.flatnonzero(scores[:count]<=threshold**2)
        tendon_values,compression_values = [],[]
        for first in range(0,len(eligible),1024):
            contact,tendon,compression = simulate(x,p[eligible[first:first+1024]],coefficient)
            ok = np.sqrt(np.mean((contact-anchor)**2,axis=1))<=0.05
            tendon_values.extend(tendon[ok,peak])
            compression_values.extend(compression[ok,peak])
        def extent(values):
            return [float(min(values)),float(max(values))] if values else None
        rows.append(dict(candidates=count,fitWithinTolerance=len(eligible),
            matches=len(tendon_values),bestCalibrationRmseBw=math.sqrt(float(np.min(scores[:count]))),
            compressionRangeBw=extent(compression_values),tendonRangeBw=extent(tendon_values)))
    return dict(trialId=trial['id'],frameIndex=peak,anchorCandidate=best,
                calibrationRmseLimitBw=threshold,contactRmsLimitBw=0.05,rows=rows)


def hardware():
    if platform.system()=='Darwin':
        try:
            return subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],stderr=subprocess.DEVNULL,text=True).strip()
        except subprocess.SubprocessError:
            pass
    return os.environ.get('HIP_CPU_NAME') or platform.processor() or platform.machine()


def file_hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(rows):
    # Participant-macro mean of trial RMSEs (each subject gets equal weight).
    return {key:avg_subject(rows,name) for key,name in (
        ('modelRmseBw','rmseBw'),('baselineRmseBw','baselineRmseBw'),
        ('verticalOnlyRmseBw','verticalOnlyRmseBw'),('coverage90','coverage90'))}


def run(args):
    args.work.mkdir(parents=True, exist_ok=True)
    geometry = json.loads((DATA/'geometry.json').read_text())
    trials = json.loads((DATA/'implants.json').read_text())['trials']
    train = [t for t in trials if t['split']=='calibration']
    test = [t for t in trials if t['split']=='evaluation']
    train_subjects = sorted({t['subject'] for t in train})
    test_subjects = sorted({t['subject'] for t in test})
    if (not train or not test or set(train_subjects)&set(test_subjects)
            or len({t['id'] for t in trials}) != len(trials)):
        raise ValueError('Expected distinct subjects in both splits and unique trial IDs.')
    xs = {t['id']:encode(t,geometry) for t in trials}
    xtrain = combine(train,xs)
    raw,p = sample(args.count,args.seed)
    cpu,flags,compiler = compile_cpu(args.work)
    threads = args.threads
    scores = score(cpu,xtrain,p,threads)
    np.savez(args.work/'benchmark-input.npz', frames=xtrain,parameters=p,cpuScores=scores)
    print('Benchmark NPZ ready:',args.work/'benchmark-input.npz',flush=True)
    ids = np.random.default_rng(args.seed+1).choice(len(p),size=min(256,len(p)),replace=False)
    contacts = simulate(xtrain,p[ids])[0]
    reference = np.sum((contacts-xtrain[:,5])**2*xtrain[:,6],axis=1)
    error = float(np.max(np.abs(reference-scores[ids])))
    if not np.allclose(reference,scores[ids],rtol=1e-5,atol=1e-6):
        raise ValueError('CPU SIMD and NumPy float64 reference disagree.')
    rank = np.argsort(scores,kind='stable')[:args.retain]
    beta = fit_baseline(xtrain)
    itb_fields = geometry.get('itb',{}).get('inwardCoefficient')
    if itb_fields is None:
        raise ValueError('The trochanteric ITB geometry is required. Run prepare_geometry first.')
    coeff = {t['id']: interpolate(np.array([f['hipFlexionDeg'] for f in t['frames']]),
                                  np.array([f['hipAdductionDeg'] for f in t['frames']]),
                                  geometry,itb_fields) for t in trials}
    rows = [measure(t,xs[t['id']],p[rank],beta,True,coeff[t['id']]) for t in trials]
    held = [r for r in rows if r['split']=='evaluation']
    result = dict(schemaVersion=1,
        summary=dict(calibrationSubjects=train_subjects,evaluationSubjects=test_subjects,
                     candidates=args.count,retained=args.retain,
                     **summarize(held)),
        scoreDefinition='Training participant-weighted mean squared error (BW²). Evaluation metrics are means of trial waveform RMSE within participant, then across participants.',
        trials=rows, parameters=[dict(name=name,min=lo,max=hi) for name,lo,hi in PARAMETERS],
        bestParameters=dict(zip([a[0] for a in PARAMETERS],raw[rank[0]].tolist())),
        baselineCoefficients=beta.tolist(),seed=args.seed,
        fitMseBw2=[float(scores[rank[0]]),float(scores[rank[-1]])],
        independentCheck=dict(count=len(ids),maxAbsMseErrorBw2=error),
        sourceHashes={name:file_hash(DATA/name) for name in ['geometry.json','implants.json']})
    eligible = sorted(test,key=lambda t:(t['activity']!='walk',t['subject'],t['id']))
    t = eligible[0]
    result['ambiguity'] = ambiguity(t,xs[t['id']],rank,p,raw,coeff[t['id']])
    result['cableSensitivity'] = ambiguity(t,xs[t['id']],rank,p,raw)
    result['geometryAudit'] = dict(model=geometry['model'],rotationDeg=0,
        postureClips=sum(1 for t in trials for f in t['frames'] if any(
            f[field]<min(geometry['axes'][axis]) or f[field]>max(geometry['axes'][axis])
            for field,axis in [('hipAdductionDeg','adductionDeg'),('hipFlexionDeg','flexionDeg')])),
        frames=sum(len(t['frames']) for t in trials))
    result['convergence'] = convergence(t,xs[t['id']],scores,p,coeff[t['id']])
    # Second analysis: each subject is unseen exactly once. These use the same
    # deterministic subset of the parameter ensemble in every outer fold.
    cv_n = min(args.count,65536)
    subjects = sorted({t['subject'] for t in trials})
    losses = np.stack([score(cpu,combine([t for t in trials if t['subject']==s],xs),p[:cv_n],threads)
                       for s in subjects])
    cv = []
    for k,s in enumerate(subjects):
        fit = [t for t in trials if t['subject']!=s]
        mask = np.arange(len(subjects))!=k
        chosen = np.argsort(np.mean(losses[mask],axis=0),kind='stable')[:args.retain]
        beta_cv = fit_baseline(combine(fit,xs))
        cv.extend(measure(t,xs[t['id']],p[chosen],beta_cv,False) for t in trials if t['subject']==s)
    alternatives = dict(geometry.get('comparisonEffectiveAbductionArmM',{}))
    alternatives[geometry['model']] = geometry['effectiveAbductionArmM']
    sensitivity = []
    for name,arms in alternatives.items():
        alternative = dict(geometry, effectiveAbductionArmM=arms)
        xalt = {t['id']:encode(t,alternative) for t in trials}
        loss = score(cpu,combine(train,xalt),p[:cv_n],threads)
        chosen = np.argsort(loss,kind='stable')[:args.retain]
        values = [measure(t,xalt[t['id']],p[chosen],beta,False) for t in test]
        sensitivity.append(dict(model=name,**summarize(values)))
    result['geometrySensitivity'] = dict(candidates=cv_n,retained=args.retain,rows=sensitivity)
    result['leaveOneSubjectOut'] = dict(candidates=cv_n,retained=args.retain,subjects=subjects,
                                        summary=summarize(cv),trials=cv)
    result['byActivity'] = {a:summarize([r for r in held if r['activity']==a]) for a in {r['activity'] for r in held}}
    dump(DATA/'results.json', result)
    print('Study:', result['summary'],flush=True)
    print('Ambiguous pair:', {k:v for k,v in result['ambiguity'].items() if k not in ('low','high')},flush=True)
    if args.science_only:
        return
    # Time the actual optimization input and all-candidate output. Retain every repetition.
    counts = sorted({n for n in [4096,65536,262144,args.count] if n<=args.count})
    timing = []
    for n in counts:
        score(cpu,xtrain,p[:n],threads)  # warm caches and instructions
        ms = []
        for _ in range(args.repeats):
            start = perf_counter()
            res = score(cpu,xtrain,p[:n],threads)
            ms.append((perf_counter()-start)*1000)
            if not np.array_equal(res,scores[:n]):
                raise RuntimeError('Repeated score mismatch.')
        timing.append(dict(candidates=n,frames=len(xtrain),cpuMs=float(np.median(ms)),cpuSamplesMs=ms))
        print('Laptop:', timing[-1],flush=True)
    payload = io.BytesIO()
    np.savez(payload,frames=xtrain,parameters=p)
    digest = hashlib.sha256(payload.getvalue()).hexdigest()
    dump(DATA/'benchmark-cpu.json',dict(payloadSha256=digest,cpuName=hardware(),cpuThreads=threads,
                    compiler=compiler,compilerFlags=flags,platform=platform.platform(),
                    floats='float32',referenceCases=len(ids),maxAbsMseErrorBw2=error,
                    kernelSha256=file_hash(SOURCE),rows=timing))


def merge():
    cpu = json.loads((DATA/'benchmark-cpu.json').read_text())
    gpu = json.loads((DATA/'benchmark-gpu.json').read_text())
    if gpu['payloadSha256'] != cpu['payloadSha256']:
        raise ValueError('CPU and GPU payload hashes differ.')
    if gpu['kernelSha256'] != cpu['kernelSha256']:
        raise ValueError('CPU and GPU kernel source hashes differ.')
    if not all(r['allClose'] for r in gpu['runs']):
        raise ValueError('GPU accuracy check failed.')
    cuda = {r['candidates']:r for r in gpu['runs']}
    for r in cpu['rows']:
        g = cuda[r['candidates']]
        if r['frames']!=g['frames']:
            raise ValueError('CPU and GPU used different calibration window sizes.')
        r.update(gpuMs=g['kernelMs'],speedup=r['cpuMs']/g['kernelMs'],
                 gpuSamplesMs=g['kernelSamplesMs'],maxAbsoluteError=g['maxAbsoluteError'],
                 topK=g['topK'],topKAgreement=g['topKAgreement'],
                 uploadAndAllocateMs=g['uploadAndAllocateMs'],downloadMs=g['downloadMs'])
    cpu.update(gpuName=gpu['hardware']['device'],gpu=gpu['hardware'],
               modalAppId=gpu['modalAppId'],created=gpu['created'],rpcWallSeconds=gpu['rpcWallSeconds'],
               singlePassRpcSeconds=gpu['singlePassRpcSeconds'],payloadSha256=gpu['payloadSha256'])
    if 'compact' in gpu:
        cpu['compact']=gpu['compact']
    dump(DATA/'benchmark.json',cpu)


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--count',type=int,default=1048576)
    ap.add_argument('--retain',type=int,default=512)
    ap.add_argument('--seed',type=int,default=20261122)
    ap.add_argument('--threads',type=int,default=os.cpu_count() or 1)
    ap.add_argument('--work',type=Path,default=Path(tempfile.gettempdir())/'hip-search-work')
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--merge',action='store_true')
    ap.add_argument('--science-only',action='store_true',help='Regenerate study results and the input bundle without replacing performance measurements.')
    args = ap.parse_args()
    if args.merge:
        merge()
    else:
        if args.count < args.retain or not 2<=args.retain<=65536 or min(args.threads,args.repeats)<1:
            ap.error('Invalid ensemble size, retained count, timing repetitions, or thread count.')
        run(args)
