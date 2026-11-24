"""Run the complete leverage-offset/participant-bootstrap study on an Apple GPU.

Requires macOS 15+, Xcode command-line tools and the NumPy + Modal environment
used by compute/bootstrap.py. This driver never submits a cloud job.
Example: python hip-study/metal/sensitivity.py
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import benchmark

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'compute'))
import bootstrap
import modal_benchmark
import study


def digest(a):
    return hashlib.sha256(a).hexdigest()


def pilot_weights():
    rows = []
    for i in range(6):
        for j in range(6):
            for k in (1, 3, 6):
                row = np.zeros(6, dtype=np.uint32)
                row[i] += k
                row[j] += 6 - k
                rows.append(row)
    rows.append(np.ones(6, dtype=np.uint32))
    return np.unique(rows, axis=0)


def cpu_reference(args, frames, weights, offsets, scratch):
    if args.reference:
        ref = np.load(args.reference, allow_pickle=False)
    else:
        cpu, _, _ = study.compile_cpu(scratch)
        raw, par = study.sample(args.count,args.seed)
        del raw
        ref = np.stack([
            bootstrap.cpu_bootstrap(cpu, [modal_benchmark.offset_moment(x,m) for x in frames],
                                    par,os.cpu_count() or 1,weights) for m in offsets
        ])
    if ref.shape != (len(offsets),len(weights)) or not np.issubdtype(ref.dtype,np.integer):
        raise RuntimeError('Invalid reference matrix')
    return np.ascontiguousarray(ref,dtype='<i8')


def metal_runner(tmp):
    libfile = tmp / 'sweep.dylib'
    cmd = ['xcrun','clang++','-O3','-std=c++17','-fobjc-arc','-dynamiclib','-framework','Metal',
           '-framework','Foundation',str(ROOT/'metal/sensitivity.mm'),'-o',str(libfile)]
    start = time.perf_counter()
    subprocess.run(cmd,check=True)
    host_compile = time.perf_counter()-start
    lib = ctypes.CDLL(str(libfile))
    source = benchmark.metal_source(study.SOURCE.read_text()) + (ROOT/'metal/sensitivity.metal').read_text()
    lib.setup_sweep.argtypes = [ctypes.c_char_p]
    lib.setup_sweep.restype = ctypes.c_char_p
    start = time.perf_counter()
    device = lib.setup_sweep(source.encode())
    if not device:
        raise RuntimeError('Failed to initialize Metal')
    pipeline_compile = time.perf_counter()-start
    f32 = np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    u32 = np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    f64 = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    u = ctypes.c_uint
    lib.run_sweep.argtypes = [f32,u,f32,u32,u,u32,u,u32,u,u32,u32,f64]
    lib.run_sweep.restype = ctypes.c_int
    return lib,device.decode(),host_compile,pipeline_compile,digest(source.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,help='Use an independently regenerated CPU ID matrix (.npy).')
    parser.add_argument('--count',type=int,default=1048576)
    parser.add_argument('--seed',type=int,default=20261122)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--exhaustive-repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'data/benchmark-metal-sensitivity.json')
    args = parser.parse_args()
    if args.repeats < 1 or args.exhaustive_repeats < 0 or args.count < 1:
        raise ValueError('Expected a positive population and repeat count')
    trials = json.loads((study.DATA/'implants.json').read_text())['trials']
    training = [t for t in trials if t['split']=='calibration']
    subjects = sorted({t['subject'] for t in training})
    if len(subjects)!=6 or len(set(subjects)&{t['subject'] for t in trials if t['split']=='evaluation'}):
        raise ValueError('Expected six independent calibration participants')
    geo = json.loads((study.DATA/'geometry.json').read_text())
    xs = {t['id']:study.encode(t,geo) for t in training}
    frames = [study.combine([t for t in training if t['subject']==s],xs) for s in subjects]
    weights,_ = bootstrap.distributions(6)
    offsets = np.linspace(-15,15,13)
    counts = weights.astype(np.uint32)
    lengths = np.array(list(map(len,frames)),np.uint32)
    pilot = pilot_weights()
    paths = ['compute/search.cu','compute/study.py','compute/bootstrap.py','compute/modal_benchmark.py',
             'metal/benchmark.py','metal/sensitivity.py','metal/sensitivity.metal','metal/sensitivity.mm',
             'data/geometry.json','data/implants.json']
    result = dict(platform=platform.platform(),numpy=np.__version__,subjects=subjects,
                  candidates=args.count,offsetsMm=offsets.tolist(),resamples=len(counts),
                  countVectorsSha256=digest(counts.tobytes()),frameCount=int(sum(map(len,frames))),
                  seed=args.seed,threadgroupSize=128,anchorPilotSize=min(args.count,4096),
                  anchorMixtures=len(pilot),sourceSha256={p:digest((ROOT/p).read_bytes()) for p in paths})
    with tempfile.TemporaryDirectory() as d:
        scratch = Path(d)
        reference = cpu_reference(args,frames,weights,offsets,scratch)
        result['referenceWinnersInt64LeSha256'] = digest(reference.tobytes())
        cold_start = time.perf_counter()
        lib,device,host_build,metal_build,msl = metal_runner(scratch)
        result.update(device=device,compileHostSeconds=host_build,initializeMetalSeconds=metal_build,
                      effectiveMslSha256=msl)
        modes = [('pareto',len(pilot),args.repeats),('exhaustive',0,args.exhaustive_repeats)]
        for mode,anchors,repeats in modes:
            if not repeats:
                continue
            timings,device_times,dispatch_times = [],[],[]
            for rep in range(repeats+1):
                start = time.perf_counter()
                raw,par = study.sample(args.count,args.seed)
                del raw
                features = np.concatenate([modal_benchmark.offset_moment(x,m) for m in offsets for x in frames])
                winners = np.empty(reference.shape,np.uint32)
                kept = np.empty(len(offsets),np.uint32)
                timer = np.empty(2,np.float64)
                status = lib.run_sweep(par,args.count,features,lengths,len(offsets),counts,len(counts),
                                       pilot,anchors,winners,kept,timer)
                elapsed = time.perf_counter()-start
                if mode=='pareto' and rep==0:
                    result['firstLocalSeconds'] = time.perf_counter()-cold_start
                if status:
                    raise RuntimeError(f'Native GPU status {status}')
                if not np.array_equal(winners,reference):
                    idx = np.argwhere(winners!=reference)
                    failures = [(i.tolist(),int(winners[tuple(i)]),int(reference[tuple(i)])) for i in idx[:10]]
                    raise RuntimeError(f'{len(idx)} discrepant winners; first differences {failures}')
                if rep==0:
                    result['parameterSha256'] = digest(par.tobytes())
                    first_execution = elapsed
                else:
                    timings.append(elapsed)
                    device_times.append(float(timer[0]))
                    dispatch_times.append(float(timer[1]))
                print(mode,rep,'wall seconds',round(elapsed,4),'device',round(timer[0],4),
                      'IDs matched',winners.size,'retained range',int(kept.min()),int(kept.max()),flush=True)
            result[mode] = dict(
                warmEndToEndSamplesSeconds=timings,warmEndToEndSeconds=statistics.median(timings),
                deviceSamplesSeconds=device_times,deviceSeconds=statistics.median(device_times),
                dispatchAndWaitSamplesSeconds=dispatch_times,dispatchAndWaitSeconds=statistics.median(dispatch_times),
                firstExecutionSeconds=first_execution,candidatesRetainedPerOffset=kept.tolist(),
                winnerAgreement=int(winners.size),
                weightedFitsEvaluated=int(sum(map(int,kept))*len(counts)),
                allCandidateOffsetResamples=args.count*len(offsets)*len(counts))
    result['method'] = (
        'Score all anatomies in six participants for each assumed lever offset. Pareto mode drops an '
        'anatomy only if a pilot anatomy has strictly lower losses in all six participants. Every '
        'resample has nonnegative weights summing to six, hence every discarded candidate has a strictly '
        'worse fit for every resample. Exhaustive mode retains all candidates. Both modes take exact '
        'weighted sums on a verified 2^-34 fixed-point lattice and resolve ties by smallest candidate ID. '
        'A runtime check verifies every participant loss converts exactly and every sum fits in the '
        '53-bit significand of the CPU/H100 FP64 reference. All returned IDs were independently checked.')
    result['timingScope'] = (
        'Warm wall starts before regenerating the population and applying offsets to preprocessed frames. '
        'Includes all temporary allocations, transfers into Metal buffers, encoding, execution, synchronization '
        'and copying 6,006 winning indices back into NumPy. One untimed execution per mode. Warm runs reuse '
        'the compiled pipelines. The first local time includes native host compilation, library loading, '
        'Metal initialization, pipeline compilation and the first full Pareto execution. Python startup, '
        'clinical feature preparation, bootstrap enumeration, CPU oracle, checks and reporting are outside '
        'all timers. OS shader cache state is uncontrolled. Device time uses Metal command-buffer timestamps.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
