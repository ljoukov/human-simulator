// The host prepends the shared subject objective and hip_score from search.cu.
constant uint lanes = 128;
constant uint subjects = 6;
constant uint tile = 1024;
constant uint anchorsMax = 64;
constant float quantumScale = 17179869184.0f;  // 2^34

struct Hit { ulong sum; uint id; };
inline Hit empty_hit() { return Hit{~ulong(0), ~uint(0)}; }
inline Hit better(Hit a, Hit b) {
    return (b.sum < a.sum || (a.sum == b.sum && b.id < a.id)) ? b : a;
}

// The screen kernel verifies every loss round-trips to this integer lattice,
// and every weighted sum is < 2^53, so the reference FP64 sums are exact.
inline Hit evaluate(device const float *losses, uint n, uint id, device const uint *w) {
    ulong total = 0;
    for (uint s = 0; s < subjects; ++s)
        total += ulong(losses[s*n+id] * quantumScale) * w[s];
    return Hit{total,id};
}
inline Hit group_minimum(threadgroup Hit *hits, uint lane, Hit best) {
    hits[lane] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes/2; stride > 0; stride /= 2) {
        if (lane < stride) hits[lane] = better(hits[lane], hits[lane+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return hits[0];
}

// Each anchor is an actual candidate, chosen for a different mixture of losses.
// The pilot subset affects screen efficiency only.
kernel void find_anchors(device const float *losses [[buffer(0)]],
                         device const uint *weights [[buffer(1)]],
                         device uint *anchors [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         uint lane [[thread_position_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
    threadgroup Hit hits[128];
    Hit best = empty_hit();
    for (uint id = lane; id < min(n,4096u); id += lanes)
        best = better(best,evaluate(losses,n,id,weights+subjects*group));
    best = group_minimum(hits,lane,best);
    if (lane == 0) anchors[group] = best.id;
}

kernel void screen(device const float *losses [[buffer(0)]],
                   device const uint *anchors [[buffer(1)]],
                   device uint *kept [[buffer(2)]],
                   device atomic_uint *count [[buffer(3)]],
                   device atomic_uint *error [[buffer(4)]],
                   constant uint &n [[buffer(5)]], constant uint &anchorCount [[buffer(6)]],
                   uint lane [[thread_position_in_threadgroup]],
                   uint id [[thread_position_in_grid]],
                   ushort simdLane [[thread_index_in_simdgroup]]) {
    threadgroup float anchorLoss[64][6];
    for (uint i = lane; i < anchorCount*subjects; i += lanes)
        anchorLoss[i/subjects][i%subjects] = losses[(i%subjects)*n+anchors[i/subjects]];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value[6];
    bool keep = id < n;
    if (keep) {
        for (uint s = 0; s < subjects; ++s) {
            float v = losses[s*n+id];
            // The upper bound guarantees exact representation of 6*v*2^34 in FP64.
            if (!(v >= 0 && v < 32768 && float(ulong(v*quantumScale)) == v*quantumScale))
                atomic_store_explicit(error,1,memory_order_relaxed);
            value[s] = v;
        }
        for (uint a = 0; a < anchorCount && keep; ++a) {
            bool strictlyWorse = true;
            for (uint s = 0; s < subjects; ++s) strictlyWorse &= value[s] > anchorLoss[a][s];
            if (strictlyWorse) keep = false;
        }
    }
    // Strict dominance protects every tied minimum, including resamples with zero weights.
    uint live = simd_sum(uint(keep));
    uint base = 0;
    if (simdLane == 0 && live) base = atomic_fetch_add_explicit(count,live,memory_order_relaxed);
    base = simd_broadcast_first(base);
    uint prefix = simd_prefix_exclusive_sum(uint(keep));
    if (keep) kept[base + prefix] = id;
}

kernel void grid_args(device uint *stats [[buffer(0)]], constant uint &nResamples [[buffer(1)]]) {
    stats[1] = (stats[0]+tile-1)/tile;
    stats[2] = nResamples;
    stats[3] = 1;
}

kernel void score_survivors(device const float *losses [[buffer(0)]],
                            device const uint *weights [[buffer(1)]],
                            device const uint *kept [[buffer(2)]],
                            device const uint *stats [[buffer(3)]],
                            device Hit *tiles [[buffer(4)]],
                            constant uint &n [[buffer(5)]],
                            uint2 local [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]]) {
    uint lane = local.x;
    threadgroup Hit hits[128];
    Hit best = empty_hit();
    uint end = min((group.x+1)*tile,stats[0]);
    for (uint i = group.x*tile+lane; i < end; i+=lanes)
        best = better(best, evaluate(losses,n,kept[i],weights+subjects*group.y));
    best = group_minimum(hits,lane,best);
    if (lane == 0) tiles[group.y*stats[1]+group.x] = best;
}

kernel void final_minimum(device const Hit *tiles [[buffer(0)]],
                          device const uint *stats [[buffer(1)]],
                          device uint *winners [[buffer(2)]],
                          uint lane [[thread_position_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]]) {
    threadgroup Hit hits[128];
    Hit best = empty_hit();
    for (uint i=lane; i<stats[1]; i+=lanes) best = better(best,tiles[group*stats[1]+i]);
    best = group_minimum(hits,lane,best);
    if (lane == 0) winners[group] = best.id;
}
