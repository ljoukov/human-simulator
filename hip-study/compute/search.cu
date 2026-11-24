// Same frontal-plane equilibrium and loss for CUDA, CPU SIMD, and scalar tails.
// The only reduction is along the frames of one virtual anatomy.
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define HIP_DEVICE __device__ __forceinline__
HIP_DEVICE float hip_sqrt(float v) { return sqrtf(v); }
#else
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
#define HIP_DEVICE inline
using Float4 = float __attribute__((vector_size(16)));
inline float hip_sqrt(float v) { return std::sqrt(v); }
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
inline Float4 hip_sqrt(Float4 v) { return vsqrtq_f32(v); }
#elif defined(__SSE__)
#include <xmmintrin.h>
inline Float4 hip_sqrt(Float4 v) { return _mm_sqrt_ps(v); }
#else
inline Float4 hip_sqrt(Float4 v) {
    for (int i=0; i<4; ++i) v[i] = std::sqrt(v[i]);
    return v;
}
#endif
#endif

// f = vertical and outward horizontal GRF (BW), positive/negative frontal
// moment magnitudes (BW m), inverse posture factor, implant target, loss
// weight, sin(thigh adduction). See study.py for physical parameter packing.
template<typename T>
HIP_DEVICE T objective(const float *frames, const T *p, int nFrames) {
    T loss = {};
    for (int k = 0; k < nFrames; ++k) {
        const float *f = frames + 8*k;
        const T coactivation = p[4] * f[0];
        const T abductorMoment = f[2] + p[2] * coactivation;
        const T gluteal = abductorMoment * p[0] * f[4];
        const T band = abductorMoment * p[1];
        const T adductor = f[3] * p[3] + coactivation;
        const T vertical = f[0] + gluteal*p[6] + band + adductor*0.85f;
        const T horizontal = f[1] - gluteal*p[5] - adductor*0.52678269f;
        const T residual = hip_sqrt(vertical*vertical + horizontal*horizontal) - f[5];
        loss += residual*residual*f[6];
    }
    return loss;
}

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__ void hip_score(const float *frames, const float *parameters,
                                     float *scores, int nFrames, int nCandidates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nCandidates) scores[i] = objective(frames, parameters + 9*i, nFrames);
}
#else
extern "C" void hip_cpu(const float *frames, const float *parameters, float *scores,
                        int nFrames, int nCandidates, int nThreads) {
    const int groups = nCandidates / 4;
    auto work = [&](int first, int last) {
        for (int g = first; g < last; ++g) {
            Float4 p[7];
            for (int j = 0; j < 7; ++j) {
                for (int lane = 0; lane < 4; ++lane)
                    p[j][lane] = parameters[(4*g+lane)*9+j];
            }
            const Float4 losses = objective(frames, p, nFrames);
            for (int lane = 0; lane < 4; ++lane) scores[4*g+lane] = losses[lane];
        }
    };
    std::vector<std::thread> workers;
    for (int t = 1; t < nThreads; ++t)
        workers.emplace_back(work, groups*t/nThreads, groups*(t+1)/nThreads);
    work(0, groups/nThreads);
    for (auto &worker : workers) worker.join();
    for (int i = groups*4; i < nCandidates; ++i)
        scores[i] = objective(frames, parameters + i*9, nFrames);
}
#endif
