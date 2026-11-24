#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define KNEE_HOST_DEVICE __host__ __device__
#else
#define KNEE_HOST_DEVICE
#endif

namespace knee {

// Match the stride and operation order of src/mechanics.ts. Compile with FMA
// contraction disabled so the C++ and CUDA implementations use the same f32 model.
struct RawFrame {
  float axial, adduction, flexion, flexion_angle;
};

struct Observation {
  float medial, lateral, total, mask;
};

struct Candidate {
  float extensor_arm, flexor_arm, medial_arm, lateral_arm;
  float muscle_offset, axial_offset, axial_scale, co_contraction;
  float extensor_capacity, flexor_capacity, extensor_projection, flexor_projection;
};

struct PreparedFrame {
  float axial, adduction, flexion, extensor_shape, flexor_shape;
  float medial, lateral, total;
  std::uint32_t mask;
};

static_assert(sizeof(float) == 4);
static_assert(sizeof(RawFrame) == 4 * sizeof(float));
static_assert(sizeof(Observation) == 4 * sizeof(float));
static_assert(sizeof(Candidate) == 12 * sizeof(float));

KNEE_HOST_DEVICE inline float minimum(float a, float b) noexcept {
  return a < b ? a : b;
}

KNEE_HOST_DEVICE inline float maximum(float a, float b) noexcept {
  return a > b ? a : b;
}

KNEE_HOST_DEVICE inline float absolute(float a) noexcept {
  return a < 0.0f ? -a : a;
}

KNEE_HOST_DEVICE inline float square(float a) noexcept {
  return a * a;
}

KNEE_HOST_DEVICE inline PreparedFrame prepare_frame(const RawFrame& raw, const Observation& measured) noexcept {
  constexpr float reference = 0.2617993950843811f;
  constexpr float end = 1.0471975803375244f;
  const float angle = minimum(maximum(raw.flexion_angle, 0.0f), end) - reference;
  const float extensor_shape = 1.0f + angle * (-0.145985f + angle * -0.1977134f);
  const float flexor_shape = 1.0f + angle * (0.4933224f + angle * -0.20718245f);
  return {raw.axial, raw.adduction, raw.flexion, extensor_shape, flexor_shape,
    measured.medial, measured.lateral, measured.total, static_cast<std::uint32_t>(measured.mask)};
}

KNEE_HOST_DEVICE inline float frame_score(const Candidate& p, const PreparedFrame& f) noexcept {
  if (f.mask == 0) return 0.0f;
  const float axial = f.axial * p.axial_scale;
  const float extensor_lever = p.extensor_arm * f.extensor_shape;
  const float flexor_lever = p.flexor_arm * f.flexor_shape;
  const float extensor_limit = extensor_lever * p.extensor_capacity;
  const float flexor_limit = flexor_lever * p.flexor_capacity;
  const float extensor_torque = minimum(maximum(f.flexion, 0.0f), extensor_limit);
  const float flexor_torque = minimum(maximum(-f.flexion, 0.0f), flexor_limit);
  const float requested_pair = (p.co_contraction * maximum(axial, 0.0f)) * extensor_lever;
  const float pair = minimum(requested_pair, minimum(extensor_limit - extensor_torque, flexor_limit - flexor_torque));
  const float extensor_force = (extensor_torque + pair) / extensor_lever;
  const float flexor_force = (flexor_torque + pair) / flexor_lever;
  const float muscle_compression = extensor_force * p.extensor_projection + flexor_force * p.flexor_projection;
  const float compression = maximum(0.0f, axial + muscle_compression);
  const float frontal_target = (f.adduction + axial * p.axial_offset) + muscle_compression * p.muscle_offset;
  const float contact_span = p.medial_arm + p.lateral_arm;
  const float medial = minimum(maximum((frontal_target + p.lateral_arm * compression) / contact_span, 0.0f), compression);
  const float lateral = compression - medial;
  const float sagittal_residual = f.flexion - (extensor_torque - flexor_torque);
  const float frontal_residual = frontal_target - (p.medial_arm * medial - p.lateral_arm * lateral);
  constexpr float moment_tolerance = 0.005f;
  constexpr float load_lever = 0.05f;
  const float excess = maximum(0.0f, absolute(sagittal_residual) - moment_tolerance)
    + maximum(0.0f, absolute(frontal_residual) - moment_tolerance);

  float loss = 0.0f;
  if (f.mask & 1u) loss += square(medial - f.medial);
  if (f.mask & 2u) loss += square(lateral - f.lateral);
  if (f.mask & 4u) loss += square(medial + lateral - f.total);
  return loss + square(excess / load_lever);
}

}  // namespace knee

#undef KNEE_HOST_DEVICE
