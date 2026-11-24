// nvcc -std=c++17 -O3 --fmad=false -arch=sm_90 h100_scores.cu -o h100-scores
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "contact_score.hpp"

namespace {

using knee::Candidate;
using knee::Observation;
using knee::PreparedFrame;
using knee::RawFrame;
constexpr unsigned block_size = 128;
constexpr unsigned warp_size = 32;
constexpr unsigned warps_per_block = block_size / warp_size;
using Clock = std::chrono::steady_clock;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// The comparison kernel follows the existing WebGPU workgroup mapping and
// reduces in shared memory. Both GPU kernels return one score per candidate.
__global__ void score_blocks(const RawFrame* frames, const Observation* observations,
    const Candidate* candidates, float* scores, unsigned frame_count) {
  __shared__ float partial[block_size];
  const Candidate candidate = candidates[blockIdx.x];
  const unsigned lane = threadIdx.x;
  float sum = 0.0f;
  for (unsigned frame = lane; frame < frame_count; frame += block_size) {
    sum += knee::frame_score(candidate, knee::prepare_frame(frames[frame], observations[frame]));
  }
  partial[lane] = sum;
  __syncthreads();
  for (unsigned offset = block_size / 2; offset > 0; offset >>= 1) {
    if (lane < offset) partial[lane] += partial[lane + offset];
    __syncthreads();
  }
  if (lane == 0) scores[blockIdx.x] = partial[0];
}

__global__ void prepare_frames(const RawFrame* frames, const Observation* observations,
    PreparedFrame* prepared, unsigned frame_count) {
  const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < frame_count) prepared[id] = knee::prepare_frame(frames[id], observations[id]);
}

// Assign one warp per candidate. Each lane evaluates a strided set of active
// frames, loading consecutive frame records across the warp. The reduction
// stays in registers and needs no cross-warp synchronization.
__global__ void score_warps(const PreparedFrame* prepared, const Candidate* candidates,
    float* scores, unsigned frame_count, unsigned scenario_count) {
  const unsigned scenario = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
  if (scenario >= scenario_count) return;
  const unsigned lane = threadIdx.x % warp_size;
  const Candidate candidate = candidates[scenario];
  float sum = 0.0f;
  for (unsigned frame = lane; frame < frame_count; frame += warp_size) {
    sum += knee::frame_score(candidate, prepared[frame]);
  }
  for (unsigned offset = warp_size / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) scores[scenario] = sum;
}

template<typename T, bool pinned = false>
class Buffer {
 public:
  explicit Buffer(std::size_t count) {
    if constexpr (pinned) check(cudaMallocHost(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
    else check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
  }

  ~Buffer() {
    if constexpr (pinned) cudaFreeHost(pointer_);
    else cudaFree(pointer_);
  }

  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  T* get() const { return pointer_; }

 private:
  T* pointer_ = nullptr;
};

class Event {
 public:
  Event() { check(cudaEventCreate(&value)); }
  ~Event() { cudaEventDestroy(value); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  cudaEvent_t value = nullptr;
};

template<typename T>
void read_records(std::istream& input, T* data, std::size_t count) {
  input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
  if (!input) throw std::runtime_error("Incomplete score inputs");
}

double wall_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double>& values) {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  return (values[(values.size() - 1) / 2] + values[middle]) / 2.0;
}

struct Timing {
  double kernel_ms, batch_ms;
};

struct Timings {
  std::vector<double> kernels, batches;

  void add(Timing value) {
    kernels.push_back(value.kernel_ms);
    batches.push_back(value.batch_ms);
  }

  Timing result() { return {median(kernels), median(batches)}; }
};

struct Agreement {
  double max_absolute = 0.0;
  double max_relative = 0.0;
  unsigned out_of_tolerance = 0;
};

Agreement compare(const float* reference, const float* optimized, unsigned scenarios) {
  Agreement agreement;
  for (unsigned i = 0; i < scenarios; ++i) {
    if (!std::isfinite(reference[i]) || !std::isfinite(optimized[i])) {
      throw std::runtime_error("Nonfinite GPU score at candidate " + std::to_string(i));
    }
    const double difference = std::abs(static_cast<double>(reference[i]) - optimized[i]);
    const double scale = std::max(1e-6, std::abs(static_cast<double>(reference[i])));
    agreement.max_absolute = std::max(agreement.max_absolute, difference);
    agreement.max_relative = std::max(agreement.max_relative, difference / scale);
    if (difference > 1e-5 + scale * 2e-5) ++agreement.out_of_tolerance;
  }
  return agreement;
}

void print_timing(const Timing& timing) {
  std::cout << "{\"kernelMs\":" << timing.kernel_ms << ",\"batchMs\":" << timing.batch_ms << '}';
}

int run(int argc, char** argv) {
  if (argc < 3 || argc > 4) throw std::runtime_error("Usage: h100-scores INPUT.bin SCORES.bin [REPEATS]");
  std::size_t repeat_end = 0;
  const int repeats = argc == 4 ? std::stoi(argv[3], &repeat_end) : 7;
  if (repeats < 1 || repeats > 50 || (argc == 4 && argv[3][repeat_end] != '\0')) {
    throw std::runtime_error("Repeats must be an integer from 1 to 50");
  }
  // Input is LE u32[2] (frameCount, scenarioCount), then raw f32 frame rows,
  // observation rows and parameter tuples in the order used in mechanics.ts.
  std::ifstream input(argv[1], std::ios::binary);
  std::array<std::uint32_t, 2> counts{};
  read_records(input, counts.data(), counts.size());
  const auto frames = counts[0];
  const auto scenarios = counts[1];
  if (frames == 0 || scenarios == 0 || frames > 1'000'000 || scenarios > 10'000'000) {
    throw std::runtime_error("Frame and candidate counts outside supported range");
  }

  std::vector<RawFrame> frame_rows(frames);
  std::vector<Observation> observed(frames);
  std::vector<Candidate> candidate_rows(scenarios);
  read_records(input, frame_rows.data(), frames);
  read_records(input, observed.data(), frames);
  read_records(input, candidate_rows.data(), scenarios);
  if (input.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("Extra bytes in score inputs");
  for (const auto& row : observed) {
    if (!std::isfinite(row.mask) || row.mask < 0 || row.mask > 7 || std::trunc(row.mask) != row.mask) {
      throw std::runtime_error("Observation mask must be a 3-bit integer");
    }
  }

  const auto setup_start = Clock::now();
  check(cudaSetDevice(0));
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0));
  int runtime_version, driver_version;
  check(cudaRuntimeGetVersion(&runtime_version));
  check(cudaDriverGetVersion(&driver_version));
  Buffer<RawFrame> device_frames(frames);
  Buffer<Observation> device_observed(frames);
  Buffer<PreparedFrame> device_prepared(frames);
  Buffer<Candidate> device_candidates(scenarios);
  Buffer<float> device_scores(scenarios);
  Buffer<Candidate, true> host_candidates(scenarios);
  Buffer<float, true> baseline_scores(scenarios);
  Buffer<float, true> optimized_scores(scenarios);
  std::copy(candidate_rows.begin(), candidate_rows.end(), host_candidates.get());
  check(cudaMemcpy(device_frames.get(), frame_rows.data(), frames * sizeof(RawFrame), cudaMemcpyHostToDevice));
  check(cudaMemcpy(device_observed.get(), observed.data(), frames * sizeof(Observation), cudaMemcpyHostToDevice));
  prepare_frames<<<(frames + block_size - 1) / block_size, block_size>>>(
    device_frames.get(), device_observed.get(), device_prepared.get(), frames);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  Event kernel_start;
  Event kernel_end;
  const double setup_ms = wall_ms(setup_start);

  auto timed_run = [&](bool optimized) {
    const auto batch_start = Clock::now();
    check(cudaMemcpyAsync(device_candidates.get(), host_candidates.get(), scenarios * sizeof(Candidate), cudaMemcpyHostToDevice));
    check(cudaEventRecord(kernel_start.value));
    if (optimized) {
      score_warps<<<(scenarios + warps_per_block - 1) / warps_per_block, block_size>>>(
        device_prepared.get(), device_candidates.get(), device_scores.get(), frames, scenarios);
    } else {
      score_blocks<<<scenarios, block_size>>>(device_frames.get(), device_observed.get(),
        device_candidates.get(), device_scores.get(), frames);
    }
    check(cudaGetLastError());
    check(cudaEventRecord(kernel_end.value));
    const auto& result = optimized ? optimized_scores : baseline_scores;
    check(cudaMemcpyAsync(result.get(), device_scores.get(), scenarios * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaStreamSynchronize(nullptr));
    const double batch_ms = wall_ms(batch_start);
    float kernel_ms;
    check(cudaEventElapsedTime(&kernel_ms, kernel_start.value, kernel_end.value));
    return Timing{kernel_ms, batch_ms};
  };

  timed_run(false);
  timed_run(true);
  Timings baseline, optimized;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    if (repeat % 2 == 0) {
      baseline.add(timed_run(false));
      optimized.add(timed_run(true));
    } else {
      optimized.add(timed_run(true));
      baseline.add(timed_run(false));
    }
  }
  const auto agreement = compare(baseline_scores.get(), optimized_scores.get(), scenarios);
  std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(optimized_scores.get()), scenarios * sizeof(float));
  output.close();
  if (!output) throw std::runtime_error("Unable to write score output");

  std::cout << std::setprecision(9) << "{\"device\":" << std::quoted(properties.name)
    << ",\"computeCapability\":\"" << properties.major << '.' << properties.minor
    << "\",\"smCount\":" << properties.multiProcessorCount << ",\"runtimeVersion\":" << runtime_version
    << ",\"driverVersion\":" << driver_version << ",\"frames\":" << frames << ",\"scenarios\":" << scenarios
    << ",\"repeats\":" << repeats << ",\"gpuSetupMs\":" << setup_ms << ",\"optimized\":";
  print_timing(optimized.result());
  std::cout << ",\"blockBaseline\":";
  print_timing(baseline.result());
  std::cout << ",\"baselineAgreement\":{\"maxAbsoluteError\":" << agreement.max_absolute
    << ",\"maxRelativeError\":" << agreement.max_relative
    << ",\"outOfTolerance\":" << agreement.out_of_tolerance << "}}\n";
  return agreement.out_of_tolerance == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
