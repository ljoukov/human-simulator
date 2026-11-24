// Native CPU reference for the CUDA calibration benchmark.
// Payload: little-endian u32 frame/scenario counts, then raw frames, observations,
// and candidate rows. The output is one little-endian f32 loss per candidate.
#include "contact_score.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

static_assert(sizeof(knee::RawFrame) == 4 * sizeof(float));
static_assert(sizeof(knee::Observation) == 4 * sizeof(float));
static_assert(sizeof(knee::Candidate) == 12 * sizeof(float));
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

struct Input {
  std::vector<knee::RawFrame> frames;
  std::vector<knee::Observation> observations;
  std::vector<knee::Candidate> candidates;
};

template <typename Record>
std::vector<Record> read_records(std::istream& file, std::size_t count) {
  std::vector<Record> values(count);
  if (!file.read(reinterpret_cast<char*>(values.data()), count * sizeof(Record))) {
    throw std::runtime_error("Input ended before all rows could be read.");
  }
  return values;
}

Input read_input(const char* path) {
  const std::uint32_t endian = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian) == 0) {
    throw std::runtime_error("The benchmark payload requires a little-endian host.");
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("Cannot open benchmark input.");
  const auto bytes = file.tellg();
  file.seekg(0);
  std::uint32_t counts[2]{};
  if (!file.read(reinterpret_cast<char*>(counts), sizeof(counts)) || !counts[0] || !counts[1]) {
    throw std::runtime_error("Input requires nonzero frame and scenario counts.");
  }
  const auto expected_bytes = sizeof(counts)
      + std::uint64_t(counts[0]) * (sizeof(knee::RawFrame) + sizeof(knee::Observation))
      + std::uint64_t(counts[1]) * sizeof(knee::Candidate);
  if (bytes < 0 || static_cast<std::uint64_t>(bytes) != expected_bytes) {
    throw std::runtime_error("Input byte count does not match the declared frame and scenario counts.");
  }
  return {
      read_records<knee::RawFrame>(file, counts[0]),
      read_records<knee::Observation>(file, counts[0]),
      read_records<knee::Candidate>(file, counts[1]),
  };
}

void write_scores(const char* path, const std::vector<float>& scores) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(scores.data()), scores.size() * sizeof(float));
  file.close();
  if (!file) throw std::runtime_error("Could not write benchmark scores.");
}

class ScorePool {
 public:
  ScorePool(const std::vector<knee::PreparedFrame>& frames,
            const std::vector<knee::Candidate>& candidates,
            std::vector<float>& scores, std::size_t worker_count)
      : frames_(frames), candidates_(candidates), scores_(scores) {
    if (worker_count == 1) return;

    workers_.reserve(worker_count);
    try {
      for (std::size_t index = 0; index < worker_count; ++index) {
        workers_.emplace_back(&ScorePool::worker, this,
                              index * candidates_.size() / worker_count,
                              (index + 1) * candidates_.size() / worker_count);
      }
    } catch (...) {
      stop();
      throw;
    }
  }

  ~ScorePool() { stop(); }

  void score() {
    if (workers_.empty()) {
      score_range(0, candidates_.size());
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    pending_ = workers_.size();
    ++generation_;
    work_.notify_all();
    done_.wait(lock, [&] { return pending_ == 0; });
  }

 private:
  void score_range(std::size_t start, std::size_t end) const {
    for (std::size_t index = start; index < end; ++index) {
      const auto& candidate = candidates_[index];
      // The reference JavaScript solver sums in double precision and returns f32.
      double loss = 0;
      for (const auto& frame : frames_) loss += knee::frame_score(candidate, frame);
      scores_[index] = static_cast<float>(loss);
    }
  }

  void worker(std::size_t start, std::size_t end) {
    std::uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      work_.wait(lock, [&] { return stopping_ || seen != generation_; });
      if (stopping_) return;
      seen = generation_;
      lock.unlock();
      score_range(start, end);
      lock.lock();
      if (--pending_ == 0) done_.notify_one();
    }
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    work_.notify_all();
    for (auto& worker : workers_) worker.join();
  }

  const std::vector<knee::PreparedFrame>& frames_;
  const std::vector<knee::Candidate>& candidates_;
  std::vector<float>& scores_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_, done_;
  std::uint64_t generation_ = 0;
  std::size_t pending_ = 0;
  bool stopping_ = false;
};

unsigned positive_integer(std::string_view text, const char* field) {
  unsigned value = 0;
  const auto end = text.data() + text.size();
  const auto parsed = std::from_chars(text.data(), end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end || value == 0) {
    throw std::runtime_error(std::string(field) + " must be a positive integer.");
  }
  return value;
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 5) {
      throw std::runtime_error("Usage: native-scores INPUT.bin SCORES.bin [REPEATS] [THREADS]");
    }
    const auto repeats = argc >= 4 ? positive_integer(argv[3], "REPEATS") : 5;
    const auto requested_workers = argc == 5 ? positive_integer(argv[4], "THREADS")
                                              : std::max(1u, std::thread::hardware_concurrency());
    const auto input = read_input(argv[1]);
    const auto workers = std::min<std::size_t>(requested_workers, input.candidates.size());
    const auto setup_start = Clock::now();
    std::vector<knee::PreparedFrame> frames;
    frames.reserve(input.frames.size());
    for (std::size_t index = 0; index < input.frames.size(); ++index) {
      frames.push_back(knee::prepare_frame(input.frames[index], input.observations[index]));
    }
    std::vector<float> scores(input.candidates.size());
    ScorePool pool(frames, input.candidates, scores, workers);
    const auto setup_ms = elapsed_ms(setup_start);
    pool.score();  // Warm the worker pool and resident input before timing.
    std::vector<double> samples;
    samples.reserve(repeats);
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
      const auto start = Clock::now();
      pool.score();
      samples.push_back(elapsed_ms(start));
    }
    write_scores(argv[2], scores);
    std::cout << std::setprecision(12)
              << "{\"backend\":\"native-cpu\",\"workers\":" << workers
              << ",\"batchMs\":" << median(samples) << ",\"setupMs\":" << setup_ms
              << ",\"batchSamplesMs\":[";
    for (std::size_t index = 0; index < samples.size(); ++index) {
      if (index) std::cout << ',';
      std::cout << samples[index];
    }
    std::cout << "]}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
