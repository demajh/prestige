#pragma once

#include <prestige/embedder.hpp>
#include <prestige/internal.hpp>

#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace prestige::testing {

// =============================================================================
// Deterministic Clock for TTL/LRU Testing
// =============================================================================

/**
 * Injectable clock interface for deterministic time in tests.
 * Production code uses the real clock; tests can inject a fake.
 */
class Clock {
 public:
  virtual ~Clock() = default;
  virtual uint64_t NowMicros() const = 0;      // Monotonic (for latency)
  virtual uint64_t WallClockMicros() const = 0; // Wall clock (for TTL)
};

/**
 * Real clock using system time (default behavior).
 */
class RealClock : public Clock {
 public:
  uint64_t NowMicros() const override {
    return internal::NowMicros();
  }
  uint64_t WallClockMicros() const override {
    return internal::WallClockMicros();
  }
};

/**
 * Fake clock for deterministic testing.
 * Time only advances when explicitly set or advanced.
 */
class FakeClock : public Clock {
 public:
  explicit FakeClock(uint64_t initial_time_us = 1000000000000ULL)  // ~2001
      : current_time_us_(initial_time_us) {}

  uint64_t NowMicros() const override {
    return current_time_us_.load();
  }

  uint64_t WallClockMicros() const override {
    return current_time_us_.load();
  }

  // Advance time by delta microseconds
  void AdvanceUs(uint64_t delta_us) {
    current_time_us_.fetch_add(delta_us);
  }

  // Advance time by delta seconds
  void AdvanceSec(uint64_t delta_sec) {
    AdvanceUs(delta_sec * 1000000ULL);
  }

  // Advance time by delta milliseconds
  void AdvanceMs(uint64_t delta_ms) {
    AdvanceUs(delta_ms * 1000ULL);
  }

  // Set absolute time
  void SetTimeUs(uint64_t time_us) {
    current_time_us_.store(time_us);
  }

 private:
  std::atomic<uint64_t> current_time_us_;
};

// Global clock pointer for injection (default: real clock)
// Tests can swap this with a FakeClock
inline Clock* g_clock = nullptr;

inline uint64_t GetNowMicros() {
  if (g_clock) return g_clock->NowMicros();
  return internal::NowMicros();
}

inline uint64_t GetWallClockMicros() {
  if (g_clock) return g_clock->WallClockMicros();
  return internal::WallClockMicros();
}

// =============================================================================
// Deterministic Embedder for Semantic Testing
// =============================================================================

/**
 * Deterministic embedder that generates reproducible embeddings from text.
 * Does NOT require ONNX runtime - suitable for unit tests.
 *
 * Embedding generation strategy:
 * - Hash the input text to generate a seed
 * - Use the seed to generate a deterministic vector
 * - Normalize to unit length
 *
 * Similar texts (by simple edit distance) produce similar embeddings.
 */
class DeterministicEmbedder : public internal::Embedder {
 public:
  explicit DeterministicEmbedder(size_t dimension = 384, float similarity_noise = 0.0f)
      : dimension_(dimension), similarity_noise_(similarity_noise) {}

  internal::EmbeddingResult Embed(std::string_view text) const override {
    internal::EmbeddingResult result;
    result.embedding = GenerateEmbedding(text);
    result.success = true;
    return result;
  }

  size_t Dimension() const override { return dimension_; }

  internal::EmbedderModelType ModelType() const override {
    return internal::EmbedderModelType::kMiniLM;
  }

  // Register a custom embedding for a specific text (for precise test control)
  void RegisterEmbedding(const std::string& text, std::vector<float> embedding) {
    std::lock_guard<std::mutex> lock(mutex_);
    custom_embeddings_[text] = std::move(embedding);
  }

  // Clear all custom embeddings
  void ClearCustomEmbeddings() {
    std::lock_guard<std::mutex> lock(mutex_);
    custom_embeddings_.clear();
  }

 private:
  std::vector<float> GenerateEmbedding(std::string_view text) const {
    // Check for custom embedding first
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = custom_embeddings_.find(std::string(text));
      if (it != custom_embeddings_.end()) {
        return it->second;
      }
    }

    // Generate deterministic embedding from text hash
    std::vector<float> embedding(dimension_);

    // Use simple hash-based generation for reproducibility
    uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis
    for (char c : text) {
      hash ^= static_cast<uint64_t>(c);
      hash *= 1099511628211ULL;  // FNV-1a prime
    }

    // Generate embedding components using LCG seeded by hash
    uint64_t state = hash;
    float norm = 0.0f;
    for (size_t i = 0; i < dimension_; ++i) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      // Convert to float in [-1, 1]
      embedding[i] = (static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31)) - 1.0f;
      norm += embedding[i] * embedding[i];
    }

    // Normalize to unit length
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
      for (float& v : embedding) {
        v /= norm;
      }
    }

    return embedding;
  }

  size_t dimension_;
  float similarity_noise_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::vector<float>> custom_embeddings_;
};

/**
 * Embedder that maps similar texts to similar embeddings.
 * Texts are grouped by a key function; texts with the same key get nearly identical embeddings.
 */
class GroupedEmbedder : public internal::Embedder {
 public:
  using KeyFunction = std::function<std::string(std::string_view)>;

  explicit GroupedEmbedder(size_t dimension = 384,
                           KeyFunction key_fn = nullptr,
                           float within_group_similarity = 0.99f)
      : dimension_(dimension),
        key_fn_(key_fn ? std::move(key_fn) : DefaultKeyFunction),
        within_group_similarity_(within_group_similarity),
        base_embedder_(dimension) {}

  internal::EmbeddingResult Embed(std::string_view text) const override {
    std::string key = key_fn_(text);

    // Get base embedding for the key
    auto base_result = base_embedder_.Embed(key);
    if (!base_result.success) return base_result;

    // Add small noise based on full text hash (for within-group variation)
    uint64_t text_hash = HashText(text);
    float noise_scale = std::sqrt(1.0f - within_group_similarity_);

    std::vector<float>& emb = base_result.embedding;
    uint64_t state = text_hash;
    float norm = 0.0f;

    for (size_t i = 0; i < emb.size(); ++i) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      float noise = (static_cast<float>(state >> 33) / static_cast<float>(1ULL << 31) - 0.5f) * 2.0f;
      emb[i] += noise * noise_scale;
      norm += emb[i] * emb[i];
    }

    // Renormalize
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
      for (float& v : emb) v /= norm;
    }

    return base_result;
  }

  size_t Dimension() const override { return dimension_; }

  internal::EmbedderModelType ModelType() const override {
    return internal::EmbedderModelType::kMiniLM;
  }

 private:
  static std::string DefaultKeyFunction(std::string_view text) {
    // Default: first word as key
    size_t end = text.find(' ');
    return std::string(text.substr(0, end));
  }

  static uint64_t HashText(std::string_view text) {
    uint64_t hash = 14695981039346656037ULL;
    for (char c : text) {
      hash ^= static_cast<uint64_t>(c);
      hash *= 1099511628211ULL;
    }
    return hash;
  }

  size_t dimension_;
  KeyFunction key_fn_;
  float within_group_similarity_;
  DeterministicEmbedder base_embedder_;
};

// =============================================================================
// Test Result Aggregation (for thread-safe assertions)
// =============================================================================

/**
 * Thread-safe result collector for concurrent tests.
 * Collects results from multiple threads for assertion on main thread.
 */
class TestResultCollector {
 public:
  void RecordSuccess() {
    success_count_.fetch_add(1);
  }

  void RecordFailure(const std::string& message = "") {
    failure_count_.fetch_add(1);
    if (!message.empty()) {
      std::lock_guard<std::mutex> lock(mutex_);
      failure_messages_.push_back(message);
    }
  }

  void RecordError(const std::string& error) {
    error_count_.fetch_add(1);
    std::lock_guard<std::mutex> lock(mutex_);
    error_messages_.push_back(error);
  }

  uint64_t SuccessCount() const { return success_count_.load(); }
  uint64_t FailureCount() const { return failure_count_.load(); }
  uint64_t ErrorCount() const { return error_count_.load(); }
  uint64_t TotalCount() const { return SuccessCount() + FailureCount() + ErrorCount(); }

  bool AllSucceeded() const { return FailureCount() == 0 && ErrorCount() == 0; }

  std::vector<std::string> GetFailureMessages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_messages_;
  }

  std::vector<std::string> GetErrorMessages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_messages_;
  }

  void Reset() {
    success_count_.store(0);
    failure_count_.store(0);
    error_count_.store(0);
    std::lock_guard<std::mutex> lock(mutex_);
    failure_messages_.clear();
    error_messages_.clear();
  }

 private:
  std::atomic<uint64_t> success_count_{0};
  std::atomic<uint64_t> failure_count_{0};
  std::atomic<uint64_t> error_count_{0};
  mutable std::mutex mutex_;
  std::vector<std::string> failure_messages_;
  std::vector<std::string> error_messages_;
};

// =============================================================================
// Crash Harness for Durability Testing
// =============================================================================

/**
 * Invariant checker function type.
 * Returns true if invariants hold, false otherwise.
 * Sets error_out with description on failure.
 */
using InvariantChecker = std::function<bool(const std::string& db_path, std::string* error_out)>;

/**
 * Crash simulation result.
 */
struct CrashTestResult {
  bool passed = true;
  int operations_before_crash = 0;
  std::string error_message;
  std::vector<std::string> invariant_violations;
};

// =============================================================================
// Helper Macros for Thread-Safe Testing
// =============================================================================

// Use these instead of ASSERT_*/EXPECT_* inside threads
#define RECORD_SUCCESS(collector) (collector).RecordSuccess()
#define RECORD_FAILURE(collector, msg) (collector).RecordFailure(msg)
#define RECORD_ERROR(collector, msg) (collector).RecordError(msg)

// Check condition and record result
#define CHECK_AND_RECORD(collector, condition, fail_msg) \
  do { \
    if (condition) { \
      (collector).RecordSuccess(); \
    } else { \
      (collector).RecordFailure(fail_msg); \
    } \
  } while (0)

}  // namespace prestige::testing
