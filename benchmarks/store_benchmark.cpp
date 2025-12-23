// Performance benchmarks for prestige Store
// Uses Google Benchmark for accurate measurement and CI regression tracking
//
// Organization:
// 1. MICROBENCHMARKS: CPU-bound operations (SHA-256, serialization, etc.)
//    - No I/O, no database operations
//    - Useful for measuring algorithm performance
// 2. MACROBENCHMARKS: Store operations (Put, Get, Delete, etc.)
//    - Full database operations with I/O
//    - Useful for measuring end-to-end performance
//
// Benchmark hygiene:
// - Pre-generate all test data outside timing loops
// - Use state.PauseTiming()/ResumeTiming() for necessary setup
// - Use fixed dataset sizes for reproducible results

#include <benchmark/benchmark.h>

#include <prestige/internal.hpp>
#include <prestige/normalize.hpp>
#include <prestige/store.hpp>

#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

// =============================================================================
// Benchmark Fixtures and Helpers
// =============================================================================

class StoreBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(const benchmark::State& state) override {
    // Get temp directory with fallback options
    std::filesystem::path temp_base;
    std::error_code ec;

    // Try standard temp directory first
    temp_base = std::filesystem::temp_directory_path(ec);
    if (ec || temp_base.empty()) {
      // Fallback to common temp paths
      if (std::filesystem::exists("/tmp", ec)) {
        temp_base = "/tmp";
      } else if (std::filesystem::exists("/var/tmp", ec)) {
        temp_base = "/var/tmp";
      } else {
        // Last resort: use current directory
        temp_base = std::filesystem::current_path(ec);
        if (ec) {
          temp_base = ".";
        }
      }
    }

    // Ensure base directory exists
    if (!std::filesystem::exists(temp_base, ec)) {
      std::filesystem::create_directories(temp_base, ec);
    }

    test_dir_ = temp_base / ("prestige_bench_" + RandomSuffix());

    // Create test directory with error handling
    ec.clear();
    std::filesystem::create_directories(test_dir_, ec);
    if (ec) {
      // If we can't create in temp, try current directory
      test_dir_ = std::filesystem::path(".") / ("prestige_bench_" + RandomSuffix());
      std::filesystem::create_directories(test_dir_, ec);
    }

    db_path_ = (test_dir_ / "bench_db").string();
  }

  void TearDown(const benchmark::State& state) override {
    store_.reset();
    std::error_code ec;
    std::filesystem::remove_all(test_dir_, ec);
  }

  rocksdb::Status OpenStore(const prestige::Options& opt = prestige::Options{}) {
    return prestige::Store::Open(db_path_, &store_, opt);
  }

  std::string RandomSuffix() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    return std::to_string(dis(gen));
  }

  std::string RandomString(size_t length) {
    static const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";
    std::string result(length, '\0');
    std::uniform_int_distribution<> dis(0, sizeof(charset) - 2);
    for (size_t i = 0; i < length; ++i) {
      result[i] = charset[dis(gen_)];
    }
    return result;
  }

  std::string RandomBytes(size_t length) {
    std::string result(length, '\0');
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < length; ++i) {
      result[i] = static_cast<char>(dis(gen_));
    }
    return result;
  }

  std::filesystem::path test_dir_;
  std::string db_path_;
  std::unique_ptr<prestige::Store> store_;
  std::mt19937 gen_{std::random_device{}()};
};

// =============================================================================
// PART 1: MICROBENCHMARKS - CPU-bound operations without I/O
// =============================================================================

// =============================================================================
// Internal Utilities Benchmarks
// =============================================================================

static void BM_SHA256_Small(benchmark::State& state) {
  std::string data(state.range(0), 'x');
  for (auto _ : state) {
    auto digest = prestige::internal::Sha256::Digest(data);
    benchmark::DoNotOptimize(digest);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_SHA256_Small)->Range(64, 4096);

static void BM_SHA256_Large(benchmark::State& state) {
  std::string data(state.range(0), 'x');
  for (auto _ : state) {
    auto digest = prestige::internal::Sha256::Digest(data);
    benchmark::DoNotOptimize(digest);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_SHA256_Large)->Range(4096, 1 << 20);

static void BM_RandomObjectId128(benchmark::State& state) {
  for (auto _ : state) {
    auto id = prestige::internal::RandomObjectId128();
    benchmark::DoNotOptimize(id);
  }
}
BENCHMARK(BM_RandomObjectId128);

static void BM_ObjectMeta_Serialize(benchmark::State& state) {
  prestige::internal::ObjectMeta meta;
  meta.digest_key = std::string(32, '\x00');
  meta.created_at_us = 1234567890123456ULL;
  meta.last_accessed_us = 1234567890654321ULL;
  meta.size_bytes = 1024 * 1024;

  for (auto _ : state) {
    auto serialized = meta.Serialize();
    benchmark::DoNotOptimize(serialized);
  }
}
BENCHMARK(BM_ObjectMeta_Serialize);

static void BM_ObjectMeta_Deserialize(benchmark::State& state) {
  prestige::internal::ObjectMeta meta;
  meta.digest_key = std::string(32, '\x00');
  meta.created_at_us = 1234567890123456ULL;
  meta.last_accessed_us = 1234567890654321ULL;
  meta.size_bytes = 1024 * 1024;
  std::string serialized = meta.Serialize();

  prestige::internal::ObjectMeta result;
  for (auto _ : state) {
    prestige::internal::ObjectMeta::Deserialize(serialized, &result);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_ObjectMeta_Deserialize);

static void BM_LRUKey_Make(benchmark::State& state) {
  std::string obj_id(16, '\x00');
  uint64_t timestamp = 1234567890123456ULL;

  for (auto _ : state) {
    auto key = prestige::internal::MakeLRUKey(timestamp, obj_id);
    benchmark::DoNotOptimize(key);
  }
}
BENCHMARK(BM_LRUKey_Make);

static void BM_LRUKey_Parse(benchmark::State& state) {
  std::string obj_id(16, '\x00');
  std::string key = prestige::internal::MakeLRUKey(1234567890123456ULL, obj_id);

  uint64_t ts;
  std::string parsed_id;
  for (auto _ : state) {
    prestige::internal::ParseLRUKey(key, &ts, &parsed_id);
    benchmark::DoNotOptimize(ts);
    benchmark::DoNotOptimize(parsed_id);
  }
}
BENCHMARK(BM_LRUKey_Parse);

// =============================================================================
// Normalization Benchmarks
// =============================================================================

static void BM_Normalize_None(benchmark::State& state) {
  std::string input(state.range(0), 'x');
  for (auto _ : state) {
    auto result = prestige::internal::Normalize(input, prestige::NormalizationMode::kNone);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_Normalize_None)->Range(64, 1 << 16);

static void BM_Normalize_Whitespace(benchmark::State& state) {
  // Create input with mixed whitespace
  std::string input;
  for (int i = 0; i < state.range(0); ++i) {
    if (i % 10 == 0) {
      input += "  ";
    }
    input += 'x';
  }

  for (auto _ : state) {
    auto result = prestige::internal::Normalize(input, prestige::NormalizationMode::kWhitespace);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * input.size());
}
BENCHMARK(BM_Normalize_Whitespace)->Range(64, 1 << 16);

static void BM_Normalize_ASCII(benchmark::State& state) {
  std::string input(state.range(0), 'A');  // All uppercase
  for (auto _ : state) {
    auto result = prestige::internal::Normalize(input, prestige::NormalizationMode::kASCII);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_Normalize_ASCII)->Range(64, 1 << 16);

static void BM_Normalize_Unicode(benchmark::State& state) {
  // Mix of ASCII and UTF-8
  std::string input;
  for (int i = 0; i < state.range(0) / 4; ++i) {
    input += "A\xc3\x80x";  // A + À + x
  }

  for (auto _ : state) {
    auto result = prestige::internal::Normalize(input, prestige::NormalizationMode::kUnicode);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * input.size());
}
BENCHMARK(BM_Normalize_Unicode)->Range(64, 1 << 16);

// =============================================================================
// Embedding Utilities Benchmarks
// =============================================================================

static void BM_Embedding_Serialize(benchmark::State& state) {
  std::vector<float> embedding(384);
  for (int i = 0; i < 384; ++i) {
    embedding[i] = static_cast<float>(i) / 384.0f;
  }

  for (auto _ : state) {
    auto serialized = prestige::internal::SerializeEmbedding(embedding);
    benchmark::DoNotOptimize(serialized);
  }
}
BENCHMARK(BM_Embedding_Serialize);

static void BM_Embedding_Deserialize(benchmark::State& state) {
  std::vector<float> embedding(384);
  for (int i = 0; i < 384; ++i) {
    embedding[i] = static_cast<float>(i) / 384.0f;
  }
  std::string serialized = prestige::internal::SerializeEmbedding(embedding);

  std::vector<float> result;
  for (auto _ : state) {
    prestige::internal::DeserializeEmbedding(serialized, &result);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_Embedding_Deserialize);

static void BM_CosineSimilarity(benchmark::State& state) {
  std::vector<float> a(384), b(384);
  for (int i = 0; i < 384; ++i) {
    a[i] = static_cast<float>(i) / 384.0f;
    b[i] = static_cast<float>(383 - i) / 384.0f;
  }

  for (auto _ : state) {
    float sim = prestige::internal::CosineSimilarity(a, b);
    benchmark::DoNotOptimize(sim);
  }
}
BENCHMARK(BM_CosineSimilarity);

// =============================================================================
// PART 2: MACROBENCHMARKS - Store operations with I/O
// =============================================================================

// =============================================================================
// Store Operation Benchmarks
// =============================================================================

BENCHMARK_DEFINE_F(StoreBenchmark, Put_UniqueValues)(benchmark::State& state) {
  OpenStore();

  // Pre-generate all values outside timing loop
  const size_t kNumValues = 10000;
  std::vector<std::string> values;
  values.reserve(kNumValues);
  for (size_t i = 0; i < kNumValues; ++i) {
    values.push_back(RandomString(state.range(0)));
  }

  int64_t key_id = 0;
  size_t value_idx = 0;

  for (auto _ : state) {
    std::string key = "key_" + std::to_string(key_id++);
    auto status = store_->Put(key, values[value_idx]);
    benchmark::DoNotOptimize(status);
    value_idx = (value_idx + 1) % kNumValues;
  }
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
BENCHMARK_REGISTER_F(StoreBenchmark, Put_UniqueValues)->Range(64, 1 << 16);

BENCHMARK_DEFINE_F(StoreBenchmark, Put_DuplicateValues)(benchmark::State& state) {
  OpenStore();
  std::string shared_value = RandomString(state.range(0));
  int64_t key_id = 0;

  for (auto _ : state) {
    std::string key = "key_" + std::to_string(key_id++);
    auto status = store_->Put(key, shared_value);
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, Put_DuplicateValues)->Range(64, 1 << 16);

BENCHMARK_DEFINE_F(StoreBenchmark, Get_Existing)(benchmark::State& state) {
  OpenStore();

  // Pre-generate keys for access pattern
  const int kNumKeys = 1000;
  const int kNumLookups = 10000;
  std::vector<std::string> keys;
  keys.reserve(kNumKeys);

  // Pre-populate store
  for (int i = 0; i < kNumKeys; ++i) {
    std::string key = "key_" + std::to_string(i);
    keys.push_back(key);
    store_->Put(key, "value_" + std::to_string(i));
  }

  // Pre-generate random lookup sequence
  std::vector<int> lookup_indices;
  lookup_indices.reserve(kNumLookups);
  std::uniform_int_distribution<> dis(0, kNumKeys - 1);
  for (int i = 0; i < kNumLookups; ++i) {
    lookup_indices.push_back(dis(gen_));
  }

  std::string value;
  size_t lookup_idx = 0;

  for (auto _ : state) {
    auto status = store_->Get(keys[lookup_indices[lookup_idx]], &value);
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(value);
    lookup_idx = (lookup_idx + 1) % kNumLookups;
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, Get_Existing);

BENCHMARK_DEFINE_F(StoreBenchmark, Get_NonExistent)(benchmark::State& state) {
  OpenStore();
  std::string value;
  int64_t key_id = 0;

  for (auto _ : state) {
    std::string key = "nonexistent_" + std::to_string(key_id++);
    auto status = store_->Get(key, &value);
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, Get_NonExistent);

BENCHMARK_DEFINE_F(StoreBenchmark, Delete_Existing)(benchmark::State& state) {
  OpenStore();

  // Pre-populate
  for (int i = 0; i < 10000; ++i) {
    store_->Put("del_key_" + std::to_string(i), "value");
  }

  int64_t key_id = 0;
  for (auto _ : state) {
    std::string key = "del_key_" + std::to_string(key_id++);
    auto status = store_->Delete(key);
    benchmark::DoNotOptimize(status);
    if (key_id >= 10000) {
      state.PauseTiming();
      key_id = 0;
      for (int i = 0; i < 10000; ++i) {
        store_->Put("del_key_" + std::to_string(i), "value");
      }
      state.ResumeTiming();
    }
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, Delete_Existing);

BENCHMARK_DEFINE_F(StoreBenchmark, MixedReadWrite)(benchmark::State& state) {
  OpenStore();

  const int kNumKeys = 1000;
  const int kNumOps = 10000;

  // Pre-generate keys
  std::vector<std::string> read_keys;
  read_keys.reserve(kNumKeys);
  for (int i = 0; i < kNumKeys; ++i) {
    std::string key = "key_" + std::to_string(i);
    read_keys.push_back(key);
    store_->Put(key, "value_" + std::to_string(i));
  }

  // Pre-generate operation sequence (70% reads, 30% writes)
  struct Op {
    bool is_read;
    int key_idx;  // For reads: index into read_keys; For writes: suffix for new key
  };
  std::vector<Op> ops;
  ops.reserve(kNumOps);

  std::uniform_int_distribution<> key_dis(0, kNumKeys - 1);
  std::uniform_int_distribution<> op_dis(0, 9);
  int write_key_id = kNumKeys;
  for (int i = 0; i < kNumOps; ++i) {
    Op op;
    op.is_read = (op_dis(gen_) < 7);
    op.key_idx = op.is_read ? key_dis(gen_) : write_key_id++;
    ops.push_back(op);
  }

  std::string value;
  size_t op_idx = 0;

  for (auto _ : state) {
    const Op& op = ops[op_idx];
    if (op.is_read) {
      auto status = store_->Get(read_keys[op.key_idx], &value);
      benchmark::DoNotOptimize(status);
    } else {
      std::string key = "key_" + std::to_string(op.key_idx);
      auto status = store_->Put(key, "new_value");
      benchmark::DoNotOptimize(status);
    }
    op_idx = (op_idx + 1) % kNumOps;
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, MixedReadWrite);

BENCHMARK_DEFINE_F(StoreBenchmark, CountKeys)(benchmark::State& state) {
  OpenStore();

  // Pre-populate with specified number of keys
  int num_keys = state.range(0);
  for (int i = 0; i < num_keys; ++i) {
    store_->Put("key_" + std::to_string(i), "value");
  }

  uint64_t count;
  for (auto _ : state) {
    auto status = store_->CountKeys(&count);
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(count);
  }
  state.SetComplexityN(num_keys);
}
BENCHMARK_REGISTER_F(StoreBenchmark, CountKeys)->Range(100, 10000)->Complexity();

BENCHMARK_DEFINE_F(StoreBenchmark, CountKeysApprox)(benchmark::State& state) {
  OpenStore();

  // Pre-populate
  for (int i = 0; i < 10000; ++i) {
    store_->Put("key_" + std::to_string(i), "value");
  }

  uint64_t count;
  for (auto _ : state) {
    auto status = store_->CountKeysApprox(&count);
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(count);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, CountKeysApprox);

BENCHMARK_DEFINE_F(StoreBenchmark, ListKeys)(benchmark::State& state) {
  OpenStore();

  // Pre-populate
  for (int i = 0; i < 1000; ++i) {
    store_->Put("key_" + std::to_string(i), "value");
  }

  std::vector<std::string> keys;
  for (auto _ : state) {
    keys.clear();
    auto status = store_->ListKeys(&keys, state.range(0));
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(keys);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, ListKeys)->Arg(10)->Arg(100)->Arg(1000);

// =============================================================================
// Deduplication Benchmarks
// =============================================================================

BENCHMARK_DEFINE_F(StoreBenchmark, DedupRatio_High)(benchmark::State& state) {
  OpenStore();

  // All keys share the same value (100% dedup)
  std::string shared_value(1000, 'x');
  int64_t key_id = 0;

  for (auto _ : state) {
    std::string key = "dedup_key_" + std::to_string(key_id++);
    auto status = store_->Put(key, shared_value);
    benchmark::DoNotOptimize(status);
  }

  state.counters["dedup_ratio"] = benchmark::Counter(
      key_id,  // keys / unique values = keys / 1
      benchmark::Counter::kDefaults);
}
BENCHMARK_REGISTER_F(StoreBenchmark, DedupRatio_High);

BENCHMARK_DEFINE_F(StoreBenchmark, DedupRatio_Low)(benchmark::State& state) {
  OpenStore();

  // All values are unique (0% dedup)
  int64_t key_id = 0;

  for (auto _ : state) {
    std::string key = "unique_key_" + std::to_string(key_id);
    std::string value = "unique_value_" + std::to_string(key_id++);
    auto status = store_->Put(key, value);
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, DedupRatio_Low);

// =============================================================================
// Cache Operation Benchmarks
// =============================================================================

BENCHMARK_DEFINE_F(StoreBenchmark, Sweep)(benchmark::State& state) {
  prestige::Options opt;
  opt.default_ttl_seconds = 0;  // No TTL for consistent benchmark
  OpenStore(opt);

  // Pre-populate
  for (int i = 0; i < 1000; ++i) {
    store_->Put("key_" + std::to_string(i), "value");
  }

  uint64_t deleted;
  for (auto _ : state) {
    auto status = store_->Sweep(&deleted);
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, Sweep);

BENCHMARK_DEFINE_F(StoreBenchmark, GetHealth)(benchmark::State& state) {
  OpenStore();

  // Pre-populate
  for (int i = 0; i < 1000; ++i) {
    store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i % 100));
  }

  prestige::HealthStats stats;
  for (auto _ : state) {
    auto status = store_->GetHealth(&stats);
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(stats);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, GetHealth);

// =============================================================================
// Normalization Mode Benchmarks
// =============================================================================

BENCHMARK_DEFINE_F(StoreBenchmark, Put_WithNormalization)(benchmark::State& state) {
  prestige::Options opt;
  opt.normalization_mode = prestige::NormalizationMode::kASCII;
  OpenStore(opt);

  int64_t key_id = 0;
  for (auto _ : state) {
    std::string key = "key_" + std::to_string(key_id++);
    std::string value = "UPPERCASE VALUE TEXT FOR NORMALIZATION";
    auto status = store_->Put(key, value);
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, Put_WithNormalization);

// =============================================================================
// Concurrent Access Benchmarks
// Note: Multi-threaded benchmarks with fixtures require careful synchronization.
// These benchmarks measure single-threaded throughput which is sufficient for
// regression tracking. For multi-threaded testing, see integration_test.cpp.
// =============================================================================

BENCHMARK_DEFINE_F(StoreBenchmark, SequentialPuts)(benchmark::State& state) {
  OpenStore();

  int64_t key_id = 0;
  for (auto _ : state) {
    std::string key = "seq_key_" + std::to_string(key_id++);
    auto status = store_->Put(key, "value");
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, SequentialPuts);

BENCHMARK_DEFINE_F(StoreBenchmark, SequentialReads)(benchmark::State& state) {
  OpenStore();

  const int kNumKeys = 10000;
  const int kNumLookups = 10000;

  // Pre-generate keys and populate store
  std::vector<std::string> keys;
  keys.reserve(kNumKeys);
  for (int i = 0; i < kNumKeys; ++i) {
    std::string key = "read_key_" + std::to_string(i);
    keys.push_back(key);
    store_->Put(key, "value");
  }

  // Pre-generate random lookup sequence
  std::vector<int> lookup_indices;
  lookup_indices.reserve(kNumLookups);
  std::uniform_int_distribution<> dis(0, kNumKeys - 1);
  for (int i = 0; i < kNumLookups; ++i) {
    lookup_indices.push_back(dis(gen_));
  }

  std::string value;
  size_t lookup_idx = 0;

  for (auto _ : state) {
    auto status = store_->Get(keys[lookup_indices[lookup_idx]], &value);
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(value);
    lookup_idx = (lookup_idx + 1) % kNumLookups;
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, SequentialReads);

}  // namespace

BENCHMARK_MAIN();
