// Performance benchmarks for prestige Store
// Uses Google Benchmark for accurate measurement and CI regression tracking

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
    test_dir_ = std::filesystem::temp_directory_path() / ("prestige_bench_" + RandomSuffix());
    std::filesystem::create_directories(test_dir_);
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
// Store Operation Benchmarks
// =============================================================================

BENCHMARK_DEFINE_F(StoreBenchmark, Put_UniqueValues)(benchmark::State& state) {
  OpenStore();
  int64_t key_id = 0;

  for (auto _ : state) {
    std::string key = "key_" + std::to_string(key_id++);
    std::string value = RandomString(state.range(0));
    auto status = store_->Put(key, value);
    benchmark::DoNotOptimize(status);
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

  // Pre-populate
  for (int i = 0; i < 1000; ++i) {
    store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i));
  }

  std::uniform_int_distribution<> dis(0, 999);
  std::string value;

  for (auto _ : state) {
    std::string key = "key_" + std::to_string(dis(gen_));
    auto status = store_->Get(key, &value);
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(value);
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

  // Pre-populate
  for (int i = 0; i < 1000; ++i) {
    store_->Put("key_" + std::to_string(i), "value_" + std::to_string(i));
  }

  std::uniform_int_distribution<> key_dis(0, 999);
  std::uniform_int_distribution<> op_dis(0, 9);  // 70% reads, 30% writes
  std::string value;
  int64_t new_key_id = 1000;

  for (auto _ : state) {
    if (op_dis(gen_) < 7) {
      // Read
      std::string key = "key_" + std::to_string(key_dis(gen_));
      auto status = store_->Get(key, &value);
      benchmark::DoNotOptimize(status);
    } else {
      // Write
      std::string key = "key_" + std::to_string(new_key_id++);
      auto status = store_->Put(key, "new_value");
      benchmark::DoNotOptimize(status);
    }
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
// =============================================================================

BENCHMARK_DEFINE_F(StoreBenchmark, ConcurrentPuts)(benchmark::State& state) {
  OpenStore();

  int64_t key_id = 0;
  for (auto _ : state) {
    std::string key = "concurrent_key_" + std::to_string(key_id++);
    auto status = store_->Put(key, "value");
    benchmark::DoNotOptimize(status);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, ConcurrentPuts)->Threads(1)->Threads(2)->Threads(4)->Threads(8);

BENCHMARK_DEFINE_F(StoreBenchmark, ConcurrentReads)(benchmark::State& state) {
  if (state.thread_index() == 0) {
    OpenStore();
    // Pre-populate
    for (int i = 0; i < 10000; ++i) {
      store_->Put("read_key_" + std::to_string(i), "value");
    }
  }

  std::uniform_int_distribution<> dis(0, 9999);
  std::string value;

  for (auto _ : state) {
    std::string key = "read_key_" + std::to_string(dis(gen_));
    auto status = store_->Get(key, &value);
    benchmark::DoNotOptimize(status);
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK_REGISTER_F(StoreBenchmark, ConcurrentReads)->Threads(1)->Threads(2)->Threads(4)->Threads(8);

}  // namespace

BENCHMARK_MAIN();
