#include <prestige/store.hpp>

#include <rocksdb/cache.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/statistics.h>
#include <rocksdb/table.h>
#include <rocksdb/utilities/transaction.h>

// RocksDB 7.x moved Cache methods to advanced_cache.h
// Check if it exists (RocksDB 7+) or use version detection
#if __has_include(<rocksdb/advanced_cache.h>)
#include <rocksdb/advanced_cache.h>
#define PRESTIGE_HAS_CACHE_METRICS 1
#else
#define PRESTIGE_HAS_CACHE_METRICS 0
#endif

#include <cstring>
#include <random>
#include <thread>
#include <unordered_set>

#include <prestige/internal.hpp>
#include <prestige/normalize.hpp>
#include <prestige/test_utils.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC
#include <prestige/embedder.hpp>
#include <prestige/vector_index.hpp>
#include <prestige/reranker.hpp>
#include <prestige/judge_llm.hpp>
#endif

namespace prestige {

namespace {

constexpr const char* kUserKvCF      = "prestige_user_kv";
constexpr const char* kObjectStoreCF = "prestige_object_store";
constexpr const char* kDedupIndexCF  = "prestige_dedup_index";
constexpr const char* kRefcountCF    = "prestige_refcount";
constexpr const char* kObjectMetaCF  = "prestige_object_meta";
constexpr const char* kLRUIndexCF    = "prestige_lru_index";
constexpr const char* kDecisionsCF   = "prestige_decisions";
#ifdef PRESTIGE_ENABLE_SEMANTIC
constexpr const char* kEmbeddingsCF  = "prestige_embeddings";
constexpr const char* kVectorPendingCF = "prestige_vector_pending";
#endif

// --------------------------
// Observability helpers
// --------------------------
inline void EmitCounter(const prestige::Options& opt,
                        std::string_view name,
                        uint64_t delta = 1) {
  if (opt.metrics) opt.metrics->Counter(name, delta);
}

inline void EmitHistogram(const prestige::Options& opt,
                          std::string_view name,
                          uint64_t value) {
  if (opt.metrics) opt.metrics->Histogram(name, value);
}

inline void EmitGauge(const prestige::Options& opt,
                      std::string_view name,
                      double value) {
  if (opt.metrics) opt.metrics->Gauge(name, value);
}

// Map RocksDB statuses to low-cardinality strings for tracing.
// (Avoid putting status.ToString() into attributes; it's high-cardinality.)
inline std::string_view StatusKind(const rocksdb::Status& s) {
  if (s.ok()) return "ok";
  if (s.IsNotFound()) return "not_found";
  if (s.IsInvalidArgument()) return "invalid_argument";
  if (s.IsTimedOut()) return "timed_out";
  if (s.IsBusy()) return "busy";
  if (s.IsTryAgain()) return "try_again";
  if (s.IsAborted()) return "aborted";
  if (s.IsCorruption()) return "corruption";
  if (s.IsIOError()) return "io_error";
  return "other";
}

inline void SpanAttr(prestige::TraceSpan* span,
                     std::string_view key,
                     uint64_t value) {
  if (span) span->SetAttribute(key, value);
}

inline void SpanAttr(prestige::TraceSpan* span,
                     std::string_view key,
                     std::string_view value) {
  if (span) span->SetAttribute(key, value);
}

inline void SpanEvent(prestige::TraceSpan* span, std::string_view name) {
  if (span) span->AddEvent(name);
}
  
// Helper: build a ColumnFamilyOptions with shared cache + bloom
rocksdb::ColumnFamilyOptions MakeCFOptions(const std::shared_ptr<rocksdb::Cache>& cache,
                                           int bloom_bits_per_key) {
  rocksdb::BlockBasedTableOptions table;
  table.block_cache = cache;
  table.filter_policy.reset(rocksdb::NewBloomFilterPolicy(bloom_bits_per_key, false));
  table.whole_key_filtering = true;

  rocksdb::ColumnFamilyOptions cfo;
  cfo.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table));
  return cfo;
}

// Compute retry delay with exponential backoff and jitter.
// Formula: min(max_delay, base_delay * 2^attempt) * random(1 ± jitter/2)
// This prevents thundering herd when multiple transactions retry simultaneously.
inline uint64_t ComputeRetryDelay(const prestige::Options& opt, int attempt) {
  // Exponential backoff: base * 2^attempt, capped at max
  uint64_t delay_us = opt.retry_base_delay_us * (1ULL << attempt);
  delay_us = std::min(delay_us, opt.retry_max_delay_us);

  // Add jitter: multiply by random factor in [1-j/2, 1+j/2]
  // Thread-local RNG for efficiency
  thread_local std::mt19937_64 rng([] {
    std::random_device rd;
    return rd();
  }());

  double jitter_min = 1.0 - opt.retry_jitter_factor / 2.0;
  double jitter_max = 1.0 + opt.retry_jitter_factor / 2.0;
  std::uniform_real_distribution<double> dist(jitter_min, jitter_max);

  return static_cast<uint64_t>(delay_us * dist(rng));
}

// Sleep for the computed backoff duration and emit metrics.
inline void BackoffBeforeRetry(const prestige::Options& opt, int attempt,
                               prestige::TraceSpan* span) {
  uint64_t delay_us = ComputeRetryDelay(opt, attempt);
  std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
  EmitHistogram(opt, "prestige.txn.backoff_us", delay_us);
  if (span) {
    span->SetAttribute("backoff_us", delay_us);
  }
}

// --------------------------
// Decision record encoding
// --------------------------
// One column family holds four kinds of keys, told apart by their first byte:
//   'd' + decision_id        -> serialized DecisionRecord
//   's' + u64be(sequence)    -> decision_id            (write order; the sweep walks this)
//   'q' + u64be(sequence)    -> decision_id            (repair queue; priority entries from read-time misses)
//   'c'                      -> u64be(sequence)        (last sequence the full walk has verified)
//   'm' + name               -> u64le counter          (dangling_reads, repairs)
constexpr char kDecRecord = 'd';
constexpr char kDecSeq = 's';
constexpr char kDecQueue = 'q';
constexpr char kDecCursor = 'c';
constexpr char kDecCounter = 'm';
constexpr size_t kDecMaxField = 4096;
constexpr size_t kDecMaxNote = 65536;
constexpr size_t kDecMaxInputs = 65536;

inline std::string EncodeU64BE(uint64_t v) {
  std::string out(8, '\0');
  for (int i = 7; i >= 0; --i) { out[static_cast<size_t>(i)] = static_cast<char>(v & 0xff); v >>= 8; }
  return out;
}
inline bool DecodeU64BE(std::string_view s, uint64_t* out) {
  if (s.size() != 8) return false;
  uint64_t v = 0;
  for (size_t i = 0; i < 8; ++i) v = (v << 8) | static_cast<uint8_t>(s[i]);
  *out = v;
  return true;
}
inline std::string DecRecordKey(std::string_view id) { std::string k(1, kDecRecord); k.append(id); return k; }
inline std::string DecSeqKey(uint64_t seq) { return std::string(1, kDecSeq) + EncodeU64BE(seq); }
inline std::string DecQueueKey(uint64_t seq) { return std::string(1, kDecQueue) + EncodeU64BE(seq); }
inline std::string DecCursorKey() { return std::string(1, kDecCursor); }
inline std::string DecCounterKey(std::string_view name) { std::string k(1, kDecCounter); k.append(name); return k; }

inline void AppendU32LE(std::string* out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out->push_back(static_cast<char>((v >> (8 * i)) & 0xff));
}
inline bool ReadU32LE(std::string_view s, size_t* off, uint32_t* v) {
  if (*off + 4 > s.size()) return false;
  uint32_t r = 0;
  for (int i = 3; i >= 0; --i) r = (r << 8) | static_cast<uint8_t>(s[*off + static_cast<size_t>(i)]);
  *off += 4; *v = r;
  return true;
}
inline void AppendStr(std::string* out, std::string_view v) {
  AppendU32LE(out, static_cast<uint32_t>(v.size()));
  out->append(v.data(), v.size());
}
inline bool ReadStr(std::string_view s, size_t* off, std::string* v) {
  uint32_t n = 0;
  if (!ReadU32LE(s, off, &n)) return false;
  if (*off + n > s.size()) return false;
  v->assign(s.data() + *off, n);
  *off += n;
  return true;
}

constexpr uint8_t kDecRecordVersion = 1;

std::string SerializeDecisionRecord(const prestige::DecisionRecord& r) {
  std::string out;
  out.push_back(static_cast<char>(kDecRecordVersion));
  out.append(prestige::internal::EncodeU64LE(r.sequence));
  out.append(prestige::internal::EncodeU64LE(r.committed_at_us));
  AppendStr(&out, r.decision.decision_id);
  AppendStr(&out, r.decision.policy_revision);
  AppendStr(&out, r.decision.parent_decision_id);
  AppendStr(&out, r.decision.note);
  AppendStr(&out, r.user_key);
  AppendStr(&out, r.output_digest);
  AppendU32LE(&out, static_cast<uint32_t>(r.decision.input_digests.size()));
  for (const auto& d : r.decision.input_digests) AppendStr(&out, d);
  return out;
}

bool DeserializeDecisionRecord(std::string_view s, prestige::DecisionRecord* r) {
  if (s.size() < 17 || static_cast<uint8_t>(s[0]) != kDecRecordVersion) return false;
  size_t off = 1;
  if (!prestige::internal::DecodeU64LE(s.substr(off, 8), &r->sequence)) return false;
  off += 8;
  if (!prestige::internal::DecodeU64LE(s.substr(off, 8), &r->committed_at_us)) return false;
  off += 8;
  if (!ReadStr(s, &off, &r->decision.decision_id)) return false;
  if (!ReadStr(s, &off, &r->decision.policy_revision)) return false;
  if (!ReadStr(s, &off, &r->decision.parent_decision_id)) return false;
  if (!ReadStr(s, &off, &r->decision.note)) return false;
  if (!ReadStr(s, &off, &r->user_key)) return false;
  if (!ReadStr(s, &off, &r->output_digest)) return false;
  uint32_t n = 0;
  if (!ReadU32LE(s, &off, &n)) return false;
  if (n > kDecMaxInputs) return false;
  r->decision.input_digests.clear();
  r->decision.input_digests.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    std::string d;
    if (!ReadStr(s, &off, &d)) return false;
    r->decision.input_digests.push_back(std::move(d));
  }
  return off == s.size();
}

// Read a persisted counter outside a transaction.
inline uint64_t ReadDecisionCounter(rocksdb::TransactionDB* db, rocksdb::ColumnFamilyHandle* cf,
                                    const rocksdb::ReadOptions& ro, std::string_view name) {
  std::string raw;
  uint64_t v = 0;
  if (db->Get(ro, cf, rocksdb::Slice(DecCounterKey(name)), &raw).ok()) {
    prestige::internal::DecodeU64LE(raw, &v);
  }
  return v;
}

// Increment a persisted counter inside a transaction (locks the key).
inline rocksdb::Status BumpDecisionCounterLocked(rocksdb::Transaction* txn, rocksdb::ColumnFamilyHandle* cf,
                                                 std::string_view name, uint64_t delta) {
  rocksdb::ReadOptions ro;
  std::string raw;
  uint64_t v = 0;
  const std::string key = DecCounterKey(name);
  rocksdb::Status s = txn->GetForUpdate(ro, cf, rocksdb::Slice(key), &raw);
  if (s.ok()) {
    if (!prestige::internal::DecodeU64LE(raw, &v)) return rocksdb::Status::Corruption("decision counter is not uint64_le");
  } else if (!s.IsNotFound()) {
    return s;
  }
  return txn->Put(cf, rocksdb::Slice(key), rocksdb::Slice(prestige::internal::EncodeU64LE(v + delta)));
}

inline uint64_t CountDecisionPrefix(rocksdb::TransactionDB* db, rocksdb::ColumnFamilyHandle* cf,
                                    const rocksdb::ReadOptions& ro, char prefix) {
  uint64_t n = 0;
  std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(ro, cf));
  for (it->Seek(rocksdb::Slice(&prefix, 1)); it->Valid() && it->key().size() > 0 && it->key()[0] == prefix; it->Next()) ++n;
  return n;
}

}  // namespace

// Forward declaration for cache management methods
static rocksdb::Status DeleteObjectIfUnreferencedLocked(rocksdb::Transaction* txn,
                                                        rocksdb::ColumnFamilyHandle* objects_cf,
                                                        rocksdb::ColumnFamilyHandle* dedup_cf,
                                                        rocksdb::ColumnFamilyHandle* refcount_cf,
                                                        rocksdb::ColumnFamilyHandle* meta_cf,
                                                        rocksdb::ColumnFamilyHandle* lru_cf,
                                                        std::atomic<uint64_t>* total_store_bytes,
                                                        const std::string& obj_id);

Store::Store(const Options& opt) : opt_(opt) {}

Store::~Store() { Close(); }

uint64_t Store::GetWallClockMicros() const {
  if (opt_.custom_clock) {
    return opt_.custom_clock->WallClockMicros();
  }
  return internal::WallClockMicros();
}

rocksdb::Status Store::Open(const std::string& db_path,
                            std::unique_ptr<Store>* out,
                            const Options& opt) {
  if (!out) return rocksdb::Status::InvalidArgument("out is null");

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Validate semantic mode options
  if (opt.dedup_mode == DedupMode::kSemantic) {
    if (opt.semantic_model_path.empty() && !opt.custom_embedder) {
      return rocksdb::Status::InvalidArgument(
          "semantic_model_path or custom_embedder is required when dedup_mode == kSemantic");
    }
    if (opt.semantic_threshold < 0.0f || opt.semantic_threshold > 1.0f) {
      return rocksdb::Status::InvalidArgument(
          "semantic_threshold must be in range [0.0, 1.0] when dedup_mode == kSemantic");
    }
  }
#else
  // Semantic mode requested but not compiled in
  if (opt.dedup_mode == DedupMode::kSemantic) {
    return rocksdb::Status::InvalidArgument(
        "Semantic dedup not enabled. Rebuild with PRESTIGE_ENABLE_SEMANTIC=ON");
  }
#endif

  auto store = std::unique_ptr<Store>(new Store(opt));

  // RocksDB options
  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  // Enable statistics for cache observability
  auto statistics = rocksdb::CreateDBStatistics();
  options.statistics = statistics;

  rocksdb::TransactionDBOptions txn_opts;

  // Shared cache for all CFs
  auto cache = rocksdb::NewLRUCache(opt.block_cache_bytes);

  // Store cache and statistics for later observability
  store->block_cache_ = cache;
  store->statistics_ = statistics;

  std::vector<rocksdb::ColumnFamilyDescriptor> cfs;
  cfs.emplace_back(rocksdb::kDefaultColumnFamilyName, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kUserKvCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kObjectStoreCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kDedupIndexCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kRefcountCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kObjectMetaCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kLRUIndexCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kDecisionsCF, MakeCFOptions(cache, opt.bloom_bits_per_key));

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Add embeddings and vector pending CFs for semantic mode
  if (opt.dedup_mode == DedupMode::kSemantic) {
    cfs.emplace_back(kEmbeddingsCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
    cfs.emplace_back(kVectorPendingCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  }
#endif

  std::vector<rocksdb::ColumnFamilyHandle*> handles;
  rocksdb::TransactionDB* db = nullptr;

  rocksdb::Status s = rocksdb::TransactionDB::Open(options, txn_opts, db_path, cfs, &handles, &db);
  if (!s.ok()) {
    for (auto* h : handles) delete h;
    return s;
  }

  store->db_ = db;
  store->handles_ = std::move(handles);

  // Descriptor order = handle order
  store->user_kv_cf_ = store->handles_[1];
  store->objects_cf_ = store->handles_[2];
  store->dedup_cf_   = store->handles_[3];
  store->refcount_cf_= store->handles_[4];
  store->meta_cf_    = store->handles_[5];
  store->lru_cf_     = store->handles_[6];
  store->decisions_cf_ = store->handles_[7];

  // Continue decision sequence numbers after the highest one on disk
  {
    rocksdb::ReadOptions ro;
    std::unique_ptr<rocksdb::Iterator> it(store->db_->NewIterator(ro, store->decisions_cf_));
    it->SeekForPrev(rocksdb::Slice(std::string(1, kDecSeq) + std::string(8, '\xff')));
    uint64_t last = 0;
    if (it->Valid() && it->key().size() == 9 && it->key()[0] == kDecSeq) {
      DecodeU64BE(std::string_view(it->key().data() + 1, 8), &last);
    }
    store->next_decision_seq_.store(last + 1);
  }

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Initialize semantic dedup components
  if (opt.dedup_mode == DedupMode::kSemantic) {
    store->embeddings_cf_ = store->handles_[8];
    store->vector_pending_cf_ = store->handles_[9];

    // Convert device option (used for both embedder and reranker)
    internal::InferenceDevice device_type = internal::InferenceDevice::kAuto;
    switch (opt.semantic_device) {
      case SemanticDevice::kCPU:
        device_type = internal::InferenceDevice::kCPU;
        break;
      case SemanticDevice::kGPU:
        device_type = internal::InferenceDevice::kGPU;
        break;
      case SemanticDevice::kAuto:
        device_type = internal::InferenceDevice::kAuto;
        break;
    }

    // Create or use provided embedder
    if (opt.custom_embedder) {
      // Use the custom embedder (takes ownership)
      store->embedder_.reset(opt.custom_embedder);
    } else {
      // Create ONNX embedder from model path
      std::string embedder_error;
      internal::EmbedderModelType embedder_type;
      switch (opt.semantic_model_type) {
        case SemanticModel::kMiniLM:
          embedder_type = internal::EmbedderModelType::kMiniLM;
          break;
        case SemanticModel::kBGESmall:
          embedder_type = internal::EmbedderModelType::kBGESmall;
          break;
        case SemanticModel::kBGELarge:
          embedder_type = internal::EmbedderModelType::kBGELarge;
          break;
        case SemanticModel::kE5Large:
          embedder_type = internal::EmbedderModelType::kE5Large;
          break;
        case SemanticModel::kBGEM3:
          embedder_type = internal::EmbedderModelType::kBGEM3;
          break;
        case SemanticModel::kNomicEmbed:
          embedder_type = internal::EmbedderModelType::kNomicEmbed;
          break;
      }
      // Convert pooling option
      internal::EmbedderPooling pooling_type = internal::EmbedderPooling::kMean;
      if (opt.semantic_pooling == SemanticPooling::kCLS) {
        pooling_type = internal::EmbedderPooling::kCLS;
      }

      store->embedder_ = internal::Embedder::Create(
          opt.semantic_model_path,
          embedder_type,
          opt.semantic_num_threads,
          pooling_type,
          device_type,
          &embedder_error);

      if (!store->embedder_) {
        store->Close();
        return rocksdb::Status::InvalidArgument(
            "Failed to create embedder: " + embedder_error);
      }
    }

    // Create vector index with optional capacity limit
    size_t dimension = store->embedder_->Dimension();
    store->vector_index_ = internal::CreateHNSWIndex(
        dimension,
        10000,  // Initial max elements (will grow as needed)
        opt.hnsw_m,
        opt.hnsw_ef_construction,
        opt.semantic_max_index_entries);

    if (!store->vector_index_) {
      store->Close();
      return rocksdb::Status::InvalidArgument("Failed to create vector index");
    }

    store->vector_index_->SetSearchParam("ef_search", opt.hnsw_ef_search);

    // Set max entries limit if specified (for memory safety)
    if (opt.semantic_max_index_entries > 0) {
      store->vector_index_->SetMaxEntries(opt.semantic_max_index_entries);
    }

    // Initialize reranker if enabled
    if (opt.semantic_reranker_enabled) {
      if (opt.custom_reranker) {
        // Use provided custom reranker (takes ownership)
        store->reranker_.reset(opt.custom_reranker);
      } else if (!opt.semantic_reranker_model_path.empty()) {
        // Create BGE reranker from model path
        std::string reranker_error;
        store->reranker_ = internal::CreateReranker(
            opt.semantic_reranker_model_path,
            opt.semantic_reranker_num_threads,
            device_type,
            &reranker_error);

        if (!store->reranker_ && !opt.semantic_reranker_fallback) {
          store->Close();
          return rocksdb::Status::InvalidArgument(
              "Failed to create reranker: " + reranker_error);
        }
      }
    }

    // Initialize judge LLM if enabled (for gray zone evaluation)
    if (opt.semantic_judge_enabled) {
      if (opt.custom_judge) {
        // Use provided custom judge (takes ownership)
        store->judge_llm_.reset(opt.custom_judge);
      } else if (!opt.semantic_judge_model_path.empty()) {
        // Create Prometheus 2 judge from model path
        std::string judge_error;
        store->judge_llm_ = internal::CreateJudgeLLM(
            opt.semantic_judge_model_path,
            opt.semantic_judge_num_threads,
            opt.semantic_judge_context_size,
            opt.semantic_judge_gpu_layers,
            opt.semantic_judge_max_tokens,
            opt.semantic_judge_min_score,
            &judge_error);

        if (!store->judge_llm_) {
          store->Close();
          return rocksdb::Status::InvalidArgument(
              "Failed to create judge LLM: " + judge_error);
        }
      }
    }

    // Load existing index if present
    store->vector_index_path_ = db_path + ".vec_index";
    if (!store->vector_index_->Load(store->vector_index_path_)) {
      // Load failure is not fatal if the file doesn't exist yet
      // But if it exists and is corrupt, we should warn (for now just continue)
    }

    // Replay any pending vector index operations from previous run (crash recovery)
    store->ReplayPendingVectorOps();
  }
#endif

  *out = std::move(store);
  return rocksdb::Status::OK();
}

rocksdb::Status Store::Put(std::string_view user_key, std::string_view value_bytes) {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  return PutImpl(user_key, value_bytes);
}

rocksdb::Status Store::Get(std::string_view user_key, std::string* value_bytes_out) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!value_bytes_out) return rocksdb::Status::InvalidArgument("value_bytes_out is null");

  EmitCounter(opt_, "prestige.get.calls", 1);

  const uint64_t op_start_us = prestige::internal::NowMicros();
  std::unique_ptr<TraceSpan> span;
  if (opt_.tracer) span = opt_.tracer->StartSpan("prestige.Get");
  SpanAttr(span.get(), "key_bytes", static_cast<uint64_t>(user_key.size()));

  auto finish = [&](const rocksdb::Status& st) -> rocksdb::Status {
    const uint64_t dur_us = prestige::internal::NowMicros() - op_start_us;
    EmitHistogram(opt_, "prestige.get.latency_us", dur_us);

    if (st.ok()) {
      EmitCounter(opt_, "prestige.get.ok_total", 1);
    } else if (st.IsNotFound()) {
      EmitCounter(opt_, "prestige.get.not_found_total", 1);
    } else {
      EmitCounter(opt_, "prestige.get.error_total", 1);
    }

    if (span) {
      SpanAttr(span.get(), "latency_us", dur_us);
      SpanAttr(span.get(), "status", StatusKind(st));
      span->End(st);
    }
    return st;
  };

  rocksdb::ReadOptions ro;

  // 1) user_key -> object_id
  std::string obj_id;
  const uint64_t map_start_us = prestige::internal::NowMicros();
  rocksdb::Status s = db_->Get(
      ro, user_kv_cf_,
      rocksdb::Slice(user_key.data(), user_key.size()),
      &obj_id);
  EmitHistogram(opt_, "prestige.get.user_lookup_us",
                prestige::internal::NowMicros() - map_start_us);
  if (!s.ok()) return finish(s);

  // Check TTL if enabled
  if (opt_.default_ttl_seconds > 0) {
    std::string meta_raw;
    rocksdb::Status ms = db_->Get(ro, meta_cf_, rocksdb::Slice(obj_id), &meta_raw);
    if (ms.ok()) {
      prestige::internal::ObjectMeta meta;
      if (prestige::internal::ObjectMeta::Deserialize(meta_raw, &meta) && !meta.IsLegacy()) {
        uint64_t now_us = GetWallClockMicros();
        uint64_t age_us = now_us - meta.created_at_us;
        uint64_t ttl_us = opt_.default_ttl_seconds * 1000000ULL;
        if (age_us > ttl_us) {
          EmitCounter(opt_, "prestige.get.expired_total", 1);
          return finish(rocksdb::Status::NotFound("Object expired"));
        }
      }
    }
  }

  // 2) object_id -> value_bytes
  const uint64_t obj_start_us = prestige::internal::NowMicros();
  rocksdb::Status s2 = db_->Get(ro, objects_cf_, rocksdb::Slice(obj_id), value_bytes_out);
  EmitHistogram(opt_, "prestige.get.object_lookup_us",
                prestige::internal::NowMicros() - obj_start_us);

  if (s2.ok()) {
    EmitHistogram(opt_, "prestige.get.value_bytes",
                  static_cast<uint64_t>(value_bytes_out->size()));

    // Update last_accessed_us for LRU tracking (if enabled)
    if (opt_.track_access_time) {
      std::string meta_raw;
      rocksdb::Status ms = db_->Get(ro, meta_cf_, rocksdb::Slice(obj_id), &meta_raw);
      if (ms.ok()) {
        prestige::internal::ObjectMeta meta;
        if (prestige::internal::ObjectMeta::Deserialize(meta_raw, &meta) && !meta.IsLegacy()) {
          uint64_t old_access_us = meta.last_accessed_us;
          uint64_t now_us = GetWallClockMicros();

          // Only update LRU if enough time has passed since last update.
          // This reduces write amplification for read-heavy workloads.
          // With lru_update_interval_seconds=3600, an object accessed 10K times/hour
          // generates only 1 LRU write instead of 10K.
          uint64_t interval_us = opt_.lru_update_interval_seconds * 1000000ULL;
          uint64_t time_since_update = now_us - old_access_us;

          if (interval_us == 0 || time_since_update >= interval_us) {
            meta.last_accessed_us = now_us;

            // Update metadata and LRU index (best-effort, don't fail the Get)
            rocksdb::WriteBatch batch;
            batch.Put(meta_cf_, rocksdb::Slice(obj_id), rocksdb::Slice(meta.Serialize()));

            // Delete old LRU entry and add new one
            std::string old_lru_key = prestige::internal::MakeLRUKey(old_access_us, obj_id);
            std::string new_lru_key = prestige::internal::MakeLRUKey(now_us, obj_id);
            batch.Delete(lru_cf_, rocksdb::Slice(old_lru_key));
            batch.Put(lru_cf_, rocksdb::Slice(new_lru_key), rocksdb::Slice());

            rocksdb::WriteOptions wo;
            (void)db_->Write(wo, &batch);

            EmitCounter(opt_, "prestige.lru.update_total", 1);
          } else {
            EmitCounter(opt_, "prestige.lru.skip_total", 1);
          }
        }
      }
    }
  }

  return finish(s2);
}

rocksdb::Status Store::CountKeys(uint64_t* out_key_count) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out_key_count) return rocksdb::Status::InvalidArgument("out_key_count is null");

  const rocksdb::Snapshot* snapshot = db_->GetSnapshot();
  rocksdb::ReadOptions ro;
  ro.snapshot = snapshot;

  uint64_t count = 0;
  rocksdb::Status iter_status;
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, user_kv_cf_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      ++count;
    }
    iter_status = it->status();
  }

  db_->ReleaseSnapshot(snapshot);
  if (!iter_status.ok()) return iter_status;

  *out_key_count = count;
  return rocksdb::Status::OK();
}

rocksdb::Status Store::CountUniqueValues(uint64_t* out_unique_value_count) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out_unique_value_count) return rocksdb::Status::InvalidArgument("out_unique_value_count is null");

  const rocksdb::Snapshot* snapshot = db_->GetSnapshot();
  rocksdb::ReadOptions ro;
  ro.snapshot = snapshot;

  uint64_t count = 0;
  rocksdb::Status iter_status;
  bool decode_failed = false;
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, refcount_cf_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      uint64_t rc = 0;
      std::string_view v(it->value().data(), it->value().size());
      if (!prestige::internal::DecodeU64LE(v, &rc)) {
        decode_failed = true;
        break;
      }
      if (rc > 0) ++count;
    }
    iter_status = it->status();
  }

  db_->ReleaseSnapshot(snapshot);
  if (decode_failed) return rocksdb::Status::Corruption("refcount value is not uint64_le");
  if (!iter_status.ok()) return iter_status;

  *out_unique_value_count = count;
  return rocksdb::Status::OK();
}

rocksdb::Status Store::CountKeysApprox(uint64_t* out_key_count) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out_key_count) return rocksdb::Status::InvalidArgument("out_key_count is null");

  // Use RocksDB's internal estimate - O(1) but approximate.
  // Can be 10-50% off, especially after many deletes (tombstones not compacted).
  uint64_t estimate = 0;
  if (!db_->GetIntProperty(user_kv_cf_, "rocksdb.estimate-num-keys", &estimate)) {
    return rocksdb::Status::NotSupported("estimate-num-keys not available");
  }

  *out_key_count = estimate;
  return rocksdb::Status::OK();
}

rocksdb::Status Store::CountUniqueValuesApprox(uint64_t* out_unique_value_count) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out_unique_value_count) return rocksdb::Status::InvalidArgument("out_unique_value_count is null");

  // Use RocksDB's internal estimate for refcount CF - O(1) but approximate.
  // Note: This counts all refcount entries, including those with rc=0 (orphaned).
  // For exact count of live objects, use CountUniqueValues().
  uint64_t estimate = 0;
  if (!db_->GetIntProperty(refcount_cf_, "rocksdb.estimate-num-keys", &estimate)) {
    return rocksdb::Status::NotSupported("estimate-num-keys not available");
  }

  *out_unique_value_count = estimate;
  return rocksdb::Status::OK();
}

rocksdb::Status Store::GetTotalStoreBytesApprox(uint64_t* out_bytes) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out_bytes) return rocksdb::Status::InvalidArgument("out_bytes is null");

  // Sum live data size estimates across relevant column families.
  // This is O(1) but approximate - doesn't account for compression ratio accurately.
  uint64_t total = 0;
  uint64_t cf_size = 0;

  // Object store (the bulk of data)
  if (db_->GetIntProperty(objects_cf_, "rocksdb.estimate-live-data-size", &cf_size)) {
    total += cf_size;
  }

  // Metadata
  if (db_->GetIntProperty(meta_cf_, "rocksdb.estimate-live-data-size", &cf_size)) {
    total += cf_size;
  }

  // User key mappings
  if (db_->GetIntProperty(user_kv_cf_, "rocksdb.estimate-live-data-size", &cf_size)) {
    total += cf_size;
  }

  *out_bytes = total;
  return rocksdb::Status::OK();
}

rocksdb::Status Store::ListKeys(std::vector<std::string>* out_keys,
                                uint64_t limit,
                                std::string_view prefix) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out_keys) return rocksdb::Status::InvalidArgument("out_keys is null");

  out_keys->clear();

  const rocksdb::Snapshot* snapshot = db_->GetSnapshot();
  rocksdb::ReadOptions ro;
  ro.snapshot = snapshot;

  rocksdb::Slice prefix_slice(prefix.data(), prefix.size());

  rocksdb::Status iter_status;
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, user_kv_cf_));
    if (prefix.empty()) {
      it->SeekToFirst();
    } else {
      it->Seek(prefix_slice);
    }

    for (; it->Valid(); it->Next()) {
      if (!prefix.empty() && !it->key().starts_with(prefix_slice)) break;
      out_keys->emplace_back(it->key().data(), it->key().size());
      if (limit != 0 && out_keys->size() >= limit) break;
    }

    iter_status = it->status();
  }

  db_->ReleaseSnapshot(snapshot);
  if (!iter_status.ok()) return iter_status;

  return rocksdb::Status::OK();
}
  
rocksdb::Status Store::Delete(std::string_view user_key) {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  return DeleteImpl(user_key);
}

rocksdb::Status Store::GetObjectId(std::string_view user_key, std::string* object_id_out) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");

  // Look up the object ID from the user_key in the user_kv column family
  rocksdb::ReadOptions ro;
  std::string object_id;
  rocksdb::Status s = db_->Get(ro, user_kv_cf_, rocksdb::Slice(user_key), &object_id);

  if (s.ok()) {
    *object_id_out = object_id;
  }
  return s;
}

void Store::EmitCacheMetrics() {
  if (!opt_.metrics) return;

#if PRESTIGE_HAS_CACHE_METRICS
  // Cache fill rate and usage (requires RocksDB 7+ advanced_cache.h)
  if (block_cache_) {
    size_t usage = block_cache_->GetUsage();
    size_t capacity = block_cache_->GetCapacity();
    double fill_ratio = capacity > 0 ? static_cast<double>(usage) / capacity : 0.0;

    EmitGauge(opt_, "prestige.cache.fill_ratio", fill_ratio);
    EmitGauge(opt_, "prestige.cache.usage_bytes", static_cast<double>(usage));
    EmitGauge(opt_, "prestige.cache.capacity_bytes", static_cast<double>(capacity));
  }
#endif

  // Block cache hit/miss from RocksDB statistics
  if (statistics_) {
    uint64_t hits = statistics_->getTickerCount(rocksdb::BLOCK_CACHE_HIT);
    uint64_t misses = statistics_->getTickerCount(rocksdb::BLOCK_CACHE_MISS);

    // Emit deltas since last call
    if (hits >= last_cache_hits_) {
      EmitCounter(opt_, "prestige.cache.hit_total", hits - last_cache_hits_);
    }
    if (misses >= last_cache_misses_) {
      EmitCounter(opt_, "prestige.cache.miss_total", misses - last_cache_misses_);
    }

    last_cache_hits_ = hits;
    last_cache_misses_ = misses;

    // Bloom filter effectiveness
    uint64_t bloom_useful = statistics_->getTickerCount(rocksdb::BLOOM_FILTER_USEFUL);
    uint64_t bloom_checked = statistics_->getTickerCount(rocksdb::BLOOM_FILTER_PREFIX_CHECKED);
    EmitGauge(opt_, "prestige.bloom.useful_total", static_cast<double>(bloom_useful));
    EmitGauge(opt_, "prestige.bloom.checked_total", static_cast<double>(bloom_checked));
  }
}

uint64_t Store::GetTotalStoreBytes() const {
  return total_store_bytes_.load();
}

rocksdb::Status Store::Sweep(uint64_t* deleted_count) {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!deleted_count) return rocksdb::Status::InvalidArgument("deleted_count is null");

  *deleted_count = 0;
  uint64_t now_us = GetWallClockMicros();
  uint64_t ttl_us = opt_.default_ttl_seconds * 1000000ULL;

  const rocksdb::Snapshot* snapshot = db_->GetSnapshot();
  rocksdb::ReadOptions ro;
  ro.snapshot = snapshot;

  std::vector<std::string> to_delete;

  // Scan all objects
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, refcount_cf_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      std::string obj_id(it->key().data(), it->key().size());

      // Check refcount
      uint64_t refcount = 0;
      std::string_view v(it->value().data(), it->value().size());
      if (!prestige::internal::DecodeU64LE(v, &refcount)) {
        continue;  // Skip corrupted entries
      }

      // Orphaned object (refcount = 0)
      if (refcount == 0) {
        to_delete.push_back(obj_id);
        continue;
      }

      // Check TTL if enabled
      if (opt_.default_ttl_seconds > 0) {
        std::string meta_raw;
        rocksdb::Status ms = db_->Get(ro, meta_cf_, rocksdb::Slice(obj_id), &meta_raw);
        if (ms.ok()) {
          prestige::internal::ObjectMeta meta;
          if (prestige::internal::ObjectMeta::Deserialize(meta_raw, &meta) &&
              !meta.IsLegacy()) {
            if (now_us - meta.created_at_us > ttl_us) {
              to_delete.push_back(obj_id);
            }
          }
        }
      }
    }
  }

  db_->ReleaseSnapshot(snapshot);

  // Delete collected objects using transactions
  rocksdb::WriteOptions wo;
  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  for (const auto& obj_id : to_delete) {
    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) continue;

    rocksdb::Status s = DeleteObjectIfUnreferencedLocked(
        txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, lru_cf_,
        &total_store_bytes_, obj_id);

    if (s.ok()) {
      s = txn->Commit();
      if (s.ok()) {
        (*deleted_count)++;
      }
    }
  }

  EmitCounter(opt_, "prestige.sweep.deleted_total", *deleted_count);
  return rocksdb::Status::OK();
}

rocksdb::Status Store::Prune(uint64_t max_age_seconds,
                             uint64_t max_idle_seconds,
                             uint64_t* deleted_count) {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!deleted_count) return rocksdb::Status::InvalidArgument("deleted_count is null");

  *deleted_count = 0;
  uint64_t now_us = GetWallClockMicros();
  uint64_t max_age_us = max_age_seconds * 1000000ULL;
  uint64_t max_idle_us = max_idle_seconds * 1000000ULL;

  if (max_age_seconds == 0 && max_idle_seconds == 0) {
    return rocksdb::Status::OK();  // Nothing to prune
  }

  const rocksdb::Snapshot* snapshot = db_->GetSnapshot();
  rocksdb::ReadOptions ro;
  ro.snapshot = snapshot;

  std::vector<std::string> to_delete;

  // Scan all objects
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, refcount_cf_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      std::string obj_id(it->key().data(), it->key().size());

      std::string meta_raw;
      rocksdb::Status ms = db_->Get(ro, meta_cf_, rocksdb::Slice(obj_id), &meta_raw);
      if (!ms.ok()) continue;

      prestige::internal::ObjectMeta meta;
      if (!prestige::internal::ObjectMeta::Deserialize(meta_raw, &meta) ||
          meta.IsLegacy()) {
        continue;
      }

      bool should_delete = false;

      // Check age
      if (max_age_seconds > 0 && now_us - meta.created_at_us > max_age_us) {
        should_delete = true;
      }

      // Check idle time
      if (max_idle_seconds > 0 && now_us - meta.last_accessed_us > max_idle_us) {
        should_delete = true;
      }

      if (should_delete) {
        to_delete.push_back(obj_id);
      }
    }
  }

  db_->ReleaseSnapshot(snapshot);

  // Delete collected objects
  rocksdb::WriteOptions wo;
  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  for (const auto& obj_id : to_delete) {
    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) continue;

    rocksdb::Status s = DeleteObjectIfUnreferencedLocked(
        txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, lru_cf_,
        &total_store_bytes_, obj_id);

    if (s.ok()) {
      s = txn->Commit();
      if (s.ok()) {
        (*deleted_count)++;
      }
    }
  }

  EmitCounter(opt_, "prestige.prune.deleted_total", *deleted_count);
  return rocksdb::Status::OK();
}

rocksdb::Status Store::EvictLRU(uint64_t target_bytes, uint64_t* evicted_count) {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!evicted_count) return rocksdb::Status::InvalidArgument("evicted_count is null");

  *evicted_count = 0;

  // Check if eviction needed
  uint64_t current_bytes = total_store_bytes_.load();
  if (current_bytes <= target_bytes) {
    return rocksdb::Status::OK();  // Already under target
  }

  rocksdb::ReadOptions ro;
  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, lru_cf_));

  rocksdb::WriteOptions wo;
  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  // Iterate from oldest (smallest timestamp) to newest
  for (it->SeekToFirst();
       it->Valid() && total_store_bytes_.load() > target_bytes;
       it->Next()) {

    uint64_t timestamp_us;
    std::string obj_id;
    std::string_view key_view(it->key().data(), it->key().size());
    if (!prestige::internal::ParseLRUKey(key_view, &timestamp_us, &obj_id)) {
      continue;
    }

    // Check refcount - only evict if refcount > 0
    std::string refcount_raw;
    rocksdb::Status rs = db_->Get(ro, refcount_cf_, rocksdb::Slice(obj_id), &refcount_raw);
    if (!rs.ok()) continue;

    uint64_t refcount = 0;
    if (!prestige::internal::DecodeU64LE(refcount_raw, &refcount) || refcount == 0) {
      continue;  // Orphaned objects should be cleaned by Sweep
    }

    // Delete this object
    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) continue;

    rocksdb::Status s = DeleteObjectIfUnreferencedLocked(
        txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, lru_cf_,
        &total_store_bytes_, obj_id);

    if (s.ok()) {
      s = txn->Commit();
      if (s.ok()) {
        (*evicted_count)++;
      }
    }
  }

  EmitCounter(opt_, "prestige.evict.count", *evicted_count);
  return rocksdb::Status::OK();
}

rocksdb::Status Store::GetHealth(HealthStats* stats) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!stats) return rocksdb::Status::InvalidArgument("stats is null");

  *stats = HealthStats{};

  const rocksdb::Snapshot* snapshot = db_->GetSnapshot();
  rocksdb::ReadOptions ro;
  ro.snapshot = snapshot;

  uint64_t now_us = GetWallClockMicros();
  uint64_t ttl_us = opt_.default_ttl_seconds * 1000000ULL;

  uint64_t oldest_created = UINT64_MAX;
  uint64_t newest_accessed = 0;

  // Count keys
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, user_kv_cf_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      stats->total_keys++;
    }
  }

  // Count objects and gather stats
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, refcount_cf_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      std::string obj_id(it->key().data(), it->key().size());

      uint64_t refcount = 0;
      std::string_view v(it->value().data(), it->value().size());
      if (!prestige::internal::DecodeU64LE(v, &refcount)) {
        continue;
      }

      if (refcount == 0) {
        stats->orphaned_objects++;
      }

      stats->total_objects++;

      // Get metadata
      std::string meta_raw;
      if (db_->Get(ro, meta_cf_, rocksdb::Slice(obj_id), &meta_raw).ok()) {
        prestige::internal::ObjectMeta meta;
        if (prestige::internal::ObjectMeta::Deserialize(meta_raw, &meta)) {
          stats->total_bytes += meta.size_bytes;

          if (!meta.IsLegacy()) {
            if (meta.created_at_us < oldest_created) {
              oldest_created = meta.created_at_us;
            }
            if (meta.last_accessed_us > newest_accessed) {
              newest_accessed = meta.last_accessed_us;
            }

            // Check for expired
            if (ttl_us > 0 && now_us - meta.created_at_us > ttl_us) {
              stats->expired_objects++;
            }
          }
        }
      }
    }
  }

  // Decision records: how many exist, how many wait for repair, and the lifetime counters
  stats->decisions_total = CountDecisionPrefix(db_, decisions_cf_, ro, kDecRecord);
  stats->decision_queue_size = CountDecisionPrefix(db_, decisions_cf_, ro, kDecQueue);
  stats->decision_dangling_reads = ReadDecisionCounter(db_, decisions_cf_, ro, "dangling_reads");
  stats->decision_repairs = ReadDecisionCounter(db_, decisions_cf_, ro, "repairs");

  db_->ReleaseSnapshot(snapshot);

  // Calculate derived stats
  if (oldest_created != UINT64_MAX) {
    stats->oldest_object_age_s = (now_us - oldest_created) / 1000000ULL;
  }
  if (newest_accessed > 0) {
    stats->newest_access_age_s = (now_us - newest_accessed) / 1000000ULL;
  }
  if (stats->total_objects > 0) {
    stats->dedup_ratio = static_cast<double>(stats->total_keys) /
                         static_cast<double>(stats->total_objects);
  }

  return rocksdb::Status::OK();
}

// ---------------------------------------------------------------------------
// Decision records (provenance)
// ---------------------------------------------------------------------------

std::string Store::ComputeDigestKey(std::string_view value_bytes) const {
  // Mirrors the digest computation in PutImpl (kept separate there for its per-step telemetry).
  std::string normalized_value;
  std::string_view digest_input = value_bytes;
  if (opt_.normalization_mode != NormalizationMode::kNone) {
    if (opt_.normalization_max_bytes == 0 || value_bytes.size() <= opt_.normalization_max_bytes) {
      normalized_value = prestige::internal::Normalize(value_bytes, opt_.normalization_mode);
      digest_input = normalized_value;
    }
  }
  auto digest = prestige::internal::Sha256::Digest(digest_input);
  return prestige::internal::ToBytes(digest.data(), digest.size());
}

rocksdb::Status Store::Digest(std::string_view value_bytes, std::string* digest_out) const {
  if (!digest_out) return rocksdb::Status::InvalidArgument("digest_out is null");
#ifdef PRESTIGE_ENABLE_SEMANTIC
  if (opt_.dedup_mode == DedupMode::kSemantic) {
    return rocksdb::Status::InvalidArgument("Digest is only defined in exact mode");
  }
#endif
  *digest_out = ComputeDigestKey(value_bytes);
  return rocksdb::Status::OK();
}

rocksdb::Status Store::PutWithDecision(std::string_view user_key,
                                       std::string_view value_bytes,
                                       const Decision& decision) {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (decision.decision_id.empty()) return rocksdb::Status::InvalidArgument("decision_id is required");
  if (decision.decision_id.size() > kDecMaxField) return rocksdb::Status::InvalidArgument("decision_id is too long");
  if (decision.policy_revision.empty()) return rocksdb::Status::InvalidArgument("policy_revision is required");
  if (decision.policy_revision.size() > kDecMaxField) return rocksdb::Status::InvalidArgument("policy_revision is too long");
  if (decision.parent_decision_id.size() > kDecMaxField) return rocksdb::Status::InvalidArgument("parent_decision_id is too long");
  if (decision.note.size() > kDecMaxNote) return rocksdb::Status::InvalidArgument("note is too long");
  if (decision.input_digests.size() > kDecMaxInputs) return rocksdb::Status::InvalidArgument("too many input digests");
  for (const auto& d : decision.input_digests) {
    if (d.size() != 32) {
      return rocksdb::Status::InvalidArgument("input digests must be 32-byte SHA-256 digests (see Store::Digest)");
    }
  }
#ifdef PRESTIGE_ENABLE_SEMANTIC
  if (opt_.dedup_mode == DedupMode::kSemantic) {
    return rocksdb::Status::InvalidArgument("decision records are not supported in semantic mode");
  }
#endif
  return PutImpl(user_key, value_bytes, &decision);
}

rocksdb::Status Store::CheckDecisionBodies(const DecisionRecord& rec, std::vector<std::string>* missing) const {
  missing->clear();
  rocksdb::ReadOptions ro;
  auto resolves = [&](const std::string& digest) -> bool {
    std::string obj_id;
    if (!db_->Get(ro, dedup_cf_, rocksdb::Slice(digest), &obj_id).ok()) return false;
    std::string meta_raw;
    return db_->Get(ro, meta_cf_, rocksdb::Slice(obj_id), &meta_raw).ok();
  };
  if (!rec.output_digest.empty() && !resolves(rec.output_digest)) missing->push_back(rec.output_digest);
  for (const auto& d : rec.decision.input_digests) {
    if (!resolves(d)) missing->push_back(d);
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Store::RecordDanglingRead(const DecisionRecord& rec, bool* queued) const {
  *queued = false;
  rocksdb::WriteOptions wo;
  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;
  const std::string qkey = DecQueueKey(rec.sequence);
  for (int attempt = 0; attempt < opt_.max_retries; ++attempt) {
    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) return rocksdb::Status::IOError("BeginTransaction returned null");
    rocksdb::ReadOptions ro;
    std::string existing;
    bool newly_queued = false;
    rocksdb::Status s = txn->GetForUpdate(ro, decisions_cf_, rocksdb::Slice(qkey), &existing);
    if (s.IsNotFound()) {
      s = txn->Put(decisions_cf_, rocksdb::Slice(qkey), rocksdb::Slice(rec.decision.decision_id));
      newly_queued = true;
    }
    if (s.ok()) s = BumpDecisionCounterLocked(txn.get(), decisions_cf_, "dangling_reads", 1);
    if (s.ok()) s = txn->Commit();
    if (s.ok()) {
      *queued = newly_queued;
      return s;
    }
    if (!prestige::internal::IsRetryableTxnStatus(s)) return s;
    BackoffBeforeRetry(opt_, attempt, nullptr);
  }
  return rocksdb::Status::TimedOut("RecordDanglingRead exceeded max_retries");
}

rocksdb::Status Store::GetDecision(std::string_view decision_id,
                                   DecisionRecord* out,
                                   DecisionCheck* check) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out) return rocksdb::Status::InvalidArgument("out is null");
  EmitCounter(opt_, "prestige.decision.get_total", 1);

  rocksdb::ReadOptions ro;
  std::string raw;
  rocksdb::Status s = db_->Get(ro, decisions_cf_, rocksdb::Slice(DecRecordKey(decision_id)), &raw);
  if (!s.ok()) return s;
  if (!DeserializeDecisionRecord(raw, out)) return rocksdb::Status::Corruption("decision record is malformed");

  std::vector<std::string> missing;
  CheckDecisionBodies(*out, &missing);
  if (check) {
    check->resolved = missing.empty();
    check->missing_digests = missing;
    check->queued_for_repair = false;
  }
  if (!missing.empty()) {
    // The read-time guard: count the miss and queue the record for a priority sweep. The read itself still
    // succeeds so the caller can see what is missing; VerifyDecision is the failing variant.
    EmitCounter(opt_, "prestige.decision.dangling_read_total", 1);
    bool queued = false;
    rocksdb::Status rs = RecordDanglingRead(*out, &queued);
    if (!rs.ok()) EmitCounter(opt_, "prestige.decision.bookkeeping_error_total", 1);
    if (check) check->queued_for_repair = queued;
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Store::VerifyDecision(std::string_view decision_id, DecisionRecord* out) const {
  DecisionRecord rec;
  DecisionCheck check;
  rocksdb::Status s = GetDecision(decision_id, &rec, &check);
  if (!s.ok()) return s;
  if (out) *out = rec;
  if (!check.resolved) {
    return rocksdb::Status::Corruption("decision " + std::string(decision_id) + " references " +
                                       std::to_string(check.missing_digests.size()) + " missing bodies");
  }
  return rocksdb::Status::OK();
}

rocksdb::Status Store::ListDecisions(std::vector<DecisionRecord>* out,
                                     uint64_t limit,
                                     uint64_t after_sequence) const {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!out) return rocksdb::Status::InvalidArgument("out is null");
  out->clear();
  rocksdb::ReadOptions ro;
  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, decisions_cf_));
  for (it->Seek(rocksdb::Slice(DecSeqKey(after_sequence + 1)));
       it->Valid() && it->key().size() == 9 && it->key()[0] == kDecSeq; it->Next()) {
    std::string raw;
    rocksdb::Status s = db_->Get(ro, decisions_cf_, rocksdb::Slice(DecRecordKey(std::string_view(it->value().data(), it->value().size()))), &raw);
    if (!s.ok()) continue;  // index entry without a record is reported by the sweep, not here
    DecisionRecord rec;
    if (!DeserializeDecisionRecord(raw, &rec)) continue;
    out->push_back(std::move(rec));
    if (limit > 0 && out->size() >= limit) break;
  }
  return it->status();
}

rocksdb::Status Store::SweepDecisions(uint64_t max_records, DecisionSweepStats* stats) {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");
  if (!stats) return rocksdb::Status::InvalidArgument("stats is null");
  *stats = DecisionSweepStats{};
  if (max_records == 0) max_records = UINT64_MAX;

  rocksdb::ReadOptions ro;
  rocksdb::WriteOptions wo;
  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  auto load = [&](std::string_view decision_id, DecisionRecord* rec) -> bool {
    std::string raw;
    if (!db_->Get(ro, decisions_cf_, rocksdb::Slice(DecRecordKey(decision_id)), &raw).ok()) return false;
    return DeserializeDecisionRecord(raw, rec);
  };

  // Sequences verified during the queue phase; the walk below skips them so a record counts once per sweep.
  std::unordered_set<uint64_t> handled;

  // 1) Drain the queue: priority entries first, in write order.
  {
    std::vector<std::pair<std::string, std::string>> queued;  // (queue key, decision id)
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, decisions_cf_));
    for (it->Seek(rocksdb::Slice(std::string(1, kDecQueue)));
         it->Valid() && it->key().size() == 9 && it->key()[0] == kDecQueue && queued.size() < max_records; it->Next()) {
      queued.emplace_back(it->key().ToString(), it->value().ToString());
    }
    for (const auto& [qkey, id] : queued) {
      stats->checked++;
      uint64_t qseq = 0;
      DecodeU64BE(std::string_view(qkey.data() + 1, 8), &qseq);
      handled.insert(qseq);
      DecisionRecord rec;
      std::vector<std::string> missing;
      const bool have = load(id, &rec);
      if (have) CheckDecisionBodies(rec, &missing);
      if (have && !missing.empty()) {
        stats->dangling++;
        continue;  // still waiting for its bodies
      }
      // Resolved (or the record itself is gone): leave the queue and count the repair
      std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
      if (!txn) continue;
      rocksdb::Status s = txn->Delete(decisions_cf_, rocksdb::Slice(qkey));
      if (s.ok() && have) s = BumpDecisionCounterLocked(txn.get(), decisions_cf_, "repairs", 1);
      if (s.ok()) s = txn->Commit();
      if (s.ok() && have) stats->repaired++;
    }
  }

  // 2) Continue the full walk from the cursor, in write order, with the remaining budget.
  uint64_t cursor = 0;
  {
    std::string raw;
    if (db_->Get(ro, decisions_cf_, rocksdb::Slice(DecCursorKey()), &raw).ok()) DecodeU64BE(raw, &cursor);
  }
  {
    std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, decisions_cf_));
    for (it->Seek(rocksdb::Slice(DecSeqKey(cursor + 1)));
         it->Valid() && it->key().size() == 9 && it->key()[0] == kDecSeq && stats->checked < max_records; it->Next()) {
      uint64_t seq = 0;
      DecodeU64BE(std::string_view(it->key().data() + 1, 8), &seq);
      cursor = seq;
      if (handled.count(seq)) continue;  // verified in the queue phase of this call
      const std::string id = it->value().ToString();
      stats->checked++;
      DecisionRecord rec;
      std::vector<std::string> missing;
      if (!load(id, &rec)) {
        missing.push_back(std::string());  // index entry whose record is missing: treat as dangling
      } else {
        CheckDecisionBodies(rec, &missing);
      }
      if (missing.empty()) continue;
      stats->dangling++;
      std::string existing;
      if (db_->Get(ro, decisions_cf_, rocksdb::Slice(DecQueueKey(seq)), &existing).ok()) continue;  // already queued
      std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
      if (!txn) continue;
      rocksdb::Status s = txn->Put(decisions_cf_, rocksdb::Slice(DecQueueKey(seq)), rocksdb::Slice(id));
      if (s.ok()) s = txn->Commit();
      (void)s;
    }
  }
  {
    rocksdb::Status s = db_->Put(wo, decisions_cf_, rocksdb::Slice(DecCursorKey()), rocksdb::Slice(EncodeU64BE(cursor)));
    if (!s.ok()) return s;
  }

  stats->cursor_sequence = cursor;
  stats->max_sequence = next_decision_seq_.load() - 1;
  stats->queue_size = CountDecisionPrefix(db_, decisions_cf_, ro, kDecQueue);

  EmitCounter(opt_, "prestige.decision.sweep_checked_total", stats->checked);
  EmitCounter(opt_, "prestige.decision.repaired_total", stats->repaired);
  EmitGauge(opt_, "prestige.decision.queue_size", static_cast<double>(stats->queue_size));
  return rocksdb::Status::OK();
}

rocksdb::Status Store::Flush() {
  if (!db_) return rocksdb::Status::InvalidArgument("db is closed");

  EmitCounter(opt_, "prestige.flush.calls", 1);
  const uint64_t start_us = prestige::internal::NowMicros();

  // Sync WAL to ensure durability of committed transactions
  rocksdb::Status s = db_->SyncWAL();
  if (!s.ok()) {
    EmitCounter(opt_, "prestige.flush.sync_error_total", 1);
    return s;
  }

  EmitHistogram(opt_, "prestige.flush.latency_us",
                prestige::internal::NowMicros() - start_us);
  EmitCounter(opt_, "prestige.flush.ok_total", 1);

  return rocksdb::Status::OK();
}

void Store::Close() {
  if (!db_) return;

  // Flush all pending data before closing
  rocksdb::Status flush_status = Flush();
  if (!flush_status.ok()) {
    EmitCounter(opt_, "prestige.close.flush_error_total", 1);
    // Continue with close even if flush fails
  }

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Save vector index before closing
  if (vector_index_ && !vector_index_path_.empty()) {
    bool save_ok = vector_index_->Save(vector_index_path_);
    if (save_ok) {
      // Clear pending ops now that vector index is saved
      ClearPendingVectorOps();
      EmitCounter(opt_, "prestige.close.vector_save_ok_total", 1);

      // Compact if we have many deleted entries
      if (opt_.semantic_compact_threshold > 0) {
        size_t deleted = vector_index_->DeletedCount();
        if (deleted >= opt_.semantic_compact_threshold) {
          if (vector_index_->Compact()) {
            // Save again after compaction
            vector_index_->Save(vector_index_path_);
            EmitCounter(opt_, "prestige.close.compact_ok_total", 1);
          }
        }
      }
    } else {
      EmitCounter(opt_, "prestige.close.vector_save_error_total", 1);
      // Don't clear pending ops - they'll be replayed on next open
    }
  }

  // Release semantic resources
  vector_index_.reset();
  embedder_.reset();
  embeddings_cf_ = nullptr;
  vector_pending_cf_ = nullptr;
#endif

  for (auto* h : handles_) delete h;
  handles_.clear();
  delete db_;
  db_ = nullptr;
  user_kv_cf_ = objects_cf_ = dedup_cf_ = refcount_cf_ = meta_cf_ = lru_cf_ = nullptr;

  EmitCounter(opt_, "prestige.close.ok_total", 1);
}

static rocksdb::Status AdjustRefcountLocked(rocksdb::Transaction* txn,
                                            rocksdb::ColumnFamilyHandle* refcount_cf,
                                            const std::string& obj_id,
                                            int64_t delta,
                                            uint64_t* out_new_value) {
  if (!txn) return rocksdb::Status::InvalidArgument("txn is null");
  if (!(delta == +1 || delta == -1)) return rocksdb::Status::InvalidArgument("delta must be +1 or -1");
  if (!out_new_value) return rocksdb::Status::InvalidArgument("out_new_value is null");

  rocksdb::ReadOptions ro;
  std::string cur;
  rocksdb::Status s = txn->GetForUpdate(ro, refcount_cf, rocksdb::Slice(obj_id), &cur);

  uint64_t v = 0;
  if (s.IsNotFound()) {
    v = 0;
  } else if (s.ok()) {
    if (!prestige::internal::DecodeU64LE(std::string_view(cur), &v)) {
      return rocksdb::Status::Corruption("refcount value is not uint64_le");
    }
  } else {
    return s;
  }

  if (delta == -1) {
    if (v == 0) return rocksdb::Status::Corruption("refcount underflow");
    v -= 1;
  } else {
    v += 1;
  }

  s = txn->Put(refcount_cf, rocksdb::Slice(obj_id), rocksdb::Slice(prestige::internal::EncodeU64LE(v)));
  if (!s.ok()) return s;

  *out_new_value = v;
  return rocksdb::Status::OK();
}

static rocksdb::Status DeleteObjectIfUnreferencedLocked(rocksdb::Transaction* txn,
                                                        rocksdb::ColumnFamilyHandle* objects_cf,
                                                        rocksdb::ColumnFamilyHandle* dedup_cf,
                                                        rocksdb::ColumnFamilyHandle* refcount_cf,
                                                        rocksdb::ColumnFamilyHandle* meta_cf,
                                                        rocksdb::ColumnFamilyHandle* lru_cf,
                                                        std::atomic<uint64_t>* total_store_bytes,
                                                        const std::string& obj_id) {
  if (!txn) return rocksdb::Status::InvalidArgument("txn is null");

  rocksdb::ReadOptions ro;

  // Lookup metadata (locks meta)
  std::string meta_raw;
  rocksdb::Status s = txn->GetForUpdate(ro, meta_cf, rocksdb::Slice(obj_id), &meta_raw);
  if (s.IsNotFound()) {
    // Best-effort cleanup of object/refcount, but cannot reliably cleanup dedup index.
    (void)txn->Delete(objects_cf, rocksdb::Slice(obj_id));
    (void)txn->Delete(refcount_cf, rocksdb::Slice(obj_id));
    return rocksdb::Status::Corruption("object_meta missing; best-effort cleanup done");
  }
  if (!s.ok()) return s;

  // Parse metadata to get digest_key and LRU info
  prestige::internal::ObjectMeta meta;
  if (!prestige::internal::ObjectMeta::Deserialize(meta_raw, &meta)) {
    return rocksdb::Status::Corruption("Failed to parse object metadata");
  }

  // Only delete dedup mapping if it still points to this obj_id
  std::string mapped_id;
  s = txn->GetForUpdate(ro, dedup_cf, rocksdb::Slice(meta.digest_key), &mapped_id);
  if (s.ok()) {
    if (mapped_id == obj_id) {
      s = txn->Delete(dedup_cf, rocksdb::Slice(meta.digest_key));
      if (!s.ok()) return s;
    }
  } else if (!s.IsNotFound()) {
    return s;
  }

  // Delete LRU index entry
  if (lru_cf && !meta.IsLegacy()) {
    std::string lru_key = prestige::internal::MakeLRUKey(meta.last_accessed_us, obj_id);
    (void)txn->Delete(lru_cf, rocksdb::Slice(lru_key));
  }

  // Remove object bytes + meta + refcount
  s = txn->Delete(objects_cf, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

  s = txn->Delete(meta_cf, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

  s = txn->Delete(refcount_cf, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

  // Update total store size
  if (total_store_bytes && meta.size_bytes > 0) {
    uint64_t old_val = total_store_bytes->load();
    if (old_val >= meta.size_bytes) {
      total_store_bytes->fetch_sub(meta.size_bytes);
    } else {
      total_store_bytes->store(0);
    }
  }

  return rocksdb::Status::OK();
}

rocksdb::Status Store::PutImpl(std::string_view user_key, std::string_view value_bytes,
                               const Decision* decision) {
  EmitCounter(opt_, "prestige.put.calls", 1);
  EmitHistogram(opt_, "prestige.put.value_bytes", static_cast<uint64_t>(value_bytes.size()));

  const uint64_t op_start_us = prestige::internal::NowMicros();
  std::unique_ptr<TraceSpan> span;
  if (opt_.tracer) span = opt_.tracer->StartSpan("prestige.Put");
  SpanAttr(span.get(), "key_bytes", static_cast<uint64_t>(user_key.size()));
  SpanAttr(span.get(), "value_bytes", static_cast<uint64_t>(value_bytes.size()));

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // For semantic mode, compute embedding instead of SHA-256
  if (opt_.dedup_mode == DedupMode::kSemantic) {
    if (decision) {
      return rocksdb::Status::InvalidArgument("decision records are not supported in semantic mode");
    }
    return PutImplSemantic(user_key, value_bytes, span.get(), op_start_us);
  }
#endif

  // Exact mode: Compute SHA-256 digest as dedup key
  const uint64_t sha_start_us = prestige::internal::NowMicros();

  // Apply normalization for dedup key computation (if enabled)
  std::string normalized_value;
  std::string_view digest_input = value_bytes;

  if (opt_.normalization_mode != NormalizationMode::kNone) {
    // Skip normalization for huge values (configurable limit)
    if (opt_.normalization_max_bytes == 0 ||
        value_bytes.size() <= opt_.normalization_max_bytes) {
      const uint64_t norm_start_us = prestige::internal::NowMicros();
      normalized_value = prestige::internal::Normalize(value_bytes, opt_.normalization_mode);
      EmitHistogram(opt_, "prestige.put.normalize_us",
                    prestige::internal::NowMicros() - norm_start_us);
      digest_input = normalized_value;
      if (span) {
        SpanAttr(span.get(), "normalized_bytes",
                 static_cast<uint64_t>(normalized_value.size()));
      }
    }
  }

  auto digest = prestige::internal::Sha256::Digest(digest_input);
  EmitHistogram(opt_, "prestige.put.sha256_us", prestige::internal::NowMicros() - sha_start_us);
  std::string digest_key = prestige::internal::ToBytes(digest.data(), digest.size());

  bool dedup_hit_final = false;
  bool had_old_final = false;
  bool noop_overwrite = false;
  int attempts_used = 0;
  uint64_t total_wait_us = 0;  // Time spent waiting on retries
  int batch_writes = 0;         // Number of CF writes in this Put
  // A decision takes its sequence number once, so retries do not burn numbers.
  const uint64_t decision_seq = decision ? next_decision_seq_.fetch_add(1) : 0;
  bool decision_replayed = false;

  auto finish = [&](const rocksdb::Status& st) -> rocksdb::Status {
    const uint64_t dur_us = prestige::internal::NowMicros() - op_start_us;
    EmitHistogram(opt_, "prestige.put.latency_us", dur_us);
    EmitHistogram(opt_, "prestige.put.attempts", static_cast<uint64_t>(attempts_used));
    if (total_wait_us > 0) {
      EmitHistogram(opt_, "prestige.txn.wait_us", total_wait_us);
    }
    if (batch_writes > 0) {
      EmitHistogram(opt_, "prestige.put.batch_writes", static_cast<uint64_t>(batch_writes));
    }

    if (st.ok()) {
      EmitCounter(opt_, "prestige.put.ok_total", 1);
    } else if (st.IsTimedOut()) {
      EmitCounter(opt_, "prestige.put.timed_out_total", 1);
    } else {
      EmitCounter(opt_, "prestige.put.error_total", 1);
    }

    if (span) {
      SpanAttr(span.get(), "latency_us", dur_us);
      SpanAttr(span.get(), "attempts", static_cast<uint64_t>(attempts_used));
      SpanAttr(span.get(), "dedup_hit", static_cast<uint64_t>(dedup_hit_final ? 1 : 0));
      SpanAttr(span.get(), "had_old", static_cast<uint64_t>(had_old_final ? 1 : 0));
      SpanAttr(span.get(), "noop_overwrite", static_cast<uint64_t>(noop_overwrite ? 1 : 0));
      SpanAttr(span.get(), "decision", static_cast<uint64_t>(decision ? 1 : 0));
      SpanAttr(span.get(), "decision_replayed", static_cast<uint64_t>(decision_replayed ? 1 : 0));
      SpanAttr(span.get(), "status", StatusKind(st));
      span->End(st);
    }
    return st;
  };

  rocksdb::WriteOptions wo;
  rocksdb::ReadOptions ro;

  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  uint64_t attempt_start_us = prestige::internal::NowMicros();
  for (int attempt = 0; attempt < opt_.max_retries; ++attempt) {
    attempts_used = attempt + 1;
    batch_writes = 0;  // Reset for this attempt

    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) return finish(rocksdb::Status::IOError("BeginTransaction returned null"));

    // Lock user_key mapping (detect overwrite)
    std::string old_obj_id;
    bool had_old = false;
    {
      rocksdb::Status s = txn->GetForUpdate(
          ro, user_kv_cf_,
          rocksdb::Slice(user_key.data(), user_key.size()),
          &old_obj_id);

      if (s.ok()) {
        had_old = true;
      } else if (!s.IsNotFound()) {
        if (prestige::internal::IsRetryableTxnStatus(s)) {
          EmitCounter(opt_, "prestige.put.retry_total", 1);
          SpanEvent(span.get(), "retry.user_key_lock");
          BackoffBeforeRetry(opt_, attempt, span.get());
          total_wait_us += prestige::internal::NowMicros() - attempt_start_us;
          attempt_start_us = prestige::internal::NowMicros();
          continue;
        }
        return finish(s);
      }
    }

    // Lock digest mapping and resolve object_id
    std::string obj_id;
    {
      rocksdb::Status s = txn->GetForUpdate(ro, dedup_cf_, rocksdb::Slice(digest_key), &obj_id);

      if (s.IsNotFound()) {
        EmitCounter(opt_, "prestige.put.dedup_miss_total", 1);
        EmitCounter(opt_, "prestige.put.object_created_total", 1);
        dedup_hit_final = false;

        auto new_id = prestige::internal::RandomObjectId128();
        obj_id = prestige::internal::ToBytes(new_id.data(), new_id.size());

        // Create: object bytes
        s = txn->Put(objects_cf_, rocksdb::Slice(obj_id),
                     rocksdb::Slice(value_bytes.data(), value_bytes.size()));
        if (!s.ok()) return finish(s);
        ++batch_writes;

        // Create full ObjectMeta with timestamps
        uint64_t now_us = GetWallClockMicros();
        prestige::internal::ObjectMeta meta;
        meta.digest_key = digest_key;
        meta.created_at_us = now_us;
        meta.last_accessed_us = now_us;
        meta.size_bytes = value_bytes.size();

        s = txn->Put(meta_cf_, rocksdb::Slice(obj_id),
                     rocksdb::Slice(meta.Serialize()));
        if (!s.ok()) return finish(s);
        ++batch_writes;

        // Add to LRU index
        std::string lru_key = prestige::internal::MakeLRUKey(now_us, obj_id);
        s = txn->Put(lru_cf_, rocksdb::Slice(lru_key), rocksdb::Slice());
        if (!s.ok()) return finish(s);
        ++batch_writes;

        s = txn->Put(dedup_cf_, rocksdb::Slice(digest_key), rocksdb::Slice(obj_id));
        if (!s.ok()) return finish(s);
        ++batch_writes;

        s = txn->Put(refcount_cf_, rocksdb::Slice(obj_id),
                     rocksdb::Slice(prestige::internal::EncodeU64LE(0)));
        if (!s.ok()) return finish(s);
        ++batch_writes;

        // Update total store size
        total_store_bytes_.fetch_add(value_bytes.size());

      } else if (!s.ok()) {
        if (prestige::internal::IsRetryableTxnStatus(s)) {
          EmitCounter(opt_, "prestige.put.retry_total", 1);
          SpanEvent(span.get(), "retry.dedup_lock");
          BackoffBeforeRetry(opt_, attempt, span.get());
          total_wait_us += prestige::internal::NowMicros() - attempt_start_us;
          attempt_start_us = prestige::internal::NowMicros();
          continue;
        }
        return finish(s);

      } else {
        EmitCounter(opt_, "prestige.put.dedup_hit_total", 1);
        dedup_hit_final = true;
      }
    }

    had_old_final = had_old;

    const bool same_object = had_old && old_obj_id == obj_id;

    // Decision record: lock it first so a replayed decision is recognized and a reused id is refused.
    if (decision) {
      std::string existing_raw;
      rocksdb::Status ds = txn->GetForUpdate(ro, decisions_cf_,
                                             rocksdb::Slice(DecRecordKey(decision->decision_id)), &existing_raw);
      if (ds.ok()) {
        DecisionRecord existing;
        if (!DeserializeDecisionRecord(existing_raw, &existing)) {
          return finish(rocksdb::Status::Corruption("existing decision record is malformed"));
        }
        if (existing.user_key == user_key && existing.output_digest == digest_key) {
          decision_replayed = true;
          EmitCounter(opt_, "prestige.decision.replayed_total", 1);
          txn->Rollback();
          return finish(rocksdb::Status::OK());
        }
        return finish(rocksdb::Status::InvalidArgument("decision_id already records a different write"));
      }
      if (!ds.IsNotFound()) {
        if (prestige::internal::IsRetryableTxnStatus(ds)) {
          EmitCounter(opt_, "prestige.put.retry_total", 1);
          SpanEvent(span.get(), "retry.decision_lock");
          BackoffBeforeRetry(opt_, attempt, span.get());
          total_wait_us += prestige::internal::NowMicros() - attempt_start_us;
          attempt_start_us = prestige::internal::NowMicros();
          continue;
        }
        return finish(ds);
      }
    }

    // If overwrite maps to same object_id and there is no decision to record, nothing to do
    if (same_object && !decision) {
      noop_overwrite = true;
      EmitCounter(opt_, "prestige.put.noop_overwrite_total", 1);
      txn->Rollback();
      return finish(rocksdb::Status::OK());
    }
    if (same_object) noop_overwrite = true;

    if (!same_object) {
    // user_key -> obj_id
    {
      rocksdb::Status s = txn->Put(
          user_kv_cf_,
          rocksdb::Slice(user_key.data(), user_key.size()),
          rocksdb::Slice(obj_id));
      if (!s.ok()) return finish(s);
      ++batch_writes;
    }

    // Incref(new)
    {
      uint64_t new_cnt = 0;
      rocksdb::Status s = AdjustRefcountLocked(txn.get(), refcount_cf_, obj_id, +1, &new_cnt);
      if (!s.ok()) return finish(s);
      ++batch_writes;  // refcount write
    }

    // Decref(old) and GC if needed
    if (had_old && old_obj_id != obj_id) {
      uint64_t old_cnt = 0;
      rocksdb::Status s = AdjustRefcountLocked(txn.get(), refcount_cf_, old_obj_id, -1, &old_cnt);
      if (!s.ok()) return finish(s);
      ++batch_writes;  // refcount write

      if (opt_.enable_gc && old_cnt == 0) {
        s = DeleteObjectIfUnreferencedLocked(
            txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, lru_cf_,
            &total_store_bytes_, old_obj_id);
        if (!s.ok()) return finish(s);
        batch_writes += 5;  // delete from objects, dedup, refcount, meta, lru

        EmitCounter(opt_, "prestige.gc.deleted_objects_total", 1);
        SpanEvent(span.get(), "gc.delete_object");
      }
    }

    }  // !same_object

    // The decision record commits with the value it explains, never beside it.
    if (decision) {
      DecisionRecord rec;
      rec.decision = *decision;
      rec.sequence = decision_seq;
      rec.committed_at_us = GetWallClockMicros();
      rec.user_key.assign(user_key.data(), user_key.size());
      rec.output_digest = digest_key;
      rocksdb::Status ws = txn->Put(decisions_cf_, rocksdb::Slice(DecRecordKey(decision->decision_id)),
                                    rocksdb::Slice(SerializeDecisionRecord(rec)));
      if (!ws.ok()) return finish(ws);
      ws = txn->Put(decisions_cf_, rocksdb::Slice(DecSeqKey(decision_seq)),
                    rocksdb::Slice(decision->decision_id));
      if (!ws.ok()) return finish(ws);
      batch_writes += 2;
      EmitCounter(opt_, "prestige.decision.recorded_total", 1);
    }

    const uint64_t commit_start_us = prestige::internal::NowMicros();
    rocksdb::Status cs = txn->Commit();
    EmitHistogram(opt_, "prestige.put.commit_us", prestige::internal::NowMicros() - commit_start_us);

    if (cs.ok()) return finish(cs);

    if (prestige::internal::IsRetryableTxnStatus(cs)) {
      EmitCounter(opt_, "prestige.put.retry_total", 1);
      SpanEvent(span.get(), "retry.commit");
      BackoffBeforeRetry(opt_, attempt, span.get());
      total_wait_us += prestige::internal::NowMicros() - attempt_start_us;
      attempt_start_us = prestige::internal::NowMicros();
      continue;
    }

    return finish(cs);
  }

  return finish(rocksdb::Status::TimedOut("Put exceeded max_retries"));
}

rocksdb::Status Store::DeleteImpl(std::string_view user_key) {
  EmitCounter(opt_, "prestige.delete.calls", 1);

  const uint64_t op_start_us = prestige::internal::NowMicros();
  std::unique_ptr<TraceSpan> span;
  if (opt_.tracer) span = opt_.tracer->StartSpan("prestige.Delete");
  SpanAttr(span.get(), "key_bytes", static_cast<uint64_t>(user_key.size()));

  int attempts_used = 0;
  uint64_t total_wait_us = 0;  // Time spent waiting on retries
  int batch_writes = 0;         // Number of CF writes in this Delete

  auto finish = [&](const rocksdb::Status& st) -> rocksdb::Status {
    const uint64_t dur_us = prestige::internal::NowMicros() - op_start_us;
    EmitHistogram(opt_, "prestige.delete.latency_us", dur_us);
    EmitHistogram(opt_, "prestige.delete.attempts", static_cast<uint64_t>(attempts_used));
    if (total_wait_us > 0) {
      EmitHistogram(opt_, "prestige.txn.wait_us", total_wait_us);
    }
    if (batch_writes > 0) {
      EmitHistogram(opt_, "prestige.delete.batch_writes", static_cast<uint64_t>(batch_writes));
    }

    if (st.ok()) {
      EmitCounter(opt_, "prestige.delete.ok_total", 1);
    } else if (st.IsNotFound()) {
      EmitCounter(opt_, "prestige.delete.not_found_total", 1);
    } else if (st.IsTimedOut()) {
      EmitCounter(opt_, "prestige.delete.timed_out_total", 1);
    } else {
      EmitCounter(opt_, "prestige.delete.error_total", 1);
    }

    if (span) {
      SpanAttr(span.get(), "latency_us", dur_us);
      SpanAttr(span.get(), "attempts", static_cast<uint64_t>(attempts_used));
      SpanAttr(span.get(), "status", StatusKind(st));
      span->End(st);
    }
    return st;
  };

  rocksdb::WriteOptions wo;
  rocksdb::ReadOptions ro;

  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  uint64_t attempt_start_us = prestige::internal::NowMicros();
  for (int attempt = 0; attempt < opt_.max_retries; ++attempt) {
    attempts_used = attempt + 1;
    batch_writes = 0;  // Reset for this attempt

    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) return finish(rocksdb::Status::IOError("BeginTransaction returned null"));

    // Lock and fetch mapping
    std::string obj_id;
    rocksdb::Status s = txn->GetForUpdate(
        ro, user_kv_cf_,
        rocksdb::Slice(user_key.data(), user_key.size()),
        &obj_id);

    if (s.IsNotFound()) return finish(s);
    if (!s.ok()) {
      if (prestige::internal::IsRetryableTxnStatus(s)) {
        EmitCounter(opt_, "prestige.delete.retry_total", 1);
        SpanEvent(span.get(), "retry.user_key_lock");
        BackoffBeforeRetry(opt_, attempt, span.get());
        total_wait_us += prestige::internal::NowMicros() - attempt_start_us;
        attempt_start_us = prestige::internal::NowMicros();
        continue;
      }
      return finish(s);
    }

    // Remove user mapping
    s = txn->Delete(user_kv_cf_, rocksdb::Slice(user_key.data(), user_key.size()));
    if (!s.ok()) return finish(s);
    ++batch_writes;

    // Decref and maybe GC
    uint64_t cnt = 0;
    s = AdjustRefcountLocked(txn.get(), refcount_cf_, obj_id, -1, &cnt);
    if (!s.ok()) return finish(s);
    ++batch_writes;  // refcount write

    // Track if we're GC'ing a semantic object (for post-commit vector index update)
    bool gc_semantic_object = false;
    std::string gc_obj_id;

    if (opt_.enable_gc && cnt == 0) {
#ifdef PRESTIGE_ENABLE_SEMANTIC
      if (opt_.dedup_mode == DedupMode::kSemantic) {
        s = DeleteSemanticObject(txn.get(), obj_id);
        batch_writes += 4;  // pending, embeddings, objects, refcount
        gc_semantic_object = true;
        gc_obj_id = obj_id;
      } else {
        s = DeleteObjectIfUnreferencedLocked(
            txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, lru_cf_,
            &total_store_bytes_, obj_id);
        batch_writes += 5;  // objects, dedup, refcount, meta, lru deletes
      }
#else
      s = DeleteObjectIfUnreferencedLocked(
          txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, lru_cf_,
          &total_store_bytes_, obj_id);
      batch_writes += 5;  // objects, dedup, refcount, meta, lru deletes
#endif
      if (!s.ok()) return finish(s);

      EmitCounter(opt_, "prestige.gc.deleted_objects_total", 1);
      SpanEvent(span.get(), "gc.delete_object");
    }

    const uint64_t commit_start_us = prestige::internal::NowMicros();
    rocksdb::Status cs = txn->Commit();
    EmitHistogram(opt_, "prestige.delete.commit_us",
                  prestige::internal::NowMicros() - commit_start_us);

    if (cs.ok()) {
#ifdef PRESTIGE_ENABLE_SEMANTIC
      // Apply pending vector ops AFTER successful commit
      if (gc_semantic_object && !gc_obj_id.empty()) {
        std::vector<std::string> pending_deletes = {gc_obj_id};
        std::vector<std::pair<std::string, std::vector<float>>> pending_adds;
        ApplyPendingVectorOps(pending_deletes, pending_adds);
      }
#endif
      return finish(cs);
    }

    if (prestige::internal::IsRetryableTxnStatus(cs)) {
      EmitCounter(opt_, "prestige.delete.retry_total", 1);
      SpanEvent(span.get(), "retry.commit");
      BackoffBeforeRetry(opt_, attempt, span.get());
      total_wait_us += prestige::internal::NowMicros() - attempt_start_us;
      attempt_start_us = prestige::internal::NowMicros();
      continue;
    }

    return finish(cs);
  }

  return finish(rocksdb::Status::TimedOut("Delete exceeded max_retries"));
}

#ifdef PRESTIGE_ENABLE_SEMANTIC

// Pending vector op format:
// Key: obj_id
// Value: [op_type:1 byte][embedding bytes if add]
// op_type: 'D' = delete, 'A' = add
constexpr char kVectorOpDelete = 'D';
constexpr char kVectorOpAdd = 'A';

void Store::ApplyPendingVectorOps(
    const std::vector<std::string>& pending_deletes,
    const std::vector<std::pair<std::string, std::vector<float>>>& pending_adds) {
  if (!vector_index_) return;

  // Apply deletes to in-memory index
  for (const auto& obj_id : pending_deletes) {
    vector_index_->MarkDeleted(obj_id);
  }

  // Apply adds to in-memory index
  for (const auto& [obj_id, embedding] : pending_adds) {
    if (!vector_index_->Add(embedding, obj_id)) {
      EmitCounter(opt_, "prestige.semantic.index_add_error_total", 1);
    }
  }

  // NOTE: We do NOT clear pending ops here. They remain in vector_pending_cf
  // until the vector index is saved to disk. This ensures crash recovery can
  // replay any ops that were applied to memory but not yet persisted.
  // The pending ops are cleared during periodic saves (after vector_index_->Save()).
}

// Clear pending vector ops from RocksDB after successful vector index save
void Store::ClearPendingVectorOps() {
  if (!db_ || !vector_pending_cf_) return;

  rocksdb::ReadOptions ro;
  rocksdb::WriteOptions wo;
  wo.sync = true;  // Ensure durability

  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, vector_pending_cf_));
  rocksdb::WriteBatch batch;

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    batch.Delete(vector_pending_cf_, it->key());
  }

  if (batch.Count() > 0) {
    rocksdb::Status s = db_->Write(wo, &batch);
    if (s.ok()) {
      EmitCounter(opt_, "prestige.semantic.pending_cleared", batch.Count());
    }
  }
}

void Store::ReplayPendingVectorOps() {
  if (!db_ || !vector_pending_cf_ || !vector_index_) return;

  std::vector<std::string> pending_deletes;
  std::vector<std::pair<std::string, std::vector<float>>> pending_adds;

  rocksdb::ReadOptions ro;
  std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(ro, vector_pending_cf_));

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    std::string obj_id(it->key().data(), it->key().size());
    std::string_view value(it->value().data(), it->value().size());

    if (value.empty()) continue;

    char op_type = value[0];
    if (op_type == kVectorOpDelete) {
      pending_deletes.push_back(std::move(obj_id));
    } else if (op_type == kVectorOpAdd && value.size() > 1) {
      std::vector<float> embedding;
      if (prestige::internal::DeserializeEmbedding(value.substr(1), &embedding)) {
        pending_adds.emplace_back(std::move(obj_id), std::move(embedding));
      }
    }
  }

  if (!pending_deletes.empty() || !pending_adds.empty()) {
    EmitCounter(opt_, "prestige.semantic.pending_replayed",
                pending_deletes.size() + pending_adds.size());
    ApplyPendingVectorOps(pending_deletes, pending_adds);
  }
}

rocksdb::Status Store::DeleteSemanticObject(rocksdb::Transaction* txn,
                                             const std::string& obj_id) {
  if (!txn) return rocksdb::Status::InvalidArgument("txn is null");

  // Write pending vector delete as part of transaction (WAL pattern).
  // The actual MarkDeleted call happens AFTER commit succeeds.
  // This ensures transactional consistency: if commit fails/retries,
  // the vector index is not mutated.
  std::string pending_value(1, kVectorOpDelete);
  rocksdb::Status s = txn->Put(vector_pending_cf_, rocksdb::Slice(obj_id),
                                rocksdb::Slice(pending_value));
  if (!s.ok()) return s;

  // Delete embedding from embeddings CF
  s = txn->Delete(embeddings_cf_, rocksdb::Slice(obj_id));
  if (!s.ok() && !s.IsNotFound()) return s;

  // Remove object bytes
  s = txn->Delete(objects_cf_, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

  // Remove refcount
  s = txn->Delete(refcount_cf_, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

  return rocksdb::Status::OK();
}

rocksdb::Status Store::PutImplSemantic(std::string_view user_key,
                                        std::string_view value_bytes,
                                        TraceSpan* span,
                                        uint64_t op_start_us) {
  // Compute embedding for the value
  const uint64_t embed_start_us = prestige::internal::NowMicros();

  // Truncate text if needed
  std::string_view text_to_embed = value_bytes;
  if (text_to_embed.size() > opt_.semantic_max_text_bytes) {
    text_to_embed = text_to_embed.substr(0, opt_.semantic_max_text_bytes);
  }

  auto embed_result = embedder_->Embed(text_to_embed);
  EmitHistogram(opt_, "prestige.put.embed_us",
                prestige::internal::NowMicros() - embed_start_us);

  if (!embed_result.success) {
    EmitCounter(opt_, "prestige.put.embed_error_total", 1);
    return rocksdb::Status::Corruption(
        "Embedding failed: " + embed_result.error_message);
  }

  const std::vector<float>& embedding = embed_result.embedding;
  std::string embedding_bytes = prestige::internal::SerializeEmbedding(embedding);

  bool dedup_hit_final = false;
  bool had_old_final = false;
  bool noop_overwrite = false;
  int attempts_used = 0;

  auto finish = [&](const rocksdb::Status& st) -> rocksdb::Status {
    const uint64_t dur_us = prestige::internal::NowMicros() - op_start_us;
    EmitHistogram(opt_, "prestige.put.latency_us", dur_us);
    EmitHistogram(opt_, "prestige.put.attempts", static_cast<uint64_t>(attempts_used));

    if (st.ok()) {
      EmitCounter(opt_, "prestige.put.ok_total", 1);
    } else if (st.IsTimedOut()) {
      EmitCounter(opt_, "prestige.put.timed_out_total", 1);
    } else {
      EmitCounter(opt_, "prestige.put.error_total", 1);
    }

    if (span) {
      SpanAttr(span, "latency_us", dur_us);
      SpanAttr(span, "attempts", static_cast<uint64_t>(attempts_used));
      SpanAttr(span, "dedup_hit", static_cast<uint64_t>(dedup_hit_final ? 1 : 0));
      SpanAttr(span, "had_old", static_cast<uint64_t>(had_old_final ? 1 : 0));
      SpanAttr(span, "noop_overwrite", static_cast<uint64_t>(noop_overwrite ? 1 : 0));
      SpanAttr(span, "status", StatusKind(rocksdb::Status::OK()));
      span->End(st);
    }
    return st;
  };

  rocksdb::WriteOptions wo;
  rocksdb::ReadOptions ro;

  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  // Search vector index for similar embeddings (outside transaction)
  const uint64_t search_start_us = prestige::internal::NowMicros();
  
  // If reranker is enabled, retrieve more candidates for better recall
  int search_k = (opt_.semantic_reranker_enabled && reranker_) 
                 ? opt_.semantic_reranker_top_k 
                 : opt_.semantic_search_k;
  
  auto candidates = vector_index_->Search(embedding, search_k);
  EmitHistogram(opt_, "prestige.semantic.lookup_us",
                prestige::internal::NowMicros() - search_start_us);
  EmitHistogram(opt_, "prestige.semantic.candidates_checked",
                static_cast<uint64_t>(candidates.size()));

  // Check candidates for similarity match
  std::string matched_obj_id;
  float best_score = -1.0f;
  
  // Use reranker if enabled and available
  if (opt_.semantic_reranker_enabled && reranker_ && !candidates.empty()) {
    const uint64_t rerank_start_us = prestige::internal::NowMicros();
    matched_obj_id = RerankCandidates(value_bytes, candidates, &best_score);
    EmitHistogram(opt_, "prestige.semantic.rerank_us",
                  prestige::internal::NowMicros() - rerank_start_us);
    
    if (!matched_obj_id.empty()) {
      dedup_hit_final = true;
      EmitCounter(opt_, "prestige.semantic.hit_total", 1);
      EmitCounter(opt_, "prestige.semantic.reranker_hit", 1);
    }
  }
  
  // Fall back to embedding-based matching if reranker didn't find a match
  // or if reranker is not enabled
  if (matched_obj_id.empty()) {
    if (opt_.semantic_verify_exact) {
    // Exact verification: load stored embeddings and compute exact cosine similarity.
    // This is more accurate than relying on approximate HNSW distances.
    rocksdb::ReadOptions ro_snap;
    size_t dim = embedding.size();

    // Collect all candidates with their exact cosine similarities
    struct CandidateScore {
      std::string object_id;
      float cos_sim;
      std::vector<float> stored_embedding;  // Cached for RNN check
    };
    std::vector<CandidateScore> scored_candidates;
    scored_candidates.reserve(candidates.size());

    for (const auto& candidate : candidates) {
      // Load stored embedding for this candidate
      std::string stored_embedding_bytes;
      rocksdb::Status s = db_->Get(ro_snap, embeddings_cf_,
                                   rocksdb::Slice(candidate.object_id),
                                   &stored_embedding_bytes);
      if (!s.ok()) {
        continue;  // Embedding not found (possibly deleted), skip candidate
      }

      // Parse stored embedding
      if (stored_embedding_bytes.size() != dim * sizeof(float)) {
        continue;  // Dimension mismatch, skip
      }
      const float* stored = reinterpret_cast<const float*>(stored_embedding_bytes.data());

      // Compute exact cosine similarity (dot product for normalized vectors)
      float cos_sim = 0.0f;
      for (size_t i = 0; i < dim; ++i) {
        cos_sim += embedding[i] * stored[i];
      }

      if (cos_sim >= opt_.semantic_threshold) {
        CandidateScore cs;
        cs.object_id = candidate.object_id;
        cs.cos_sim = cos_sim;
        // Cache embedding for RNN and margin gating checks
        if (opt_.semantic_rnn_enabled || opt_.semantic_margin_enabled) {
          cs.stored_embedding.assign(stored, stored + dim);
        }
        scored_candidates.push_back(std::move(cs));
      } else if (opt_.semantic_judge_enabled && judge_llm_ &&
                 cos_sim >= opt_.semantic_judge_threshold) {
        // Gray zone candidate: above judge threshold but below semantic threshold
        // Will be evaluated by judge LLM if no above-threshold matches found
        CandidateScore cs;
        cs.object_id = candidate.object_id;
        cs.cos_sim = cos_sim;
        cs.stored_embedding.assign(stored, stored + dim);
        scored_candidates.push_back(std::move(cs));
      }
    }

    // Sort by cosine similarity descending (best match first)
    std::sort(scored_candidates.begin(), scored_candidates.end(),
              [](const CandidateScore& a, const CandidateScore& b) {
                return a.cos_sim > b.cos_sim;
              });

    // Apply RNN + margin gating checks to find valid match
    bool need_b_search = opt_.semantic_rnn_enabled || opt_.semantic_margin_enabled;
    int rnn_k = opt_.semantic_rnn_k > 0 ? opt_.semantic_rnn_k : opt_.semantic_search_k;

    for (size_t i = 0; i < scored_candidates.size(); ++i) {
      const auto& best = scored_candidates[i];
      bool accept = true;

      // Margin gating from A's perspective: cos(A,B) - cos(A,2nd_best) >= margin
      if (opt_.semantic_margin_enabled) {
        float second_best_sim = (i + 1 < scored_candidates.size())
                                 ? scored_candidates[i + 1].cos_sim
                                 : 0.0f;
        float margin_a = best.cos_sim - second_best_sim;

        if (margin_a < opt_.semantic_margin_threshold) {
          EmitCounter(opt_, "prestige.semantic.margin_reject_a", 1);
          accept = false;
        }
      }

      // Search from B's embedding (shared for RNN and margin checks)
      std::vector<internal::SearchResult> b_neighbors;
      if (need_b_search && accept) {
        // Request enough neighbors for both RNN (k) and margin (need 2nd best)
        int search_k = rnn_k + 1;  // +1 to skip self and still have k neighbors
        b_neighbors = vector_index_->Search(best.stored_embedding, search_k);
      }

      // Margin gating from B's perspective: cos(B,A) - cos(B,2nd_best) >= margin
      if (opt_.semantic_margin_enabled && accept && !b_neighbors.empty()) {
        // Find B's second-best neighbor (first non-self neighbor after position 0)
        float b_second_best = 0.0f;
        int valid_neighbor_count = 0;
        for (const auto& neighbor : b_neighbors) {
          if (neighbor.object_id == best.object_id) continue;  // Skip self
          valid_neighbor_count++;
          if (valid_neighbor_count == 2) {
            // This is the second-best neighbor
            // Convert L2 squared to cosine: cos = 1 - d²/2
            b_second_best = 1.0f - neighbor.distance / 2.0f;
            break;
          }
        }

        float margin_b = best.cos_sim - b_second_best;
        if (margin_b < opt_.semantic_margin_threshold) {
          EmitCounter(opt_, "prestige.semantic.margin_reject_b", 1);
          accept = false;
        }
      }

      // Reciprocal kNN check: verify A is in B's top-k neighbors
      if (opt_.semantic_rnn_enabled && accept && !b_neighbors.empty()) {
        // A is in B's top-k if cos(A,B) >= cos(B, B_kth)
        // Find B's k-th valid neighbor (excluding self)
        float kth_neighbor_cos = 0.0f;
        int valid_count = 0;
        for (const auto& neighbor : b_neighbors) {
          if (neighbor.object_id == best.object_id) continue;  // Skip self
          valid_count++;
          if (valid_count == rnn_k) {
            // This is the k-th neighbor
            kth_neighbor_cos = 1.0f - neighbor.distance / 2.0f;
            break;
          }
        }

        // If fewer than k neighbors, use the last one's similarity
        if (valid_count > 0 && valid_count < rnn_k) {
          for (const auto& neighbor : b_neighbors) {
            if (neighbor.object_id != best.object_id) {
              kth_neighbor_cos = 1.0f - neighbor.distance / 2.0f;
            }
          }
        }

        // A would be in B's top-k if A's similarity is better than B's k-th neighbor
        if (best.cos_sim < kth_neighbor_cos) {
          EmitCounter(opt_, "prestige.semantic.rnn_reject", 1);
          accept = false;
        }
      }

      if (accept) {
        // Check if this is a gray zone candidate that needs judge evaluation
        if (best.cos_sim < opt_.semantic_threshold &&
            opt_.semantic_judge_enabled && judge_llm_) {
          // Gray zone candidate: evaluate with judge LLM
          const uint64_t judge_start_us = prestige::internal::NowMicros();

          // Retrieve the candidate's text for judge evaluation
          std::string candidate_bytes;
          rocksdb::Status s = db_->Get(ro_snap, objects_cf_,
                                       rocksdb::Slice(best.object_id),
                                       &candidate_bytes);
          if (s.ok()) {
            bool is_duplicate = JudgeCandidate(value_bytes, candidate_bytes, best.cos_sim);
            EmitHistogram(opt_, "prestige.semantic.judge_us",
                          prestige::internal::NowMicros() - judge_start_us);

            if (is_duplicate) {
              matched_obj_id = best.object_id;
              best_score = best.cos_sim;
              dedup_hit_final = true;
              EmitCounter(opt_, "prestige.semantic.hit_total", 1);
              EmitCounter(opt_, "prestige.semantic.judge_accepted", 1);
              break;
            } else {
              EmitCounter(opt_, "prestige.semantic.judge_rejected", 1);
              // Judge said not a duplicate, continue to next candidate
              continue;
            }
          }
        } else {
          // Above threshold candidate: accept directly
          matched_obj_id = best.object_id;
          best_score = best.cos_sim;
          dedup_hit_final = true;
          EmitCounter(opt_, "prestige.semantic.hit_total", 1);
          EmitCounter(opt_, "prestige.semantic.exact_verified", 1);
          if (opt_.semantic_rnn_enabled) {
            EmitCounter(opt_, "prestige.semantic.rnn_accepted", 1);
          }
          if (opt_.semantic_margin_enabled) {
            EmitCounter(opt_, "prestige.semantic.margin_accepted", 1);
          }
          break;
        }
      }
    }
  } else {
    // Approximate verification: use HNSW L2 distance (L2 squared for hnswlib)
    // L2² for normalized vectors: d² = 2 * (1 - cos_sim)
    // We want cos_sim >= threshold, which means d² <= 2 * (1 - threshold)
    // Note: RNN and margin gating require exact verification mode
    float max_l2_sq = 2.0f * (1.0f - opt_.semantic_threshold);

    for (const auto& candidate : candidates) {
      if (candidate.distance <= max_l2_sq) {
        // Found a semantic match (approximate)
        matched_obj_id = candidate.object_id;
        dedup_hit_final = true;
        EmitCounter(opt_, "prestige.semantic.hit_total", 1);
        break;
      }
    }
  }
  }  // End of embedding-based matching fallback

  if (!dedup_hit_final) {
    EmitCounter(opt_, "prestige.semantic.miss_total", 1);
  }

  for (int attempt = 0; attempt < opt_.max_retries; ++attempt) {
    attempts_used = attempt + 1;

    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) return finish(rocksdb::Status::IOError("BeginTransaction returned null"));

    // Lock user_key mapping (detect overwrite)
    std::string old_obj_id;
    bool had_old = false;
    {
      rocksdb::Status s = txn->GetForUpdate(
          ro, user_kv_cf_,
          rocksdb::Slice(user_key.data(), user_key.size()),
          &old_obj_id);

      if (s.ok()) {
        had_old = true;
      } else if (!s.IsNotFound()) {
        if (prestige::internal::IsRetryableTxnStatus(s)) {
          EmitCounter(opt_, "prestige.put.retry_total", 1);
          SpanEvent(span, "retry.user_key_lock");
          BackoffBeforeRetry(opt_, attempt, span);
          continue;
        }
        return finish(s);
      }
    }

    std::string obj_id;
    bool creating_new_object = false;

    if (dedup_hit_final) {
      // Semantic match found outside transaction - must verify object still exists.
      //
      // Why this check is needed:
      // The vector search happens BEFORE the transaction starts. Between finding
      // matched_obj_id and now, another thread could have:
      //   1. Deleted the last user_key pointing to matched_obj_id
      //   2. Decremented its refcount to 0
      //   3. GC'd the object, deleting: object_store, refcount row, embeddings
      //
      // Without this check, AdjustRefcountLocked would see "refcount not found",
      // interpret it as "this is a brand new object, start at 0", and increment
      // to 1. The transaction commits, creating:
      //   - user_key -> matched_obj_id (mapping exists)
      //   - refcount[matched_obj_id] = 1 (newly written)
      //   - object_store[matched_obj_id] = MISSING (deleted by GC!)
      //
      // This causes Get(user_key) to fail with NotFound on the object lookup.
      //
      // The fix: GetForUpdate on object bytes to lock and verify existence.
      // If the matched object was deleted, we cannot reuse it. Instead, we
      // store the new value in a fresh object (same as if no match was found).
      std::string existing_bytes;
      rocksdb::Status s = txn->GetForUpdate(ro, objects_cf_,
                                             rocksdb::Slice(matched_obj_id),
                                             &existing_bytes);
      if (s.ok()) {
        // Object still exists - safe to reuse
        obj_id = matched_obj_id;
      } else if (s.IsNotFound()) {
        // Matched object was GC'd - store value in a new object instead
        EmitCounter(opt_, "prestige.semantic.stale_match_total", 1);
        SpanEvent(span, "semantic.stale_match");
        dedup_hit_final = false;
        creating_new_object = true;
      } else {
        if (prestige::internal::IsRetryableTxnStatus(s)) {
          EmitCounter(opt_, "prestige.put.retry_total", 1);
          SpanEvent(span, "retry.object_lock");
          BackoffBeforeRetry(opt_, attempt, span);
          continue;
        }
        return finish(s);
      }
    }

    if (!dedup_hit_final) {
      // No match - create new object
      EmitCounter(opt_, "prestige.put.object_created_total", 1);
      creating_new_object = true;

      auto new_id = prestige::internal::RandomObjectId128();
      obj_id = prestige::internal::ToBytes(new_id.data(), new_id.size());

      // Store object bytes
      rocksdb::Status s = txn->Put(
          objects_cf_, rocksdb::Slice(obj_id),
          rocksdb::Slice(value_bytes.data(), value_bytes.size()));
      if (!s.ok()) return finish(s);

      // Update total store size
      total_store_bytes_.fetch_add(value_bytes.size());

      // Store embedding
      s = txn->Put(embeddings_cf_, rocksdb::Slice(obj_id),
                   rocksdb::Slice(embedding_bytes));
      if (!s.ok()) return finish(s);

      // Initialize refcount to 0 (will be incremented below)
      s = txn->Put(refcount_cf_, rocksdb::Slice(obj_id),
                   rocksdb::Slice(prestige::internal::EncodeU64LE(0)));
      if (!s.ok()) return finish(s);

      // Write pending vector add as part of transaction (WAL pattern).
      // The actual Add call happens AFTER commit succeeds.
      std::string pending_value(1, kVectorOpAdd);
      pending_value.append(embedding_bytes);
      s = txn->Put(vector_pending_cf_, rocksdb::Slice(obj_id),
                   rocksdb::Slice(pending_value));
      if (!s.ok()) return finish(s);
    }

    had_old_final = had_old;

    // If overwrite maps to same object_id, nothing to do
    if (had_old && old_obj_id == obj_id) {
      noop_overwrite = true;
      EmitCounter(opt_, "prestige.put.noop_overwrite_total", 1);
      txn->Rollback();
      return finish(rocksdb::Status::OK());
    }

    // user_key -> obj_id
    {
      rocksdb::Status s = txn->Put(
          user_kv_cf_,
          rocksdb::Slice(user_key.data(), user_key.size()),
          rocksdb::Slice(obj_id));
      if (!s.ok()) return finish(s);
    }

    // Incref(new)
    {
      uint64_t new_cnt = 0;
      rocksdb::Status s = AdjustRefcountLocked(txn.get(), refcount_cf_, obj_id, +1, &new_cnt);
      if (!s.ok()) return finish(s);
    }

    // Track GC'd object for post-commit vector index update
    bool gc_old_object = false;
    std::string gc_old_obj_id;

    // Decref(old) and GC if needed
    if (had_old && old_obj_id != obj_id) {
      uint64_t old_cnt = 0;
      rocksdb::Status s = AdjustRefcountLocked(txn.get(), refcount_cf_, old_obj_id, -1, &old_cnt);
      if (!s.ok()) return finish(s);

      if (opt_.enable_gc && old_cnt == 0) {
        s = DeleteSemanticObject(txn.get(), old_obj_id);
        if (!s.ok()) return finish(s);
        gc_old_object = true;
        gc_old_obj_id = old_obj_id;

        EmitCounter(opt_, "prestige.gc.deleted_objects_total", 1);
        SpanEvent(span, "gc.delete_object");
      }
    }

    const uint64_t commit_start_us = prestige::internal::NowMicros();
    rocksdb::Status cs = txn->Commit();
    EmitHistogram(opt_, "prestige.put.commit_us",
                  prestige::internal::NowMicros() - commit_start_us);

    if (cs.ok()) {
      // Apply pending vector ops AFTER successful commit
      std::vector<std::string> pending_deletes;
      std::vector<std::pair<std::string, std::vector<float>>> pending_adds;

      if (creating_new_object) {
        pending_adds.emplace_back(obj_id, embedding);
      }
      if (gc_old_object && !gc_old_obj_id.empty()) {
        pending_deletes.push_back(gc_old_obj_id);
      }

      if (!pending_adds.empty() || !pending_deletes.empty()) {
        ApplyPendingVectorOps(pending_deletes, pending_adds);
      }

      // Periodic index save with synchronization
      if (creating_new_object) {
        semantic_inserts_since_save_++;
        if (opt_.semantic_index_save_interval > 0 &&
            semantic_inserts_since_save_ >= static_cast<uint64_t>(opt_.semantic_index_save_interval)) {
          // Sync RocksDB WAL before saving vector index to ensure consistency
          rocksdb::Status sync_status = db_->SyncWAL();
          if (sync_status.ok()) {
            if (vector_index_->Save(vector_index_path_)) {
              semantic_inserts_since_save_ = 0;
              EmitCounter(opt_, "prestige.semantic.index_save_ok_total", 1);

              // Now that vector index is saved, clear pending ops from RocksDB
              ClearPendingVectorOps();

              // Check if compaction is needed
              if (opt_.semantic_compact_threshold > 0) {
                size_t deleted = vector_index_->DeletedCount();
                if (deleted >= opt_.semantic_compact_threshold) {
                  if (vector_index_->Compact()) {
                    vector_index_->Save(vector_index_path_);
                    EmitCounter(opt_, "prestige.semantic.compact_ok_total", 1);
                  }
                }
              }
            } else {
              EmitCounter(opt_, "prestige.semantic.index_save_error_total", 1);
            }
          } else {
            EmitCounter(opt_, "prestige.semantic.wal_sync_error_total", 1);
          }
        }
      }
      return finish(cs);
    }

    if (prestige::internal::IsRetryableTxnStatus(cs)) {
      EmitCounter(opt_, "prestige.put.retry_total", 1);
      SpanEvent(span, "retry.commit");
      BackoffBeforeRetry(opt_, attempt, span);
      continue;
    }

    return finish(cs);
  }

  return finish(rocksdb::Status::TimedOut("Put exceeded max_retries"));
}

std::string Store::RerankCandidates(std::string_view query_text,
                                    const std::vector<internal::SearchResult>& candidates,
                                    float* best_score_out) const {
  // Load candidate texts from RocksDB
  std::vector<std::string> candidate_texts;
  std::vector<std::string> candidate_ids;
  
  rocksdb::ReadOptions ro;
  for (const auto& candidate : candidates) {
    std::string text;
    rocksdb::Status s = db_->Get(ro, objects_cf_, 
                                 rocksdb::Slice(candidate.object_id), 
                                 &text);
    if (s.ok()) {
      candidate_texts.push_back(text);
      candidate_ids.push_back(candidate.object_id);
    }
  }
  
  if (candidate_texts.empty()) {
    return "";  // No valid candidates to rerank
  }
  
  // Score candidates with reranker
  std::vector<internal::ScoringResult> scores;
  
  if (opt_.semantic_reranker_batch_size > 1) {
    // Process in batches for efficiency
    for (size_t i = 0; i < candidate_texts.size(); i += opt_.semantic_reranker_batch_size) {
      size_t batch_end = std::min(i + static_cast<size_t>(opt_.semantic_reranker_batch_size), 
                                  candidate_texts.size());
      std::vector<std::string_view> batch;
      for (size_t j = i; j < batch_end; ++j) {
        batch.push_back(candidate_texts[j]);
      }
      
      auto batch_results = reranker_->ScoreBatch(query_text, batch);
      scores.insert(scores.end(), batch_results.begin(), batch_results.end());
    }
  } else {
    // Score one at a time
    for (const auto& candidate_text : candidate_texts) {
      scores.push_back(reranker_->Score(query_text, candidate_text));
    }
  }
  
  // Find best match above threshold
  std::string best_match_id;
  float best_score = -1.0f;
  
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i].success && 
        scores[i].score >= opt_.semantic_reranker_threshold && 
        scores[i].score > best_score) {
      best_score = scores[i].score;
      best_match_id = candidate_ids[i];
    }
  }
  
  // Emit metrics
  if (!best_match_id.empty()) {
    EmitHistogram(opt_, "prestige.semantic.reranker_score", 
                  static_cast<uint64_t>(best_score * 1000));  // Convert to millis for histogram
  }
  
  if (best_score_out) {
    *best_score_out = best_score;
  }

  return best_match_id;
}

bool Store::JudgeCandidate(std::string_view query_text,
                           std::string_view candidate_text,
                           float similarity_score) const {
  if (!judge_llm_) {
    return false;
  }

  // Truncate texts if they exceed judge's max length
  size_t max_len = judge_llm_->MaxTextLength();
  std::string_view truncated_query = query_text.substr(0, std::min(query_text.size(), max_len));
  std::string_view truncated_candidate = candidate_text.substr(0, std::min(candidate_text.size(), max_len));

  // Call the judge LLM
  auto result = judge_llm_->Judge(truncated_query, truncated_candidate, similarity_score);

  if (!result.success) {
    // Log error and return false (conservative: don't mark as duplicate on error)
    EmitCounter(opt_, "prestige.semantic.judge_error", 1);
    return false;
  }

  // Emit confidence as histogram (scaled to 0-1000 for millis precision)
  EmitHistogram(opt_, "prestige.semantic.judge_confidence",
                static_cast<uint64_t>(result.confidence * 1000));

  return result.is_duplicate;
}

#endif  // PRESTIGE_ENABLE_SEMANTIC

}  // namespace prestige
