#include <prestige/store.hpp>

#include <rocksdb/advanced_cache.h>
#include <rocksdb/cache.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/statistics.h>
#include <rocksdb/table.h>
#include <rocksdb/utilities/transaction.h>

#include <cstring>

#include <prestige/internal.hpp>
#include <prestige/normalize.hpp>

#ifdef PRESTIGE_ENABLE_SEMANTIC
#include <prestige/embedder.hpp>
#include <prestige/vector_index.hpp>
#endif

namespace prestige {

namespace {

constexpr const char* kUserKvCF      = "prestige_user_kv";
constexpr const char* kObjectStoreCF = "prestige_object_store";
constexpr const char* kDedupIndexCF  = "prestige_dedup_index";
constexpr const char* kRefcountCF    = "prestige_refcount";
constexpr const char* kObjectMetaCF  = "prestige_object_meta";
constexpr const char* kLRUIndexCF    = "prestige_lru_index";
#ifdef PRESTIGE_ENABLE_SEMANTIC
constexpr const char* kEmbeddingsCF  = "prestige_embeddings";
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

rocksdb::Status Store::Open(const std::string& db_path,
                            std::unique_ptr<Store>* out,
                            const Options& opt) {
  if (!out) return rocksdb::Status::InvalidArgument("out is null");

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Validate semantic mode options
  if (opt.dedup_mode == DedupMode::kSemantic) {
    if (opt.semantic_model_path.empty()) {
      return rocksdb::Status::InvalidArgument(
          "semantic_model_path is required when dedup_mode == kSemantic");
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

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Add embeddings CF for semantic mode
  if (opt.dedup_mode == DedupMode::kSemantic) {
    cfs.emplace_back(kEmbeddingsCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
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

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Initialize semantic dedup components
  if (opt.dedup_mode == DedupMode::kSemantic) {
    store->embeddings_cf_ = store->handles_[7];

    // Create embedder
    std::string embedder_error;
    store->embedder_ = internal::Embedder::Create(
        opt.semantic_model_path,
        opt.semantic_model_type == SemanticModel::kMiniLM
            ? internal::EmbedderModelType::kMiniLM
            : internal::EmbedderModelType::kBGESmall,
        &embedder_error);

    if (!store->embedder_) {
      store->Close();
      return rocksdb::Status::InvalidArgument(
          "Failed to create embedder: " + embedder_error);
    }

    // Create vector index
    size_t dimension = store->embedder_->Dimension();
    store->vector_index_ = internal::CreateHNSWIndex(
        dimension,
        10000,  // Initial max elements (will grow as needed)
        opt.hnsw_m,
        opt.hnsw_ef_construction);

    if (!store->vector_index_) {
      store->Close();
      return rocksdb::Status::InvalidArgument("Failed to create vector index");
    }

    store->vector_index_->SetSearchParam("ef_search", opt.hnsw_ef_search);

    // Load existing index if present
    store->vector_index_path_ = db_path + ".vec_index";
    if (!store->vector_index_->Load(store->vector_index_path_)) {
      // Load failure is not fatal if the file doesn't exist yet
      // But if it exists and is corrupt, we should warn (for now just continue)
    }
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
        uint64_t now_us = prestige::internal::WallClockMicros();
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
          uint64_t now_us = prestige::internal::WallClockMicros();
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

void Store::EmitCacheMetrics() {
  if (!opt_.metrics) return;

  // Cache fill rate and usage
  if (block_cache_) {
    size_t usage = block_cache_->GetUsage();
    size_t capacity = block_cache_->GetCapacity();
    double fill_ratio = capacity > 0 ? static_cast<double>(usage) / capacity : 0.0;

    EmitGauge(opt_, "prestige.cache.fill_ratio", fill_ratio);
    EmitGauge(opt_, "prestige.cache.usage_bytes", static_cast<double>(usage));
    EmitGauge(opt_, "prestige.cache.capacity_bytes", static_cast<double>(capacity));
  }

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
  uint64_t now_us = prestige::internal::WallClockMicros();
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
  uint64_t now_us = prestige::internal::WallClockMicros();
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

  uint64_t now_us = prestige::internal::WallClockMicros();
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

void Store::Close() {
  if (!db_) return;

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Save vector index before closing
  if (vector_index_ && !vector_index_path_.empty()) {
    vector_index_->Save(vector_index_path_);
  }

  // Release semantic resources
  vector_index_.reset();
  embedder_.reset();
  embeddings_cf_ = nullptr;
#endif

  for (auto* h : handles_) delete h;
  handles_.clear();
  delete db_;
  db_ = nullptr;
  user_kv_cf_ = objects_cf_ = dedup_cf_ = refcount_cf_ = meta_cf_ = lru_cf_ = nullptr;
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

rocksdb::Status Store::PutImpl(std::string_view user_key, std::string_view value_bytes) {
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
        uint64_t now_us = prestige::internal::WallClockMicros();
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

    const uint64_t commit_start_us = prestige::internal::NowMicros();
    rocksdb::Status cs = txn->Commit();
    EmitHistogram(opt_, "prestige.put.commit_us", prestige::internal::NowMicros() - commit_start_us);

    if (cs.ok()) return finish(cs);

    if (prestige::internal::IsRetryableTxnStatus(cs)) {
      EmitCounter(opt_, "prestige.put.retry_total", 1);
      SpanEvent(span.get(), "retry.commit");
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

    if (opt_.enable_gc && cnt == 0) {
#ifdef PRESTIGE_ENABLE_SEMANTIC
      if (opt_.dedup_mode == DedupMode::kSemantic) {
        s = DeleteSemanticObject(txn.get(), obj_id);
        batch_writes += 3;  // embeddings, objects, refcount deletes
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

    if (cs.ok()) return finish(cs);

    if (prestige::internal::IsRetryableTxnStatus(cs)) {
      EmitCounter(opt_, "prestige.delete.retry_total", 1);
      SpanEvent(span.get(), "retry.commit");
      total_wait_us += prestige::internal::NowMicros() - attempt_start_us;
      attempt_start_us = prestige::internal::NowMicros();
      continue;
    }

    return finish(cs);
  }

  return finish(rocksdb::Status::TimedOut("Delete exceeded max_retries"));
}

#ifdef PRESTIGE_ENABLE_SEMANTIC

rocksdb::Status Store::DeleteSemanticObject(rocksdb::Transaction* txn,
                                             const std::string& obj_id) {
  if (!txn) return rocksdb::Status::InvalidArgument("txn is null");

  rocksdb::ReadOptions ro;

  // Mark as deleted in vector index (soft delete)
  if (vector_index_) {
    vector_index_->MarkDeleted(obj_id);
  }

  // Delete embedding from embeddings CF
  rocksdb::Status s = txn->Delete(embeddings_cf_, rocksdb::Slice(obj_id));
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
  auto candidates = vector_index_->Search(embedding, opt_.semantic_search_k);
  EmitHistogram(opt_, "prestige.semantic.lookup_us",
                prestige::internal::NowMicros() - search_start_us);
  EmitHistogram(opt_, "prestige.semantic.candidates_checked",
                static_cast<uint64_t>(candidates.size()));

  // Check candidates for similarity match
  // L2 distance for normalized vectors: d = 2 * (1 - cos_sim)
  // So cos_sim = 1 - d/2
  // We want cos_sim >= threshold, which means d <= 2 * (1 - threshold)
  float max_l2_dist = 2.0f * (1.0f - opt_.semantic_threshold);
  std::string matched_obj_id;

  for (const auto& candidate : candidates) {
    if (candidate.distance <= max_l2_dist) {
      // Found a semantic match
      matched_obj_id = candidate.object_id;
      dedup_hit_final = true;
      EmitCounter(opt_, "prestige.semantic.hit_total", 1);
      break;
    }
  }

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
          continue;
        }
        return finish(s);
      }
    }

    std::string obj_id;

    if (dedup_hit_final) {
      // Semantic match found - reuse existing object
      obj_id = matched_obj_id;
    } else {
      // No match - create new object
      EmitCounter(opt_, "prestige.put.object_created_total", 1);

      auto new_id = prestige::internal::RandomObjectId128();
      obj_id = prestige::internal::ToBytes(new_id.data(), new_id.size());

      // Store object bytes
      rocksdb::Status s = txn->Put(
          objects_cf_, rocksdb::Slice(obj_id),
          rocksdb::Slice(value_bytes.data(), value_bytes.size()));
      if (!s.ok()) return finish(s);

      // Store embedding
      s = txn->Put(embeddings_cf_, rocksdb::Slice(obj_id),
                   rocksdb::Slice(embedding_bytes));
      if (!s.ok()) return finish(s);

      // Initialize refcount to 0 (will be incremented below)
      s = txn->Put(refcount_cf_, rocksdb::Slice(obj_id),
                   rocksdb::Slice(prestige::internal::EncodeU64LE(0)));
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

    // Decref(old) and GC if needed
    if (had_old && old_obj_id != obj_id) {
      uint64_t old_cnt = 0;
      rocksdb::Status s = AdjustRefcountLocked(txn.get(), refcount_cf_, old_obj_id, -1, &old_cnt);
      if (!s.ok()) return finish(s);

      if (opt_.enable_gc && old_cnt == 0) {
        s = DeleteSemanticObject(txn.get(), old_obj_id);
        if (!s.ok()) return finish(s);

        EmitCounter(opt_, "prestige.gc.deleted_objects_total", 1);
        SpanEvent(span, "gc.delete_object");
      }
    }

    const uint64_t commit_start_us = prestige::internal::NowMicros();
    rocksdb::Status cs = txn->Commit();
    EmitHistogram(opt_, "prestige.put.commit_us",
                  prestige::internal::NowMicros() - commit_start_us);

    if (cs.ok()) {
      // Add to vector index after successful commit (if new object)
      if (!dedup_hit_final) {
        if (!vector_index_->Add(embedding, obj_id)) {
          // Log warning but don't fail - object is stored, just not indexed
          EmitCounter(opt_, "prestige.semantic.index_add_error_total", 1);
        }

        // Periodic index save
        semantic_inserts_since_save_++;
        if (opt_.semantic_index_save_interval > 0 &&
            semantic_inserts_since_save_ >= static_cast<uint64_t>(opt_.semantic_index_save_interval)) {
          if (vector_index_->Save(vector_index_path_)) {
            semantic_inserts_since_save_ = 0;
          }
        }
      }
      return finish(cs);
    }

    if (prestige::internal::IsRetryableTxnStatus(cs)) {
      EmitCounter(opt_, "prestige.put.retry_total", 1);
      SpanEvent(span, "retry.commit");
      continue;
    }

    return finish(cs);
  }

  return finish(rocksdb::Status::TimedOut("Put exceeded max_retries"));
}

#endif  // PRESTIGE_ENABLE_SEMANTIC

}  // namespace prestige
