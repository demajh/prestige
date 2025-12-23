#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <rocksdb/cache.h>
#include <rocksdb/options.h>
#include <rocksdb/statistics.h>
#include <rocksdb/status.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction_db.h>

namespace prestige {

/** A minimal metrics sink interface (counters + histograms + gauges). */
struct MetricsSink {
  virtual ~MetricsSink() = default;

  /** Monotonic counters (e.g., calls, retries, dedup hits). */
  virtual void Counter(std::string_view name, uint64_t delta) = 0;

  /** Histograms (e.g., latency in microseconds, sizes in bytes). */
  virtual void Histogram(std::string_view name, uint64_t value) = 0;

  /** Gauges for point-in-time values (e.g., cache fill ratio, queue depth).
   *  Default implementation does nothing for backwards compatibility. */
  virtual void Gauge(std::string_view name, double value) { (void)name; (void)value; }
};

/** A trace span interface (very small surface area). */
struct TraceSpan {
  virtual ~TraceSpan() = default;

  virtual void SetAttribute(std::string_view key, uint64_t value) = 0;
  virtual void SetAttribute(std::string_view key, std::string_view value) = 0;
  virtual void AddEvent(std::string_view name) = 0;

  /** Must be called exactly once to finish the span. */
  virtual void End(const rocksdb::Status& status) = 0;
};

/** A tracer creates spans. If unset, tracing is disabled. */
struct Tracer {
  virtual ~Tracer() = default;
  virtual std::unique_ptr<TraceSpan> StartSpan(std::string_view name) = 0;
};

#ifdef PRESTIGE_ENABLE_SEMANTIC
// Forward declarations for semantic dedup
namespace internal {
class Embedder;
class VectorIndex;
}  // namespace internal
#endif

/** Deduplication mode: either exact (SHA-256) or semantic (embeddings). */
enum class DedupMode {
  kExact,     // SHA-256 content hash (default, current behavior)
  kSemantic   // Embedding similarity via vector index
};

/** Supported embedding models for semantic dedup. */
enum class SemanticModel {
  kMiniLM,    // all-MiniLM-L6-v2 (384 dimensions)
  kBGESmall   // BGE-small-en-v1.5 (384 dimensions)
};

/** Vector index backend for semantic similarity search. */
enum class SemanticIndexType {
  kHNSW,      // hnswlib - graph-based, excellent recall, recommended
  kFAISS     // FAISS IVF+PQ - for large scale / memory constrained
};

/**
 * Options for the prestige unique value store.
 *
 * These are layered on top of RocksDB's Options/TransactionDBOptions. The store
 * creates its own column families and uses RocksDB TransactionDB for atomic,
 * concurrency-safe deduplication.
 */
struct Options {
  // RocksDB performance knobs
  size_t block_cache_bytes = 256ull * 1024ull * 1024ull;
  int bloom_bits_per_key = 10;

  // Transaction behavior
  int lock_timeout_ms = 2000;
  int max_retries = 16;

  // GC behavior
  bool enable_gc = true;

  // ---------------------------------------------------------------------------
  // Cache behavior settings (TTL and LRU eviction)
  // ---------------------------------------------------------------------------

  // Default TTL for objects in seconds (0 = no expiration).
  // Objects older than this will return NotFound on Get.
  uint64_t default_ttl_seconds = 0;

  // Maximum store size in bytes (0 = unlimited).
  // When exceeded, LRU eviction can be triggered via EvictLRU().
  uint64_t max_store_bytes = 0;

  // Target store size ratio for eviction (0.0-1.0).
  // When evicting, remove objects until this ratio of max_store_bytes is reached.
  // E.g., 0.8 means evict until store is at 80% of max_store_bytes.
  double eviction_target_ratio = 0.8;

  // Enable tracking of last access time (required for LRU eviction).
  // Has slight write overhead on every Get().
  bool track_access_time = true;

  // Observability hooks (optional)
  //
  // If set, Store operations will emit a small number of counters/histograms
  // and attach attributes/events to spans.
  std::shared_ptr<MetricsSink> metrics;
  std::shared_ptr<Tracer> tracer;

  // ---------------------------------------------------------------------------
  // Semantic deduplication settings (only used when dedup_mode == kSemantic)
  // ---------------------------------------------------------------------------

  // Deduplication mode: kExact (SHA-256, default) or kSemantic (embeddings)
  DedupMode dedup_mode = DedupMode::kExact;

  // Path to ONNX model file (REQUIRED for semantic mode)
  // The vocabulary file (vocab.txt) is expected to be in the same directory.
  std::string semantic_model_path;

  // Which embedding model architecture the ONNX file contains
  SemanticModel semantic_model_type = SemanticModel::kMiniLM;

  // Cosine similarity threshold for dedup [0.0, 1.0] (REQUIRED for semantic mode)
  // Values >= this threshold are considered duplicates.
  // Set to -1.0 by default to force user to explicitly set it.
  float semantic_threshold = -1.0f;

  // Maximum text bytes to embed (longer texts are truncated)
  size_t semantic_max_text_bytes = 8192;

  // Vector index backend
  SemanticIndexType semantic_index_type = SemanticIndexType::kHNSW;

  // Number of nearest neighbors to retrieve during search
  int semantic_search_k = 10;

  // Auto-save vector index every N inserts (0 = disabled, save only on Close)
  int semantic_index_save_interval = 1000;

  // HNSW-specific parameters (when semantic_index_type == kHNSW)
  int hnsw_m = 16;                // Max connections per node
  int hnsw_ef_construction = 200; // Build-time search depth
  int hnsw_ef_search = 50;        // Query-time search depth

  // FAISS-specific parameters (when semantic_index_type == kFAISS)
  int faiss_nlist = 100;          // Number of IVF clusters
  int faiss_nprobe = 10;          // Clusters to search at query time
  int faiss_pq_m = 48;            // PQ sub-quantizers (must divide 384)
  int faiss_pq_nbits = 8;         // Bits per sub-quantizer
};

/**
 * Health statistics for the store.
 * Returned by Store::GetHealth().
 */
struct HealthStats {
  uint64_t total_keys = 0;          // Number of user keys
  uint64_t total_objects = 0;       // Number of unique deduplicated objects
  uint64_t total_bytes = 0;         // Total size of all objects in bytes
  uint64_t expired_objects = 0;     // Objects past TTL (not yet swept)
  uint64_t orphaned_objects = 0;    // Objects with refcount=0 (not yet GC'd)
  uint64_t oldest_object_age_s = 0; // Age of oldest object in seconds
  uint64_t newest_access_age_s = 0; // Time since most recent access in seconds
  double dedup_ratio = 0.0;         // Ratio of keys to objects (higher = more dedup)
};

/**
 * prestige::Store
 *
 * A RocksDB-backed KV store with unique-value semantics:
 * - Put(user_key, value) deduplicates identical values (SHA-256 of bytes).
 * - Get(user_key) returns the value bytes transparently.
 *
 * Internally:
 *  user_key -> object_id
 *  object_id -> value bytes
 */
class Store {
 public:
  ~Store();

  Store(const Store&) = delete;
  Store& operator=(const Store&) = delete;

  /**
   * Open or create a prestige store at db_path.
   *
   * This uses RocksDB TransactionDB and creates required column families:
   * - prestige_user_kv
   * - prestige_object_store
   * - prestige_dedup_index
   * - prestige_refcount
   * - prestige_object_meta
   */
  static rocksdb::Status Open(const std::string& db_path,
                              std::unique_ptr<Store>* out,
                              const Options& opt = Options{});

  /** Put user_key -> value_bytes (deduplicating by content hash). */
  rocksdb::Status Put(std::string_view user_key, std::string_view value_bytes);

  /** Get value bytes for user_key. */
  rocksdb::Status Get(std::string_view user_key, std::string* value_bytes_out) const;

  /** Delete user_key mapping and GC objects when unreferenced. */
  rocksdb::Status Delete(std::string_view user_key);

  rocksdb::Status CountKeys(uint64_t* out_key_count) const;
  rocksdb::Status CountUniqueValues(uint64_t* out_unique_value_count) const;

  rocksdb::Status ListKeys(std::vector<std::string>* out_keys,
			   uint64_t limit = 0,
			   std::string_view prefix = {}) const;
  
  /** Close the store and release RocksDB resources. Safe to call multiple times. */
  void Close();

  /** Emit current cache statistics to the metrics sink.
   *  Call periodically (e.g., every few seconds) to get cache observability.
   *  Metrics emitted:
   *   - prestige.cache.hit_total (counter delta since last call)
   *   - prestige.cache.miss_total (counter delta since last call)
   *   - prestige.cache.fill_ratio (gauge, 0.0-1.0)
   *   - prestige.cache.usage_bytes (gauge)
   *   - prestige.cache.capacity_bytes (gauge)
   */
  void EmitCacheMetrics();

  // ---------------------------------------------------------------------------
  // Cache management methods
  // ---------------------------------------------------------------------------

  /**
   * Sweep: Full scan to find and delete expired and orphaned objects.
   * - Deletes objects past TTL (if default_ttl_seconds > 0)
   * - Deletes objects with refcount=0 (orphaned)
   * @param deleted_count Output: number of objects deleted
   * @return OK on success
   */
  rocksdb::Status Sweep(uint64_t* deleted_count);

  /**
   * Prune: Selective deletion by age and/or idle time.
   * @param max_age_seconds Delete objects older than this (0 = no age limit)
   * @param max_idle_seconds Delete objects not accessed for this long (0 = no idle limit)
   * @param deleted_count Output: number of objects deleted
   * @return OK on success
   */
  rocksdb::Status Prune(uint64_t max_age_seconds,
                        uint64_t max_idle_seconds,
                        uint64_t* deleted_count);

  /**
   * EvictLRU: Evict least recently used objects until target size is reached.
   * @param target_bytes Evict until store size is at or below this
   * @param evicted_count Output: number of objects evicted
   * @return OK on success
   */
  rocksdb::Status EvictLRU(uint64_t target_bytes, uint64_t* evicted_count);

  /**
   * GetHealth: Get store health statistics.
   * Performs a scan of all objects to collect stats.
   * @param stats Output: health statistics
   * @return OK on success
   */
  rocksdb::Status GetHealth(HealthStats* stats) const;

  /** Get current approximate total store size in bytes. */
  uint64_t GetTotalStoreBytes() const;

 private:
  explicit Store(const Options& opt);

  rocksdb::Status PutImpl(std::string_view user_key, std::string_view value_bytes);
  rocksdb::Status DeleteImpl(std::string_view user_key);

#ifdef PRESTIGE_ENABLE_SEMANTIC
  rocksdb::Status PutImplSemantic(std::string_view user_key,
                                   std::string_view value_bytes,
                                   TraceSpan* span,
                                   uint64_t op_start_us);
  rocksdb::Status DeleteSemanticObject(rocksdb::Transaction* txn,
                                        const std::string& obj_id);
#endif

  Options opt_;

  rocksdb::TransactionDB* db_ = nullptr;
  std::vector<rocksdb::ColumnFamilyHandle*> handles_;

  // Cache and statistics for observability
  std::shared_ptr<rocksdb::Cache> block_cache_;
  std::shared_ptr<rocksdb::Statistics> statistics_;
  uint64_t last_cache_hits_ = 0;   // For computing deltas
  uint64_t last_cache_misses_ = 0;

  rocksdb::ColumnFamilyHandle* user_kv_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* objects_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* dedup_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* refcount_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* meta_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* lru_cf_ = nullptr;  // LRU index for eviction

  // Cached total store size (updated on Put/Delete)
  mutable std::atomic<uint64_t> total_store_bytes_{0};

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // Semantic dedup members (only used when dedup_mode == kSemantic)
  rocksdb::ColumnFamilyHandle* embeddings_cf_ = nullptr;
  std::unique_ptr<internal::Embedder> embedder_;
  std::unique_ptr<internal::VectorIndex> vector_index_;
  std::string vector_index_path_;  // Path to vector index file
  uint64_t semantic_inserts_since_save_ = 0;  // For periodic index saves
#endif
};

}  // namespace prestige
