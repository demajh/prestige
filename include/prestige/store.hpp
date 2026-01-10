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

#include <prestige/normalize.hpp>

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

// Forward declarations for semantic dedup and testing
namespace internal {
class Embedder;
class VectorIndex;
class Reranker;
class JudgeLLM;
struct SearchResult;  // From vector_index.hpp
}  // namespace internal

namespace testing {
class Clock;
}  // namespace testing

/** Deduplication mode: either exact (SHA-256) or semantic (embeddings). */
enum class DedupMode {
  kExact,     // SHA-256 content hash (default, current behavior)
  kSemantic   // Embedding similarity via vector index
};

/** Supported embedding models for semantic dedup. */
enum class SemanticModel {
  kMiniLM,      // all-MiniLM-L6-v2 (384 dimensions)
  kBGESmall,    // BGE-small-en-v1.5 (384 dimensions)
  kBGELarge,    // BGE-large-en-v1.5 (1024 dimensions)
  kE5Large,     // intfloat/e5-large-v2 (1024 dimensions)
  kBGEM3,       // BAAI/bge-m3 (1024 dimensions)
  kNomicEmbed   // nomic-ai/nomic-embed-text-v1.5 (768 dimensions)
};

/** Vector index backend for semantic similarity search. */
enum class SemanticIndexType {
  kHNSW,      // hnswlib - graph-based, excellent recall, recommended
  kFAISS     // FAISS IVF+PQ - for large scale / memory constrained
};

/** Pooling strategy for transformer embeddings. */
enum class SemanticPooling {
  kMean,      // Mean pooling over sequence (default, good for most models)
  kCLS        // Use [CLS] token output (better for some retrieval models)
};

/** Device for ONNX Runtime inference. */
enum class SemanticDevice {
  kCPU,       // CPU inference
  kGPU,       // GPU inference (CUDA)
  kAuto       // Auto-detect: use GPU if available, fall back to CPU (default)
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

  // Retry backoff settings (exponential backoff with jitter).
  // When a transaction conflicts, wait before retrying to avoid thundering herd.
  // Delay formula: min(max_delay, base_delay * 2^attempt) * random(1 ± jitter/2)
  //
  // Example with defaults (1ms base, 100ms max, 50% jitter):
  //   Attempt 1: 0.75-1.25ms
  //   Attempt 2: 1.5-2.5ms
  //   Attempt 3: 3-5ms
  //   Attempt 4: 6-10ms
  //   ...capped at 75-125ms
  uint64_t retry_base_delay_us = 1000;     // 1ms base delay
  uint64_t retry_max_delay_us = 100000;    // 100ms maximum delay
  double retry_jitter_factor = 0.5;        // ±50% randomization

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
  bool track_access_time = true;

  // Minimum interval between LRU timestamp updates for the same object (seconds).
  // When an object is accessed, the LRU index is only updated if at least this
  // much time has passed since the last update. This dramatically reduces write
  // amplification for read-heavy workloads.
  //
  // Trade-off: LRU ordering becomes approximate within this time window.
  // Objects accessed at t=0 and t=3599 (with 1-hour interval) have the same
  // LRU priority until the next access after the interval expires.
  //
  // Recommended values:
  //   0      = Update on every Get (original behavior, maximum write load)
  //   60     = 1 minute (good balance for most workloads)
  //   3600   = 1 hour (minimal writes, coarse LRU granularity)
  //   86400  = 1 day (very coarse, suitable for archival workloads)
  //
  // Default: 3600 (1 hour) - max 24 LRU writes per object per day
  uint64_t lru_update_interval_seconds = 3600;

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

  // ---------------------------------------------------------------------------
  // Text normalization settings (for dedup key computation)
  // ---------------------------------------------------------------------------

  // Normalization mode for computing dedup keys (default: kNone for exact byte match).
  // Note: Normalization affects ONLY the digest key; original values are stored unchanged.
  NormalizationMode normalization_mode = NormalizationMode::kNone;

  // Maximum bytes to normalize (longer values skip normalization and use raw bytes).
  // This prevents excessive memory allocation for huge values.
  // Set to 0 for unlimited.
  size_t normalization_max_bytes = 1024 * 1024;  // 1MB default

  // ---------------------------------------------------------------------------
  // Semantic deduplication settings (only used when dedup_mode == kSemantic)
  // ---------------------------------------------------------------------------

  // Path to ONNX model file (required for semantic mode if custom_embedder is null)
  // The vocabulary file (vocab.txt) is expected to be in the same directory.
  std::string semantic_model_path;

  // Which embedding model architecture the ONNX file contains
  SemanticModel semantic_model_type = SemanticModel::kMiniLM;

  // Custom embedder for testing (optional).
  // If provided, semantic_model_path is ignored and this embedder is used.
  // The Store takes ownership of this pointer.
  // Set to nullptr to use the ONNX model from semantic_model_path.
  internal::Embedder* custom_embedder = nullptr;

  // Custom clock for testing (optional).
  // If provided, the store uses this clock for TTL and LRU timing.
  // This allows deterministic time control in tests.
  // The caller retains ownership - must outlive the Store.
  testing::Clock* custom_clock = nullptr;

  // Cosine similarity threshold for dedup [0.0, 1.0] (REQUIRED for semantic mode)
  // Values >= this threshold are considered duplicates.
  // Set to -1.0 by default to force user to explicitly set it.
  float semantic_threshold = -1.0f;

  // Maximum text bytes to embed (longer texts are truncated)
  size_t semantic_max_text_bytes = 8192;

  // Vector index backend
  SemanticIndexType semantic_index_type = SemanticIndexType::kHNSW;

  // Number of nearest neighbors to retrieve during search
  int semantic_search_k = 50;

  // Number of threads for ONNX embedding inference (0 = use all available cores)
  int semantic_num_threads = 0;

  // Device for ONNX inference (CPU, GPU, or Auto)
  SemanticDevice semantic_device = SemanticDevice::kAuto;

  // Pooling strategy for transformer output
  SemanticPooling semantic_pooling = SemanticPooling::kMean;

  // Verify ANN candidates with exact cosine similarity.
  // When true, loads stored embeddings and computes exact dot product
  // instead of relying on approximate HNSW distances. More accurate
  // but slightly slower. Recommended: true.
  bool semantic_verify_exact = true;

  // Auto-save vector index every N inserts (0 = disabled, save only on Close)
  int semantic_index_save_interval = 1000;

  // HNSW-specific parameters (when semantic_index_type == kHNSW)
  int hnsw_m = 16;                // Max connections per node
  int hnsw_ef_construction = 200; // Build-time search depth
  int hnsw_ef_search = 100;       // Query-time search depth

  // FAISS-specific parameters (when semantic_index_type == kFAISS)
  int faiss_nlist = 100;          // Number of IVF clusters
  int faiss_nprobe = 10;          // Clusters to search at query time
  int faiss_pq_m = 48;            // PQ sub-quantizers (must divide 384)
  int faiss_pq_nbits = 8;         // Bits per sub-quantizer

  // Vector index memory limit (semantic mode only).
  // Maximum number of entries in the vector index before LRU eviction.
  // When exceeded, oldest entries are evicted to make room for new ones.
  // Set to 0 for unlimited (default, but will OOM at scale).
  // Recommended: 1000000 (1M entries) for most deployments.
  size_t semantic_max_index_entries = 0;

  // Compact vector index after this many deletions.
  // Compaction rebuilds the index to reclaim memory from soft-deleted entries.
  // Set to 0 to disable automatic compaction.
  size_t semantic_compact_threshold = 10000;

  // ---------------------------------------------------------------------------
  // Reranker settings (only used when semantic_reranker_enabled = true)
  // ---------------------------------------------------------------------------

  // Enable two-stage retrieval with reranker for higher accuracy
  bool semantic_reranker_enabled = false;

  // Path to cross-encoder reranker model (e.g., BGE-reranker-v2-m3)
  std::string semantic_reranker_model_path;

  // Number of candidates to retrieve for reranking (higher = better recall, slower)
  // This replaces semantic_search_k when reranker is enabled
  int semantic_reranker_top_k = 100;

  // Reranker score threshold [0.0, 1.0]
  // Note: Reranker scores are on a different scale than cosine similarity
  float semantic_reranker_threshold = 0.7f;

  // Batch size for reranking multiple candidates (for efficiency)
  int semantic_reranker_batch_size = 8;

  // Number of threads for reranker ONNX inference (0 = use all cores)
  int semantic_reranker_num_threads = 0;

  // Fall back to embedding-based matching if reranker fails
  bool semantic_reranker_fallback = true;

  // Custom reranker for testing (optional)
  // If provided, semantic_reranker_model_path is ignored
  // The Store takes ownership of this pointer
  internal::Reranker* custom_reranker = nullptr;

  // ---------------------------------------------------------------------------
  // Judge LLM settings (for evaluating borderline matches using Prometheus 2)
  // ---------------------------------------------------------------------------

  // Enable judge LLM for evaluating potential matches in the "gray zone".
  // When enabled, candidates with similarity >= semantic_judge_threshold
  // but < semantic_threshold are evaluated by a judge LLM (Prometheus 2)
  // to make the final dedup decision.
  bool semantic_judge_enabled = false;

  // Path to judge LLM model file (e.g., prometheus-7b-v2.0 GGUF)
  // Supports GGUF format for efficient local inference via llama.cpp
  std::string semantic_judge_model_path;

  // Minimum similarity threshold to trigger judge evaluation [0.0, 1.0]
  // Candidates with: judge_threshold <= similarity < semantic_threshold
  // will be sent to the judge LLM for evaluation.
  // Set this lower than semantic_threshold to define the "gray zone".
  // Example: semantic_threshold=0.90, judge_threshold=0.75 means
  // the judge evaluates candidates with 0.75 <= similarity < 0.90
  float semantic_judge_threshold = 0.75f;

  // Number of threads for judge LLM inference (0 = use all cores)
  int semantic_judge_num_threads = 0;

  // Maximum tokens for judge LLM response
  int semantic_judge_max_tokens = 256;

  // Context size for judge LLM (default: 4096 for 7B models)
  int semantic_judge_context_size = 4096;

  // GPU layers to offload for judge LLM (0 = CPU only, -1 = all layers)
  int semantic_judge_gpu_layers = 0;

  // Custom judge LLM for testing (optional)
  // If provided, semantic_judge_model_path is ignored
  // The Store takes ownership of this pointer
  internal::JudgeLLM* custom_judge = nullptr;

  // ---------------------------------------------------------------------------
  // Reciprocal Nearest Neighbor (RNN) + Margin Gating settings
  // These are fast false-positive reduction techniques that don't need a reranker.
  // ---------------------------------------------------------------------------

  // Enable reciprocal nearest neighbor check.
  // Accept A~B only if B is in A's top-k neighbors AND A is in B's top-k neighbors.
  // This prevents "hub" documents that match many things generically.
  bool semantic_rnn_enabled = false;

  // Number of neighbors to consider for reciprocal check.
  // Higher values are more permissive (fewer FP reductions), lower values stricter.
  // Typically 5-20. Uses semantic_search_k if not set (0).
  int semantic_rnn_k = 0;

  // Enable margin gating for additional FP reduction.
  // Requires: cos(A,B) - cos(A,2nd_best) >= margin AND cos(B,A) - cos(B,2nd_best) >= margin
  // This ensures matches are significantly better than alternatives.
  bool semantic_margin_enabled = false;

  // Margin threshold for margin gating [0.0, 1.0].
  // Higher values = stricter (more FP reduction, potentially lower recall).
  // Typical values: 0.02-0.10
  float semantic_margin_threshold = 0.05f;
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

  /**
   * Get the internal object ID for a key (for debugging/benchmarking).
   * This allows checking if two keys point to the same deduplicated object.
   * Returns OK and sets object_id_out if key exists, NotFound otherwise.
   */
  rocksdb::Status GetObjectId(std::string_view user_key, std::string* object_id_out) const;

  /** Count user keys (exact, requires full scan - O(N)). */
  rocksdb::Status CountKeys(uint64_t* out_key_count) const;

  /** Count unique deduplicated objects (exact, requires full scan - O(N)). */
  rocksdb::Status CountUniqueValues(uint64_t* out_unique_value_count) const;

  /**
   * Approximate key count using RocksDB's internal estimates (O(1)).
   * Fast but may be 10-50% off, especially after many deletes.
   * Use for dashboards, monitoring, or when exact count isn't critical.
   */
  rocksdb::Status CountKeysApprox(uint64_t* out_key_count) const;

  /**
   * Approximate unique object count using RocksDB's internal estimates (O(1)).
   * Fast but may be 10-50% off, especially after many deletes.
   */
  rocksdb::Status CountUniqueValuesApprox(uint64_t* out_unique_value_count) const;

  /**
   * Get approximate total store size using RocksDB's internal estimates (O(1)).
   * Returns estimated live data size across all column families.
   */
  rocksdb::Status GetTotalStoreBytesApprox(uint64_t* out_bytes) const;

  rocksdb::Status ListKeys(std::vector<std::string>* out_keys,
			   uint64_t limit = 0,
			   std::string_view prefix = {}) const;
  
  /**
   * Flush all pending writes to disk with fsync.
   * Ensures durability of all data written so far.
   * Returns OK on success.
   */
  rocksdb::Status Flush();

  /**
   * Close the store and release RocksDB resources.
   * Automatically flushes all pending data before closing.
   * Safe to call multiple times.
   */
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

  // Get current wall clock time in microseconds (uses custom clock if set)
  uint64_t GetWallClockMicros() const;

  rocksdb::Status PutImpl(std::string_view user_key, std::string_view value_bytes);
  rocksdb::Status DeleteImpl(std::string_view user_key);

#ifdef PRESTIGE_ENABLE_SEMANTIC
  rocksdb::Status PutImplSemantic(std::string_view user_key,
                                   std::string_view value_bytes,
                                   TraceSpan* span,
                                   uint64_t op_start_us);
  rocksdb::Status DeleteSemanticObject(rocksdb::Transaction* txn,
                                        const std::string& obj_id);
  // Apply pending vector index operations after successful commit
  void ApplyPendingVectorOps(const std::vector<std::string>& pending_deletes,
                             const std::vector<std::pair<std::string, std::vector<float>>>& pending_adds);
  // Clear pending vector ops from RocksDB after successful vector index save
  void ClearPendingVectorOps();
  // Replay any pending vector ops from previous run (crash recovery)
  void ReplayPendingVectorOps();
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
  rocksdb::ColumnFamilyHandle* vector_pending_cf_ = nullptr;  // WAL for vector index ops
  std::unique_ptr<internal::Embedder> embedder_;
  std::unique_ptr<internal::VectorIndex> vector_index_;
  std::unique_ptr<internal::Reranker> reranker_;  // Optional reranker for two-stage retrieval
  std::unique_ptr<internal::JudgeLLM> judge_llm_;  // Optional judge LLM for gray zone evaluation
  std::string vector_index_path_;  // Path to vector index file
  uint64_t semantic_inserts_since_save_ = 0;  // For periodic index saves

  // Private helper for reranking candidates
  std::string RerankCandidates(std::string_view query_text,
                               const std::vector<internal::SearchResult>& candidates,
                               float* best_score_out) const;

  // Private helper for judge LLM evaluation of gray zone candidates
  bool JudgeCandidate(std::string_view query_text,
                      std::string_view candidate_text,
                      float similarity_score) const;
#endif
};

}  // namespace prestige
