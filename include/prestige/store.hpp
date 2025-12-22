#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <rocksdb/options.h>
#include <rocksdb/status.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction_db.h>

namespace prestige {

/** A minimal metrics sink interface (counters + histograms). */
struct MetricsSink {
  virtual ~MetricsSink() = default;

  /** Monotonic counters (e.g., calls, retries, dedup hits). */
  virtual void Counter(std::string_view name, uint64_t delta) = 0;

  /** Histograms (e.g., latency in microseconds, sizes in bytes). */
  virtual void Histogram(std::string_view name, uint64_t value) = 0;
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

  rocksdb::ColumnFamilyHandle* user_kv_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* objects_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* dedup_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* refcount_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* meta_cf_ = nullptr;

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
