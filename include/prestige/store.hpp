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

  Options opt_;

  rocksdb::TransactionDB* db_ = nullptr;
  std::vector<rocksdb::ColumnFamilyHandle*> handles_;

  rocksdb::ColumnFamilyHandle* user_kv_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* objects_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* dedup_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* refcount_cf_ = nullptr;
  rocksdb::ColumnFamilyHandle* meta_cf_ = nullptr;
};

}  // namespace prestige
