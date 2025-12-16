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
