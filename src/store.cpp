#include <prestige/store.hpp>

#include <rocksdb/cache.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/table.h>
#include <rocksdb/utilities/transaction.h>

#include <cstring>

#include <prestige/internal.hpp>

namespace prestige {

namespace {

constexpr const char* kUserKvCF      = "prestige_user_kv";
constexpr const char* kObjectStoreCF = "prestige_object_store";
constexpr const char* kDedupIndexCF  = "prestige_dedup_index";
constexpr const char* kRefcountCF    = "prestige_refcount";
constexpr const char* kObjectMetaCF  = "prestige_object_meta";

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

Store::Store(const Options& opt) : opt_(opt) {}

Store::~Store() { Close(); }

rocksdb::Status Store::Open(const std::string& db_path,
                            std::unique_ptr<Store>* out,
                            const Options& opt) {
  if (!out) return rocksdb::Status::InvalidArgument("out is null");

  auto store = std::unique_ptr<Store>(new Store(opt));

  // RocksDB options
  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  rocksdb::TransactionDBOptions txn_opts;

  // Shared cache for all CFs
  auto cache = rocksdb::NewLRUCache(opt.block_cache_bytes);

  std::vector<rocksdb::ColumnFamilyDescriptor> cfs;
  cfs.emplace_back(rocksdb::kDefaultColumnFamilyName, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kUserKvCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kObjectStoreCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kDedupIndexCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kRefcountCF, MakeCFOptions(cache, opt.bloom_bits_per_key));
  cfs.emplace_back(kObjectMetaCF, MakeCFOptions(cache, opt.bloom_bits_per_key));

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

  // 2) object_id -> value_bytes
  const uint64_t obj_start_us = prestige::internal::NowMicros();
  rocksdb::Status s2 = db_->Get(ro, objects_cf_, rocksdb::Slice(obj_id), value_bytes_out);
  EmitHistogram(opt_, "prestige.get.object_lookup_us",
                prestige::internal::NowMicros() - obj_start_us);

  if (s2.ok()) {
    EmitHistogram(opt_, "prestige.get.value_bytes",
                  static_cast<uint64_t>(value_bytes_out->size()));
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

void Store::Close() {
  if (!db_) return;
  for (auto* h : handles_) delete h;
  handles_.clear();
  delete db_;
  db_ = nullptr;
  user_kv_cf_ = objects_cf_ = dedup_cf_ = refcount_cf_ = meta_cf_ = nullptr;
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
                                                        const std::string& obj_id) {
  if (!txn) return rocksdb::Status::InvalidArgument("txn is null");

  rocksdb::ReadOptions ro;

  // Lookup digest via meta (locks meta)
  std::string digest_key;
  rocksdb::Status s = txn->GetForUpdate(ro, meta_cf, rocksdb::Slice(obj_id), &digest_key);
  if (s.IsNotFound()) {
    // Best-effort cleanup of object/refcount, but cannot reliably cleanup dedup index.
    (void)txn->Delete(objects_cf, rocksdb::Slice(obj_id));
    (void)txn->Delete(refcount_cf, rocksdb::Slice(obj_id));
    return rocksdb::Status::Corruption("object_meta missing; best-effort cleanup done");
  }
  if (!s.ok()) return s;

  // Only delete dedup mapping if it still points to this obj_id
  std::string mapped_id;
  s = txn->GetForUpdate(ro, dedup_cf, rocksdb::Slice(digest_key), &mapped_id);
  if (s.ok()) {
    if (mapped_id == obj_id) {
      s = txn->Delete(dedup_cf, rocksdb::Slice(digest_key));
      if (!s.ok()) return s;
    }
  } else if (!s.IsNotFound()) {
    return s;
  }

  // Remove object bytes + meta + refcount
  s = txn->Delete(objects_cf, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

  s = txn->Delete(meta_cf, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

  s = txn->Delete(refcount_cf, rocksdb::Slice(obj_id));
  if (!s.ok()) return s;

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

  // Compute SHA-256 digest as dedup key
  const uint64_t sha_start_us = prestige::internal::NowMicros();
  auto digest = prestige::internal::Sha256::Digest(value_bytes);
  EmitHistogram(opt_, "prestige.put.sha256_us", prestige::internal::NowMicros() - sha_start_us);
  std::string digest_key = prestige::internal::ToBytes(digest.data(), digest.size());

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
          SpanEvent(span.get(), "retry.user_key_lock");
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

        // Create: object bytes, meta, digest->obj_id, refcount initialized to 0
        s = txn->Put(objects_cf_, rocksdb::Slice(obj_id),
                     rocksdb::Slice(value_bytes.data(), value_bytes.size()));
        if (!s.ok()) return finish(s);

        s = txn->Put(meta_cf_, rocksdb::Slice(obj_id), rocksdb::Slice(digest_key));
        if (!s.ok()) return finish(s);

        s = txn->Put(dedup_cf_, rocksdb::Slice(digest_key), rocksdb::Slice(obj_id));
        if (!s.ok()) return finish(s);

        s = txn->Put(refcount_cf_, rocksdb::Slice(obj_id),
                     rocksdb::Slice(prestige::internal::EncodeU64LE(0)));
        if (!s.ok()) return finish(s);

      } else if (!s.ok()) {
        if (prestige::internal::IsRetryableTxnStatus(s)) {
          EmitCounter(opt_, "prestige.put.retry_total", 1);
          SpanEvent(span.get(), "retry.dedup_lock");
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
        s = DeleteObjectIfUnreferencedLocked(
            txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, old_obj_id);
        if (!s.ok()) return finish(s);

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

  auto finish = [&](const rocksdb::Status& st) -> rocksdb::Status {
    const uint64_t dur_us = prestige::internal::NowMicros() - op_start_us;
    EmitHistogram(opt_, "prestige.delete.latency_us", dur_us);
    EmitHistogram(opt_, "prestige.delete.attempts", static_cast<uint64_t>(attempts_used));

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

  for (int attempt = 0; attempt < opt_.max_retries; ++attempt) {
    attempts_used = attempt + 1;

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
        continue;
      }
      return finish(s);
    }

    // Remove user mapping
    s = txn->Delete(user_kv_cf_, rocksdb::Slice(user_key.data(), user_key.size()));
    if (!s.ok()) return finish(s);

    // Decref and maybe GC
    uint64_t cnt = 0;
    s = AdjustRefcountLocked(txn.get(), refcount_cf_, obj_id, -1, &cnt);
    if (!s.ok()) return finish(s);

    if (opt_.enable_gc && cnt == 0) {
      s = DeleteObjectIfUnreferencedLocked(
          txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, obj_id);
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
      continue;
    }

    return finish(cs);
  }

  return finish(rocksdb::Status::TimedOut("Delete exceeded max_retries"));
}

}  // namespace prestige
