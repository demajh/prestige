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

  rocksdb::ReadOptions ro;
  std::string obj_id;
  rocksdb::Status s = db_->Get(ro, user_kv_cf_, rocksdb::Slice(user_key.data(), user_key.size()), &obj_id);
  if (!s.ok()) return s;

  return db_->Get(ro, objects_cf_, rocksdb::Slice(obj_id), value_bytes_out);
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
  // Compute SHA-256 digest as dedup key
  auto digest = prestige::internal::Sha256::Digest(value_bytes);
  std::string digest_key = prestige::internal::ToBytes(digest.data(), digest.size());

  rocksdb::WriteOptions wo;
  rocksdb::ReadOptions ro;

  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  for (int attempt = 0; attempt < opt_.max_retries; ++attempt) {
    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) return rocksdb::Status::IOError("BeginTransaction returned null");

    // Lock user_key mapping (detect overwrite)
    std::string old_obj_id;
    bool had_old = false;
    {
      rocksdb::Status s = txn->GetForUpdate(ro, user_kv_cf_,
                                           rocksdb::Slice(user_key.data(), user_key.size()),
                                           &old_obj_id);
      if (s.ok()) {
        had_old = true;
      } else if (!s.IsNotFound()) {
        if (prestige::internal::IsRetryableTxnStatus(s)) continue;
        return s;
      }
    }

    // Lock digest mapping and resolve object_id
    std::string obj_id;
    {
      rocksdb::Status s = txn->GetForUpdate(ro, dedup_cf_, rocksdb::Slice(digest_key), &obj_id);
      if (s.IsNotFound()) {
        auto new_id = prestige::internal::RandomObjectId128();
        obj_id = prestige::internal::ToBytes(new_id.data(), new_id.size());

        // Create: object bytes, meta, digest->obj_id, refcount initialized to 0
        s = txn->Put(objects_cf_, rocksdb::Slice(obj_id),
                     rocksdb::Slice(value_bytes.data(), value_bytes.size()));
        if (!s.ok()) return s;

        s = txn->Put(meta_cf_, rocksdb::Slice(obj_id), rocksdb::Slice(digest_key));
        if (!s.ok()) return s;

        s = txn->Put(dedup_cf_, rocksdb::Slice(digest_key), rocksdb::Slice(obj_id));
        if (!s.ok()) return s;

        s = txn->Put(refcount_cf_, rocksdb::Slice(obj_id),
                     rocksdb::Slice(prestige::internal::EncodeU64LE(0)));
        if (!s.ok()) return s;
      } else if (!s.ok()) {
        if (prestige::internal::IsRetryableTxnStatus(s)) continue;
        return s;
      }
    }

    // If overwrite maps to same object_id, nothing to do
    if (had_old && old_obj_id == obj_id) {
      txn->Rollback();
      return rocksdb::Status::OK();
    }

    // user_key -> obj_id
    {
      rocksdb::Status s = txn->Put(user_kv_cf_,
                                   rocksdb::Slice(user_key.data(), user_key.size()),
                                   rocksdb::Slice(obj_id));
      if (!s.ok()) return s;
    }

    // Incref(new)
    {
      uint64_t new_cnt = 0;
      rocksdb::Status s = AdjustRefcountLocked(txn.get(), refcount_cf_, obj_id, +1, &new_cnt);
      if (!s.ok()) return s;
    }

    // Decref(old) and GC if needed
    if (had_old && old_obj_id != obj_id) {
      uint64_t old_cnt = 0;
      rocksdb::Status s = AdjustRefcountLocked(txn.get(), refcount_cf_, old_obj_id, -1, &old_cnt);
      if (!s.ok()) return s;

      if (opt_.enable_gc && old_cnt == 0) {
        s = DeleteObjectIfUnreferencedLocked(txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, old_obj_id);
        if (!s.ok()) return s;
      }
    }

    rocksdb::Status cs = txn->Commit();
    if (cs.ok()) return cs;

    if (prestige::internal::IsRetryableTxnStatus(cs)) {
      continue;  // retry
    }
    return cs;
  }

  return rocksdb::Status::TimedOut("Put exceeded max_retries");
}

rocksdb::Status Store::DeleteImpl(std::string_view user_key) {
  rocksdb::WriteOptions wo;
  rocksdb::ReadOptions ro;

  rocksdb::TransactionOptions to;
  to.lock_timeout = opt_.lock_timeout_ms;

  for (int attempt = 0; attempt < opt_.max_retries; ++attempt) {
    std::unique_ptr<rocksdb::Transaction> txn(db_->BeginTransaction(wo, to));
    if (!txn) return rocksdb::Status::IOError("BeginTransaction returned null");

    // Lock and fetch mapping
    std::string obj_id;
    rocksdb::Status s = txn->GetForUpdate(ro, user_kv_cf_,
                                         rocksdb::Slice(user_key.data(), user_key.size()),
                                         &obj_id);
    if (s.IsNotFound()) return s;
    if (!s.ok()) {
      if (prestige::internal::IsRetryableTxnStatus(s)) continue;
      return s;
    }

    // Remove user mapping
    s = txn->Delete(user_kv_cf_, rocksdb::Slice(user_key.data(), user_key.size()));
    if (!s.ok()) return s;

    // Decref and maybe GC
    uint64_t cnt = 0;
    s = AdjustRefcountLocked(txn.get(), refcount_cf_, obj_id, -1, &cnt);
    if (!s.ok()) return s;

    if (opt_.enable_gc && cnt == 0) {
      s = DeleteObjectIfUnreferencedLocked(txn.get(), objects_cf_, dedup_cf_, refcount_cf_, meta_cf_, obj_id);
      if (!s.ok()) return s;
    }

    rocksdb::Status cs = txn->Commit();
    if (cs.ok()) return cs;
    if (prestige::internal::IsRetryableTxnStatus(cs)) continue;
    return cs;
  }

  return rocksdb::Status::TimedOut("Delete exceeded max_retries");
}

}  // namespace prestige
