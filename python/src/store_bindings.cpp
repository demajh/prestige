/**
 * Store class bindings for Prestige Python bindings.
 */

#include "store_bindings.hpp"
#include "exceptions.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <prestige/store.hpp>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace prestige::python {

namespace {

// A digest may arrive as 32 raw bytes or as 64 hex characters.
std::string DigestFromPy(const py::handle& h) {
  if (py::isinstance<py::bytes>(h)) {
    std::string d = h.cast<std::string>();
    if (d.size() != 32) throw py::value_error("digest bytes must be exactly 32 bytes (SHA-256)");
    return d;
  }
  if (py::isinstance<py::str>(h)) {
    std::string hex = h.cast<std::string>();
    if (hex.size() != 64) throw py::value_error("digest hex string must be 64 characters");
    std::string out;
    out.reserve(32);
    for (size_t i = 0; i < 64; i += 2) {
      auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw py::value_error("digest hex string contains a non-hex character");
      };
      out.push_back(static_cast<char>((nib(hex[i]) << 4) | nib(hex[i + 1])));
    }
    return out;
  }
  throw py::type_error("digests must be bytes or hex str");
}

std::vector<std::string> DigestsFromPy(const py::object& obj) {
  std::vector<std::string> out;
  if (obj.is_none()) return out;
  for (const auto& item : obj) out.push_back(DigestFromPy(item));
  return out;
}

py::list DigestsToPy(const std::vector<std::string>& digests) {
  py::list out;
  for (const auto& d : digests) out.append(py::bytes(d));
  return out;
}

py::dict RecordToDict(const DecisionRecord& rec) {
  return py::dict("decision_id"_a = rec.decision.decision_id,
                  "policy_revision"_a = rec.decision.policy_revision,
                  "parent_decision_id"_a = rec.decision.parent_decision_id,
                  "input_digests"_a = DigestsToPy(rec.decision.input_digests),
                  "note"_a = rec.decision.note,
                  "sequence"_a = rec.sequence,
                  "committed_at_us"_a = rec.committed_at_us,
                  "key"_a = rec.user_key,
                  "output_digest"_a = py::bytes(rec.output_digest));
}

}  // namespace

/**
 * Wrapper class for Store that provides Python-friendly API.
 *
 * Uses shared_ptr for Python ownership while wrapping unique_ptr internally.
 */
class PyStore : public std::enable_shared_from_this<PyStore> {
 public:
  /**
   * Factory method to open or create a store.
   */
  static std::shared_ptr<PyStore> Open(const std::string& path,
                                       const Options& options) {
    auto wrapper = std::make_shared<PyStore>();
    rocksdb::Status status;

    {
      py::gil_scoped_release release;  // Release GIL for I/O
      status = Store::Open(path, &wrapper->store_, options);
    }

    CheckStatus(status);
    wrapper->path_ = path;
    wrapper->closed_ = false;
    return wrapper;
  }

  /**
   * Store a key-value pair.
   * Accepts bytes or str for value.
   */
  void Put(const std::string& key, py::object value, py::object decision = py::none()) {
    EnsureOpen();
    std::string value_bytes;

    if (py::isinstance<py::bytes>(value)) {
      value_bytes = value.cast<std::string>();
    } else if (py::isinstance<py::str>(value)) {
      value_bytes = value.cast<std::string>();
    } else {
      throw py::type_error("value must be bytes or str");
    }

    rocksdb::Status status;
    if (decision.is_none()) {
      py::gil_scoped_release release;
      status = store_->Put(key, value_bytes);
    } else {
      const Decision d = decision.cast<Decision>();
      py::gil_scoped_release release;
      status = store_->PutWithDecision(key, value_bytes, d);
    }
    CheckStatus(status);
  }

  /** Read a decision record; the read-time guard runs and the result reports what is missing. */
  py::dict GetDecision(const std::string& decision_id, bool strict = false) {
    EnsureOpen();
    DecisionRecord rec;
    DecisionCheck check;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->GetDecision(decision_id, &rec, &check);
    }
    CheckStatusNotFound(status, decision_id);
    if (strict && !check.resolved) {
      CheckStatus(rocksdb::Status::Corruption("decision " + decision_id + " references " +
                                              std::to_string(check.missing_digests.size()) +
                                              " missing bodies"));
    }
    py::dict out = RecordToDict(rec);
    out["resolved"] = check.resolved;
    out["missing_digests"] = DigestsToPy(check.missing_digests);
    out["queued_for_repair"] = check.queued_for_repair;
    return out;
  }

  py::list ListDecisions(uint64_t limit = 0, uint64_t after_sequence = 0) {
    EnsureOpen();
    std::vector<DecisionRecord> recs;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->ListDecisions(&recs, limit, after_sequence);
    }
    CheckStatus(status);
    py::list out;
    for (const auto& r : recs) out.append(RecordToDict(r));
    return out;
  }

  py::dict SweepDecisions(uint64_t max_records = 1000) {
    EnsureOpen();
    DecisionSweepStats st;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->SweepDecisions(max_records, &st);
    }
    CheckStatus(status);
    return py::dict("checked"_a = st.checked, "dangling"_a = st.dangling, "repaired"_a = st.repaired,
                    "queue_size"_a = st.queue_size, "cursor_sequence"_a = st.cursor_sequence,
                    "max_sequence"_a = st.max_sequence);
  }

  py::bytes Digest(py::object value) {
    EnsureOpen();
    std::string value_bytes;
    if (py::isinstance<py::bytes>(value) || py::isinstance<py::str>(value)) {
      value_bytes = value.cast<std::string>();
    } else {
      throw py::type_error("value must be bytes or str");
    }
    std::string digest;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->Digest(value_bytes, &digest);
    }
    CheckStatus(status);
    return py::bytes(digest);
  }

  /**
   * Get a value by key.
   * Returns bytes by default, or str if decode=True.
   * Raises NotFoundError if key doesn't exist.
   */
  py::object Get(const std::string& key, bool decode = false) {
    EnsureOpen();
    std::string value;
    rocksdb::Status status;

    {
      py::gil_scoped_release release;
      status = store_->Get(key, &value);
    }

    CheckStatusNotFound(status, key);

    if (decode) {
      return py::str(value);
    }
    return py::bytes(value);
  }

  /**
   * Get a value with a default if not found.
   * Returns default_value if key doesn't exist.
   */
  py::object GetDefault(const std::string& key,
                        py::object default_value = py::none(),
                        bool decode = false) {
    EnsureOpen();
    std::string value;
    rocksdb::Status status;

    {
      py::gil_scoped_release release;
      status = store_->Get(key, &value);
    }

    if (status.IsNotFound()) {
      return default_value;
    }
    CheckStatus(status);

    if (decode) {
      return py::str(value);
    }
    return py::bytes(value);
  }

  /**
   * Delete a key.
   */
  void Delete(const std::string& key) {
    EnsureOpen();
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->Delete(key);
    }
    CheckStatus(status);
  }

  /**
   * Count total keys.
   */
  uint64_t CountKeys(bool approximate = false) {
    EnsureOpen();
    uint64_t count = 0;
    rocksdb::Status status;

    {
      py::gil_scoped_release release;
      if (approximate) {
        status = store_->CountKeysApprox(&count);
      } else {
        status = store_->CountKeys(&count);
      }
    }
    CheckStatus(status);
    return count;
  }

  /**
   * Count unique deduplicated values.
   */
  uint64_t CountUniqueValues(bool approximate = false) {
    EnsureOpen();
    uint64_t count = 0;
    rocksdb::Status status;

    {
      py::gil_scoped_release release;
      if (approximate) {
        status = store_->CountUniqueValuesApprox(&count);
      } else {
        status = store_->CountUniqueValues(&count);
      }
    }
    CheckStatus(status);
    return count;
  }

  /**
   * List keys with optional limit and prefix filter.
   */
  std::vector<std::string> ListKeys(uint64_t limit = 0,
                                    const std::string& prefix = "") {
    EnsureOpen();
    std::vector<std::string> keys;
    rocksdb::Status status;

    {
      py::gil_scoped_release release;
      status = store_->ListKeys(&keys, limit, prefix);
    }
    CheckStatus(status);
    return keys;
  }

  /**
   * Sweep expired and orphaned objects.
   * Returns number of objects deleted.
   */
  uint64_t Sweep() {
    EnsureOpen();
    uint64_t deleted = 0;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->Sweep(&deleted);
    }
    CheckStatus(status);
    return deleted;
  }

  /**
   * Prune objects by age or idle time.
   * Returns number of objects deleted.
   */
  uint64_t Prune(uint64_t max_age_seconds = 0, uint64_t max_idle_seconds = 0) {
    EnsureOpen();
    uint64_t deleted = 0;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->Prune(max_age_seconds, max_idle_seconds, &deleted);
    }
    CheckStatus(status);
    return deleted;
  }

  /**
   * Evict LRU objects until target size is reached.
   * Returns number of objects evicted.
   */
  uint64_t EvictLRU(uint64_t target_bytes) {
    EnsureOpen();
    uint64_t evicted = 0;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->EvictLRU(target_bytes, &evicted);
    }
    CheckStatus(status);
    return evicted;
  }

  /**
   * Get health statistics as a dictionary.
   */
  py::dict GetHealth() {
    EnsureOpen();
    HealthStats stats;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->GetHealth(&stats);
    }
    CheckStatus(status);

    return py::dict("total_keys"_a = stats.total_keys,
                    "total_objects"_a = stats.total_objects,
                    "total_bytes"_a = stats.total_bytes,
                    "expired_objects"_a = stats.expired_objects,
                    "orphaned_objects"_a = stats.orphaned_objects,
                    "oldest_object_age_s"_a = stats.oldest_object_age_s,
                    "newest_access_age_s"_a = stats.newest_access_age_s,
                    "dedup_ratio"_a = stats.dedup_ratio,
                    "decisions_total"_a = stats.decisions_total,
                    "decision_queue_size"_a = stats.decision_queue_size,
                    "decision_dangling_reads"_a = stats.decision_dangling_reads,
                    "decision_repairs"_a = stats.decision_repairs);
  }

  /**
   * Get total store size in bytes.
   */
  uint64_t TotalBytes() {
    EnsureOpen();
    return store_->GetTotalStoreBytes();
  }

  /**
   * Flush pending writes to disk.
   */
  void Flush() {
    EnsureOpen();
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->Flush();
    }
    CheckStatus(status);
  }

  /**
   * Get the internal object ID for a key.
   * Returns the object ID as bytes if the key exists.
   * Raises NotFoundError if the key doesn't exist.
   * Useful for benchmarking to check if two keys deduplicated to the same object.
   */
  py::bytes GetObjectId(const std::string& key) {
    EnsureOpen();
    std::string object_id;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->GetObjectId(key, &object_id);
    }
    CheckStatusNotFound(status, key);
    // Return as bytes to avoid UTF-8 decoding errors (object IDs are binary hashes)
    return py::bytes(object_id);
  }

  /**
   * Close the store.
   */
  void Close() {
    if (!closed_ && store_) {
      py::gil_scoped_release release;
      store_->Close();
    }
    closed_ = true;
  }

  bool IsClosed() const { return closed_; }
  std::string Path() const { return path_; }

  // Context manager support
  PyStore* Enter() {
    EnsureOpen();
    return this;
  }

  void Exit(py::object /*exc_type*/, py::object /*exc_val*/,
            py::object /*exc_tb*/) {
    Close();
  }

  // Dict-like interface
  bool Contains(const std::string& key) {
    EnsureOpen();
    std::string value;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->Get(key, &value);
    }
    return status.ok();
  }

 private:
  void EnsureOpen() const {
    if (closed_ || !store_) {
      throw py::value_error("Store is closed");
    }
  }

  std::unique_ptr<Store> store_;
  std::string path_;
  bool closed_ = true;
};

void BindStore(py::module_& m) {
  py::class_<Decision>(m, "Decision", R"doc(A decision record to commit together with a write.

        Ties the write to the reason it happened: a caller-chosen decision id, the policy revision in
        force, optionally the parent decision and the digests of the values the decision consumed.
        The record is stored in the same transaction as the value (see docs/provenance.md).

        Args:
            decision_id: Unique id for this decision (required)
            policy_revision: Identifier or hash of the policy in force (required)
            parent_decision_id: The decision this one follows (optional)
            input_digests: Digests of consumed values, as 32-byte bytes or 64-char hex (optional)
            note: Short free text (optional)
        )doc")
      .def(py::init([](std::string decision_id, std::string policy_revision, std::string parent_decision_id,
                       py::object input_digests, std::string note) {
             Decision d;
             d.decision_id = std::move(decision_id);
             d.policy_revision = std::move(policy_revision);
             d.parent_decision_id = std::move(parent_decision_id);
             d.input_digests = DigestsFromPy(input_digests);
             d.note = std::move(note);
             return d;
           }),
           py::arg("decision_id"), py::arg("policy_revision"), py::arg("parent_decision_id") = "",
           py::arg("input_digests") = py::none(), py::arg("note") = "")
      .def_readwrite("decision_id", &Decision::decision_id)
      .def_readwrite("policy_revision", &Decision::policy_revision)
      .def_readwrite("parent_decision_id", &Decision::parent_decision_id)
      .def_readwrite("note", &Decision::note)
      .def_property(
          "input_digests", [](const Decision& d) { return DigestsToPy(d.input_digests); },
          [](Decision& d, py::object v) { d.input_digests = DigestsFromPy(v); })
      .def("__repr__", [](const Decision& d) {
        return "Decision(decision_id=" + py::repr(py::str(d.decision_id)).cast<std::string>() +
               ", policy_revision=" + py::repr(py::str(d.policy_revision)).cast<std::string>() +
               ", inputs=" + std::to_string(d.input_digests.size()) + ")";
      });

  py::class_<PyStore, std::shared_ptr<PyStore>>(
      m, "Store",
      R"doc(Content-deduplicated key-value store.

    Can be used as a context manager or with explicit open/close.

    Example (context manager):
        with prestige.Store.open("/path/to/db") as store:
            store.put("key", "value")
            value = store.get("key")

    Example (explicit):
        store = prestige.Store.open("/path/to/db")
        try:
            store.put("key", "value")
        finally:
            store.close()
    )doc")

      // Factory method
      .def_static("open", &PyStore::Open, py::arg("path"),
                  py::arg("options") = Options{},
                  R"doc(Open or create a store at the given path.

        Args:
            path: Path to the database directory
            options: Store configuration options

        Returns:
            Store instance

        Raises:
            IOError: If the database cannot be opened
        )doc")

      // Core KV operations
      .def("put", &PyStore::Put, py::arg("key"), py::arg("value"), py::arg("decision") = py::none(),
           R"doc(Store a key-value pair.

        The value is automatically deduplicated by content hash. Pass a Decision to commit a
        provenance record in the same transaction as the write (exact mode only).

        Args:
            key: Key string
            value: Value as bytes or str
            decision: Optional Decision recorded atomically with the write
        )doc")

      .def("get_decision", &PyStore::GetDecision, py::arg("decision_id"), py::arg("strict") = false,
           R"doc(Read a decision record and resolve its commitments.

        This is the read-time guard: when a referenced body is missing, the record is queued for a
        priority sweep and the store's dangling-read counter is incremented.

        Args:
            decision_id: The decision to read
            strict: If True, raise CorruptionError instead of returning resolved=False

        Returns:
            Dict with decision_id, policy_revision, parent_decision_id, input_digests, note,
            sequence, committed_at_us, key, output_digest, resolved, missing_digests,
            queued_for_repair

        Raises:
            NotFoundError: If the decision id is unknown
            CorruptionError: If strict and a referenced body is missing
        )doc")

      .def("list_decisions", &PyStore::ListDecisions, py::arg("limit") = 0, py::arg("after_sequence") = 0,
           "Decision records in write order, optionally limited and starting after a sequence number.")

      .def("sweep_decisions", &PyStore::SweepDecisions, py::arg("max_records") = 1000,
           R"doc(Drain the repair queue, then continue the write-order walk of decision records.

        Returns:
            Dict with checked, dangling, repaired, queue_size (the integrity debt),
            cursor_sequence, max_sequence
        )doc")

      .def("digest", &PyStore::Digest, py::arg("value"),
           "The 32-byte content key this store computes for a value (normalization-aware), for Decision.input_digests.")

      .def("get", &PyStore::Get, py::arg("key"), py::arg("decode") = false,
           R"doc(Get value for key.

        Args:
            key: Key string
            decode: If True, return str instead of bytes

        Returns:
            Value as bytes (or str if decode=True)

        Raises:
            NotFoundError: If key doesn't exist
        )doc")

      .def("get", &PyStore::GetDefault, py::arg("key"),
           py::arg("default") = py::none(), py::arg("decode") = false,
           R"doc(Get value for key, or return default if not found.

        Args:
            key: Key string
            default: Value to return if key not found
            decode: If True, return str instead of bytes

        Returns:
            Value as bytes (or str if decode=True), or default
        )doc")

      .def("delete", &PyStore::Delete, py::arg("key"),
           "Delete a key from the store.")

      // Counting and listing
      .def("count_keys", &PyStore::CountKeys, py::arg("approximate") = false,
           R"doc(Count total keys in the store.

        Args:
            approximate: If True, use fast O(1) estimate

        Returns:
            Number of keys
        )doc")

      .def("count_unique_values", &PyStore::CountUniqueValues,
           py::arg("approximate") = false,
           R"doc(Count unique deduplicated values.

        Args:
            approximate: If True, use fast O(1) estimate

        Returns:
            Number of unique values
        )doc")

      .def("list_keys", &PyStore::ListKeys, py::arg("limit") = 0,
           py::arg("prefix") = "",
           R"doc(List keys with optional limit and prefix filter.

        Args:
            limit: Maximum number of keys (0 = unlimited)
            prefix: Only return keys starting with this prefix

        Returns:
            List of key strings
        )doc")

      // Cache management
      .def("sweep", &PyStore::Sweep,
           R"doc(Delete expired and orphaned objects.

        Returns:
            Number of objects deleted
        )doc")

      .def("prune", &PyStore::Prune, py::arg("max_age_seconds") = 0,
           py::arg("max_idle_seconds") = 0,
           R"doc(Delete objects by age or idle time.

        Args:
            max_age_seconds: Delete objects older than this (0 = ignore)
            max_idle_seconds: Delete objects not accessed for this long (0 = ignore)

        Returns:
            Number of objects deleted
        )doc")

      .def("evict_lru", &PyStore::EvictLRU, py::arg("target_bytes"),
           R"doc(Evict LRU objects until target size is reached.

        Args:
            target_bytes: Target store size in bytes

        Returns:
            Number of objects evicted
        )doc")

      .def("get_health", &PyStore::GetHealth,
           R"doc(Get store health statistics.

        Returns:
            Dict with keys: total_keys, total_objects, total_bytes,
            expired_objects, orphaned_objects, oldest_object_age_s,
            newest_access_age_s, dedup_ratio
        )doc")

      // Properties
      .def_property_readonly("total_bytes", &PyStore::TotalBytes,
                             "Total store size in bytes.")
      .def_property_readonly("path", &PyStore::Path, "Database path.")
      .def_property_readonly("closed", &PyStore::IsClosed,
                             "True if store is closed.")

      // Lifecycle
      .def("flush", &PyStore::Flush, "Flush pending writes to disk.")
      .def("get_object_id", &PyStore::GetObjectId, py::arg("key"),
           R"doc(Get internal object ID for a key (returns bytes).

           Used for benchmarking to check if two keys deduplicated to the same object.
           Returns bytes to avoid UTF-8 decoding errors on binary hash values.

           Args:
               key: Key string

           Returns:
               Object ID as bytes

           Raises:
               NotFoundError: If key doesn't exist
           )doc")
      .def("close", &PyStore::Close, "Close the store.")

      // Context manager
      .def("__enter__", &PyStore::Enter, py::return_value_policy::reference)
      .def("__exit__", &PyStore::Exit)

      // Dict-like interface
      .def("__contains__", &PyStore::Contains)
      .def("__setitem__", [](PyStore& self, const std::string& key, py::object value) { self.Put(key, std::move(value), py::none()); },
           "Store a key-value pair (dict-style).")
      .def(
          "__getitem__",
          [](PyStore& self, const std::string& key) {
            return self.Get(key, false);
          })
      .def("__delitem__", &PyStore::Delete)
      .def("__len__",
           [](PyStore& self) {
             return self.CountKeys(true);  // Use approximate for len()
           })

      // Repr
      .def("__repr__", [](const PyStore& self) {
        return "<prestige.Store path='" + self.Path() +
               "' closed=" + (self.IsClosed() ? "True" : "False") + ">";
      });
}

}  // namespace prestige::python
