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

Metadata MetadataFromPy(const py::object& obj) {
  Metadata m;
  if (obj.is_none()) return m;
  for (const auto& item : obj.cast<py::dict>()) {
    m[item.first.cast<std::string>()] = item.second.cast<std::string>();
  }
  return m;
}

py::dict MetadataToPy(const Metadata& m) {
  py::dict d;
  for (const auto& [k, v] : m) d[py::str(k)] = py::str(v);
  return d;
}

py::dict CandidateToDict(const Candidate& c) {
  py::dict d("rank"_a = c.rank, "similarity"_a = c.similarity, "object_id"_a = py::bytes(c.object_id),
             "digest"_a = py::bytes(c.digest), "metadata"_a = MetadataToPy(c.metadata), "size_bytes"_a = c.size_bytes,
             "created_at_us"_a = c.created_at_us);
  d["reranker_score"] = c.reranker_score < 0.0f ? py::object(py::none()) : py::object(py::float_(c.reranker_score));
  if (!c.value.empty() || c.size_bytes == 0) d["value"] = py::bytes(c.value);
  return d;
}

OutcomeVerdict VerdictFromString(const std::string& v) {
  if (v == "accepted") return OutcomeVerdict::kAccepted;
  if (v == "rejected") return OutcomeVerdict::kRejected;
  if (v == "no_candidate") return OutcomeVerdict::kNoCandidate;
  throw py::value_error("verdict must be 'accepted', 'rejected' or 'no_candidate'");
}

const char* VerdictToString(OutcomeVerdict v) {
  switch (v) {
    case OutcomeVerdict::kAccepted: return "accepted";
    case OutcomeVerdict::kRejected: return "rejected";
    case OutcomeVerdict::kNoCandidate: return "no_candidate";
  }
  return "unknown";
}

py::object OptionalScore(float f) { return f < 0.0f ? py::object(py::none()) : py::object(py::float_(f)); }

py::dict OutcomeRecordToDict(const OutcomeRecord& r) {
  return py::dict("sequence"_a = r.sequence, "recorded_at_us"_a = r.recorded_at_us, "family_id"_a = r.outcome.family_id,
                  "verdict"_a = VerdictToString(r.outcome.verdict), "tool_version"_a = r.outcome.tool_version,
                  "schema_version"_a = r.outcome.schema_version,
                  "candidate_object_id"_a = py::bytes(r.outcome.candidate_object_id),
                  "candidate_digest"_a = py::bytes(r.outcome.candidate_digest), "rank"_a = r.outcome.rank,
                  "similarity"_a = OptionalScore(r.outcome.similarity),
                  "reranker_score"_a = OptionalScore(r.outcome.reranker_score),
                  "threshold"_a = OptionalScore(r.outcome.threshold), "reason"_a = r.outcome.reason);
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
  void Put(const std::string& key, py::object value, py::object decision = py::none(),
           py::object metadata = py::none()) {
    EnsureOpen();
    std::string value_bytes;

    if (py::isinstance<py::bytes>(value)) {
      value_bytes = value.cast<std::string>();
    } else if (py::isinstance<py::str>(value)) {
      value_bytes = value.cast<std::string>();
    } else {
      throw py::type_error("value must be bytes or str");
    }

    const Metadata md = MetadataFromPy(metadata);
    const bool has_md = !metadata.is_none();
    rocksdb::Status status;
    if (decision.is_none()) {
      py::gil_scoped_release release;
      status = has_md ? store_->Put(key, value_bytes, md) : store_->Put(key, value_bytes);
    } else {
      const Decision d = decision.cast<Decision>();
      py::gil_scoped_release release;
      status = has_md ? store_->PutWithDecision(key, value_bytes, d, md) : store_->PutWithDecision(key, value_bytes, d);
    }
    CheckStatus(status);
  }

  py::dict GetMetadata(const std::string& key) {
    EnsureOpen();
    Metadata m;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->GetMetadata(key, &m);
    }
    CheckStatusNotFound(status, key);
    return MetadataToPy(m);
  }

  py::list Candidates(py::object value, size_t k, py::object filter, py::object min_similarity, bool include_values) {
    EnsureOpen();
    std::string value_bytes;
    if (py::isinstance<py::bytes>(value) || py::isinstance<py::str>(value)) {
      value_bytes = value.cast<std::string>();
    } else {
      throw py::type_error("value must be bytes or str");
    }
    CandidateQuery q;
    q.k = k;
    q.filter = MetadataFromPy(filter);
    q.min_similarity = min_similarity.is_none() ? -1.0f : min_similarity.cast<float>();
    q.include_values = include_values;
    std::vector<Candidate> out;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->Candidates(value_bytes, q, &out);
    }
    CheckStatus(status);
    py::list result;
    for (const auto& c : out) result.append(CandidateToDict(c));
    return result;
  }

  uint64_t RecordOutcome(const std::string& family_id, const std::string& verdict, py::object candidate, py::object rank,
                         py::object similarity, py::object reranker_score, py::object threshold,
                         const std::string& tool_version, const std::string& schema_version, const std::string& reason) {
    EnsureOpen();
    Outcome o;
    o.family_id = family_id;
    o.verdict = VerdictFromString(verdict);
    o.tool_version = tool_version;
    o.schema_version = schema_version;
    o.reason = reason;
    if (!candidate.is_none()) {
      // A dict from candidates() carries the object id, digest, rank and scores.
      py::dict c = candidate.cast<py::dict>();
      if (c.contains("object_id")) o.candidate_object_id = c["object_id"].cast<std::string>();
      if (c.contains("digest")) o.candidate_digest = c["digest"].cast<std::string>();
      if (c.contains("rank")) o.rank = c["rank"].cast<uint32_t>();
      if (c.contains("similarity")) o.similarity = c["similarity"].cast<float>();
      if (c.contains("reranker_score") && !c["reranker_score"].is_none()) o.reranker_score = c["reranker_score"].cast<float>();
    }
    if (!rank.is_none()) o.rank = rank.cast<uint32_t>();
    if (!similarity.is_none()) o.similarity = similarity.cast<float>();
    if (!reranker_score.is_none()) o.reranker_score = reranker_score.cast<float>();
    if (!threshold.is_none()) o.threshold = threshold.cast<float>();
    uint64_t seq = 0;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->RecordOutcome(o, &seq);
    }
    CheckStatus(status);
    return seq;
  }

  py::list ListOutcomes(const std::string& family_id, uint64_t limit, uint64_t after_sequence) {
    EnsureOpen();
    std::vector<OutcomeRecord> recs;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->ListOutcomes(family_id, &recs, limit, after_sequence);
    }
    CheckStatus(status);
    py::list out;
    for (const auto& r : recs) out.append(OutcomeRecordToDict(r));
    return out;
  }

  py::dict FamilyReport(const std::string& family_id) {
    EnsureOpen();
    prestige::FamilyReport r;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->GetFamilyReport(family_id, &r);
    }
    CheckStatusNotFound(status, family_id);
    return py::dict("family_id"_a = r.family_id, "accepted"_a = r.accepted, "rejected"_a = r.rejected,
                    "no_candidate"_a = r.no_candidate, "false_accept_rate"_a = r.false_accept_rate,
                    "accepted_similarity_hist"_a = r.accepted_similarity_hist,
                    "rejected_similarity_hist"_a = r.rejected_similarity_hist,
                    "rejected_rank_hist"_a = r.rejected_rank_hist,
                    "suggested_threshold"_a = OptionalScore(r.suggested_threshold),
                    "first_sequence"_a = r.first_sequence, "last_sequence"_a = r.last_sequence,
                    "last_recorded_at_us"_a = r.last_recorded_at_us);
  }

  std::vector<std::string> ListFamilies() {
    EnsureOpen();
    std::vector<std::string> out;
    rocksdb::Status status;
    {
      py::gil_scoped_release release;
      status = store_->ListFamilies(&out);
    }
    CheckStatus(status);
    return out;
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
           py::arg("metadata") = py::none(),
           R"doc(Store a key-value pair.

        The value is automatically deduplicated by content hash. Pass a Decision to commit a
        provenance record in the same transaction as the write (exact mode only). Pass metadata
        (a dict of str to str, e.g. task family, tool version, constraint-schema version) to attach
        it to the stored value; it is returned with every candidate and merged into the value's
        existing metadata when the same bytes are written again.

        Args:
            key: Key string
            value: Value as bytes or str
            decision: Optional Decision recorded atomically with the write
            metadata: Optional dict of str to str attached to the value
        )doc")

      .def("get_metadata", &PyStore::GetMetadata, py::arg("key"),
           "Metadata stored with the value behind key (empty dict when none). Raises NotFoundError for unknown keys.")

      .def("candidates", &PyStore::Candidates, py::arg("value"), py::arg("k") = 10, py::arg("filter") = py::none(),
           py::arg("min_similarity") = py::none(), py::arg("include_values") = false,
           R"doc(The nearest stored values to `value`, ranked by similarity, with their metadata.

        The store does not decide whether any candidate is a hit; apply your own thresholds and
        constraint checks to the list. Exact mode returns the identical value at similarity 1.0 or
        an empty list; semantic mode searches the vector index.

        Args:
            value: Query value as bytes or str
            k: Maximum number of candidates
            filter: Dict of metadata pairs every candidate must carry
            min_similarity: Drop candidates below this cosine similarity
            include_values: Also return the stored bytes under "value"

        Returns:
            List of dicts with rank, similarity, reranker_score (None without a reranker),
            object_id, digest, metadata, size_bytes, created_at_us and optionally value
        )doc")

      .def("record_outcome", &PyStore::RecordOutcome, py::arg("family_id"), py::arg("verdict"),
           py::arg("candidate") = py::none(), py::arg("rank") = py::none(), py::arg("similarity") = py::none(),
           py::arg("reranker_score") = py::none(), py::arg("threshold") = py::none(), py::arg("tool_version") = "",
           py::arg("schema_version") = "", py::arg("reason") = "",
           R"doc(Record what you decided about a candidate, keyed by its task family.

        Args:
            family_id: Explicit, versioned task-family id, e.g. "support-faq@v3"
            verdict: "accepted" (reused), "rejected" (above your threshold but failed a
                constraint check: a false accept) or "no_candidate"
            candidate: A dict from candidates(); supplies object_id, digest, rank and scores
            rank, similarity, reranker_score, threshold: Override or supply the numbers directly
            tool_version, schema_version, reason: Free-form context

        Returns:
            The outcome's sequence number
        )doc")

      .def("list_outcomes", &PyStore::ListOutcomes, py::arg("family_id"), py::arg("limit") = 0,
           py::arg("after_sequence") = 0, "Outcomes of one family in the order recorded.")

      .def("family_report", &PyStore::FamilyReport, py::arg("family_id"),
           R"doc(False-accept rate and distributions for one family.

        Returns:
            Dict with accepted, rejected, no_candidate, false_accept_rate, accepted_similarity_hist
            and rejected_similarity_hist (20 buckets over [0, 1]), rejected_rank_hist (ranks 0..15+),
            suggested_threshold (advisory, None without enough evidence), first_sequence,
            last_sequence, last_recorded_at_us

        Raises:
            NotFoundError: If the family has no outcomes
        )doc")

      .def("list_families", &PyStore::ListFamilies, "Every family with at least one recorded outcome.")

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
      .def("__setitem__", [](PyStore& self, const std::string& key, py::object value) { self.Put(key, std::move(value), py::none(), py::none()); },
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
