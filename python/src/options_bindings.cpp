/**
 * Options and enum bindings for Prestige Python bindings.
 */

#include "options_bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <prestige/normalize.hpp>
#include <prestige/store.hpp>

namespace py = pybind11;

namespace prestige::python {

void BindOptions(py::module_& m) {
  // DedupMode enum
  py::enum_<DedupMode>(m, "DedupMode", "Deduplication mode")
      .value("EXACT", DedupMode::kExact, "SHA-256 content hash (default)")
      .value("SEMANTIC", DedupMode::kSemantic, "Embedding similarity");

  // NormalizationMode enum
  py::enum_<NormalizationMode>(m, "NormalizationMode",
                               "Text normalization for dedup")
      .value("NONE", NormalizationMode::kNone, "Exact byte match (default)")
      .value("WHITESPACE", NormalizationMode::kWhitespace, "Collapse whitespace")
      .value("ASCII", NormalizationMode::kASCII,
             "Whitespace + ASCII case folding")
      .value("UNICODE", NormalizationMode::kUnicode,
             "Whitespace + Unicode normalization");

#ifdef PRESTIGE_ENABLE_SEMANTIC
  // SemanticModel enum
  py::enum_<SemanticModel>(m, "SemanticModel", "Embedding model type")
      .value("MINILM", SemanticModel::kMiniLM,
             "all-MiniLM-L6-v2 (384 dimensions)")
      .value("BGE_SMALL", SemanticModel::kBGESmall,
             "BGE-small-en-v1.5 (384 dimensions)")
      .value("BGE_LARGE", SemanticModel::kBGELarge,
             "BGE-large-en-v1.5 (1024 dimensions)");

  // SemanticIndexType enum
  py::enum_<SemanticIndexType>(m, "SemanticIndexType", "Vector index backend")
      .value("HNSW", SemanticIndexType::kHNSW,
             "hnswlib graph-based (recommended)")
      .value("FAISS", SemanticIndexType::kFAISS,
             "FAISS IVF+PQ (large scale)");
#endif

  // Options class
  py::class_<Options>(m, "Options", "Store configuration options")
      .def(py::init<>())

      // RocksDB performance
      .def_readwrite("block_cache_bytes", &Options::block_cache_bytes,
                     "Block cache size in bytes (default: 256MB)")
      .def_readwrite("bloom_bits_per_key", &Options::bloom_bits_per_key,
                     "Bloom filter bits per key (default: 10)")

      // Transaction behavior
      .def_readwrite("lock_timeout_ms", &Options::lock_timeout_ms,
                     "Transaction lock timeout (default: 2000ms)")
      .def_readwrite("max_retries", &Options::max_retries,
                     "Maximum transaction retries (default: 16)")
      .def_readwrite("retry_base_delay_us", &Options::retry_base_delay_us,
                     "Retry base delay in microseconds (default: 1000)")
      .def_readwrite("retry_max_delay_us", &Options::retry_max_delay_us,
                     "Maximum retry delay (default: 100000)")
      .def_readwrite("retry_jitter_factor", &Options::retry_jitter_factor,
                     "Retry jitter factor 0-1 (default: 0.5)")

      // GC and cache
      .def_readwrite("enable_gc", &Options::enable_gc,
                     "Enable garbage collection (default: True)")
      .def_readwrite("default_ttl_seconds", &Options::default_ttl_seconds,
                     "Default TTL in seconds, 0=no expiration (default: 0)")
      .def_readwrite("max_store_bytes", &Options::max_store_bytes,
                     "Maximum store size, 0=unlimited (default: 0)")
      .def_readwrite("eviction_target_ratio", &Options::eviction_target_ratio,
                     "Target fill ratio for eviction (default: 0.8)")
      .def_readwrite("track_access_time", &Options::track_access_time,
                     "Enable LRU tracking (default: True)")
      .def_readwrite("lru_update_interval_seconds",
                     &Options::lru_update_interval_seconds,
                     "Minimum seconds between LRU updates (default: 3600)")

      // Normalization
      .def_readwrite("normalization_mode", &Options::normalization_mode,
                     "Text normalization mode (default: NONE)")
      .def_readwrite("normalization_max_bytes", &Options::normalization_max_bytes,
                     "Max bytes to normalize (default: 1MB)")

#ifdef PRESTIGE_ENABLE_SEMANTIC
      // Semantic dedup
      .def_readwrite("dedup_mode", &Options::dedup_mode,
                     "Deduplication mode (default: EXACT)")
      .def_readwrite("semantic_model_path", &Options::semantic_model_path,
                     "Path to ONNX model file (required for SEMANTIC mode)")
      .def_readwrite("semantic_model_type", &Options::semantic_model_type,
                     "Semantic model type (default: MINILM)")
      .def_readwrite("semantic_threshold", &Options::semantic_threshold,
                     "Cosine similarity threshold for dedup (default: -1)")
      .def_readwrite("semantic_max_text_bytes", &Options::semantic_max_text_bytes,
                     "Max text bytes to embed (default: 8192)")
      .def_readwrite("semantic_index_type", &Options::semantic_index_type,
                     "Vector index backend (default: HNSW)")
      .def_readwrite("semantic_search_k", &Options::semantic_search_k,
                     "Nearest neighbors to search (default: 10)")
      .def_readwrite("semantic_index_save_interval",
                     &Options::semantic_index_save_interval,
                     "Auto-save interval in inserts (default: 1000)")

      // HNSW parameters
      .def_readwrite("hnsw_m", &Options::hnsw_m,
                     "HNSW max connections per node (default: 16)")
      .def_readwrite("hnsw_ef_construction", &Options::hnsw_ef_construction,
                     "HNSW build-time search depth (default: 200)")
      .def_readwrite("hnsw_ef_search", &Options::hnsw_ef_search,
                     "HNSW query-time search depth (default: 50)")

#ifdef PRESTIGE_USE_FAISS
      // FAISS parameters
      .def_readwrite("faiss_nlist", &Options::faiss_nlist,
                     "FAISS IVF clusters (default: 100)")
      .def_readwrite("faiss_nprobe", &Options::faiss_nprobe,
                     "FAISS search clusters (default: 10)")
      .def_readwrite("faiss_pq_m", &Options::faiss_pq_m,
                     "FAISS PQ sub-quantizers (default: 48)")
      .def_readwrite("faiss_pq_nbits", &Options::faiss_pq_nbits,
                     "FAISS PQ bits per quantizer (default: 8)")
#endif

      // Vector index limits
      .def_readwrite("semantic_max_index_entries",
                     &Options::semantic_max_index_entries,
                     "Max vector index entries (default: 0=unlimited)")
      .def_readwrite("semantic_compact_threshold",
                     &Options::semantic_compact_threshold,
                     "Compaction threshold (default: 10000)")
#endif

      // Repr
      .def("__repr__", [](const Options& o) {
        std::string repr = "<prestige.Options";
#ifdef PRESTIGE_ENABLE_SEMANTIC
        repr += " dedup_mode=";
        repr += (o.dedup_mode == DedupMode::kExact ? "EXACT" : "SEMANTIC");
#endif
        repr += " block_cache_bytes=" + std::to_string(o.block_cache_bytes);
        repr += " default_ttl_seconds=" + std::to_string(o.default_ttl_seconds);
        repr += ">";
        return repr;
      });

  // HealthStats (read-only structure returned by get_health)
  py::class_<HealthStats>(m, "HealthStats", "Store health statistics")
      .def_readonly("total_keys", &HealthStats::total_keys,
                    "Total number of keys")
      .def_readonly("total_objects", &HealthStats::total_objects,
                    "Total number of unique objects")
      .def_readonly("total_bytes", &HealthStats::total_bytes,
                    "Total storage size in bytes")
      .def_readonly("expired_objects", &HealthStats::expired_objects,
                    "Number of expired objects")
      .def_readonly("orphaned_objects", &HealthStats::orphaned_objects,
                    "Number of orphaned objects (refcount=0)")
      .def_readonly("oldest_object_age_s", &HealthStats::oldest_object_age_s,
                    "Age of oldest object in seconds")
      .def_readonly("newest_access_age_s", &HealthStats::newest_access_age_s,
                    "Time since most recent access in seconds")
      .def_readonly("dedup_ratio", &HealthStats::dedup_ratio,
                    "Deduplication ratio (keys / objects)")
      .def("__repr__", [](const HealthStats& h) {
        return "<prestige.HealthStats keys=" + std::to_string(h.total_keys) +
               " objects=" + std::to_string(h.total_objects) +
               " bytes=" + std::to_string(h.total_bytes) +
               " dedup_ratio=" + std::to_string(h.dedup_ratio) + ">";
      });
}

}  // namespace prestige::python
