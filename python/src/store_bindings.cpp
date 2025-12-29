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
  void Put(const std::string& key, py::object value) {
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
    {
      py::gil_scoped_release release;
      status = store_->Put(key, value_bytes);
    }
    CheckStatus(status);
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
                    "dedup_ratio"_a = stats.dedup_ratio);
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
      .def("put", &PyStore::Put, py::arg("key"), py::arg("value"),
           R"doc(Store a key-value pair.

        The value is automatically deduplicated by content hash.

        Args:
            key: Key string
            value: Value as bytes or str
        )doc")

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
      .def("close", &PyStore::Close, "Close the store.")

      // Context manager
      .def("__enter__", &PyStore::Enter, py::return_value_policy::reference)
      .def("__exit__", &PyStore::Exit)

      // Dict-like interface
      .def("__contains__", &PyStore::Contains)
      .def("__setitem__", &PyStore::Put)
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
