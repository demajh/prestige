/**
 * Main pybind11 module definition for Prestige.
 */

#include <pybind11/pybind11.h>
#include <prestige/version.hpp>

#include "exceptions.hpp"
#include "options_bindings.hpp"
#include "store_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_prestige, m) {
  m.doc() = R"doc(
Prestige: Content-deduplicated key-value store.

Prestige is a high-performance key-value store with automatic content
deduplication. Identical values are stored only once, reducing storage
requirements for data with redundancy.

Basic usage:
    import prestige

    # Context manager (recommended)
    with prestige.open("/path/to/db") as store:
        store.put("key", "value")
        value = store.get("key", decode=True)

    # Or explicit open/close
    store = prestige.Store.open("/path/to/db")
    store.put("key", "value")
    store.close()
)doc";

  // Register exceptions first
  prestige::python::RegisterExceptions(m);

  // Bind options and enums
  prestige::python::BindOptions(m);

  // Bind Store class
  prestige::python::BindStore(m);

  // Version info
  m.attr("__version__") = prestige::Version();

  // Feature availability flags
#ifdef PRESTIGE_ENABLE_SEMANTIC
  m.attr("SEMANTIC_AVAILABLE") = true;
#else
  m.attr("SEMANTIC_AVAILABLE") = false;
#endif

#ifdef PRESTIGE_BUILD_SERVER
  m.attr("SERVER_AVAILABLE") = true;
#else
  m.attr("SERVER_AVAILABLE") = false;
#endif
}
