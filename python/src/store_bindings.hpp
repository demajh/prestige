/**
 * Store class bindings for Prestige Python bindings.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace prestige::python {

/**
 * Bind Store class to the Python module.
 */
void BindStore(pybind11::module_& m);

}  // namespace prestige::python
