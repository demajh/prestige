/**
 * Options and enum bindings for Prestige Python bindings.
 */

#pragma once

#include <pybind11/pybind11.h>

namespace prestige::python {

/**
 * Bind Options struct and related enums to the Python module.
 */
void BindOptions(pybind11::module_& m);

}  // namespace prestige::python
