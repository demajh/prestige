/**
 * Exception handling for Prestige Python bindings.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <rocksdb/status.h>

#include <string>

namespace prestige::python {

/**
 * Register exception types with the Python module.
 */
void RegisterExceptions(pybind11::module_& m);

/**
 * Check a rocksdb::Status and throw appropriate Python exception if not OK.
 */
void CheckStatus(const rocksdb::Status& status);

/**
 * Check a rocksdb::Status with a custom "not found" message including the key.
 */
void CheckStatusNotFound(const rocksdb::Status& status, const std::string& key);

}  // namespace prestige::python
