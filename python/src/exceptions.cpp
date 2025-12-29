/**
 * Exception handling for Prestige Python bindings.
 *
 * Maps rocksdb::Status to Python exceptions.
 */

#include "exceptions.hpp"

#include <pybind11/pybind11.h>
#include <rocksdb/status.h>

namespace py = pybind11;

namespace prestige::python {

// Static exception type pointers
static PyObject* PrestigeError = nullptr;
static PyObject* NotFoundError = nullptr;
static PyObject* InvalidArgumentError = nullptr;
static PyObject* IOError_ = nullptr;
static PyObject* CorruptionError = nullptr;
static PyObject* BusyError = nullptr;
static PyObject* TimedOutError = nullptr;

void RegisterExceptions(py::module_& m) {
  // Base exception
  PrestigeError =
      PyErr_NewException("prestige.PrestigeError", PyExc_Exception, nullptr);
  py::setattr(m, "PrestigeError", py::handle(PrestigeError));

  // Derived exceptions
  NotFoundError =
      PyErr_NewException("prestige.NotFoundError", PrestigeError, nullptr);
  py::setattr(m, "NotFoundError", py::handle(NotFoundError));

  InvalidArgumentError = PyErr_NewException("prestige.InvalidArgumentError",
                                            PrestigeError, nullptr);
  py::setattr(m, "InvalidArgumentError", py::handle(InvalidArgumentError));

  IOError_ = PyErr_NewException("prestige.IOError", PrestigeError, nullptr);
  py::setattr(m, "IOError", py::handle(IOError_));

  CorruptionError =
      PyErr_NewException("prestige.CorruptionError", PrestigeError, nullptr);
  py::setattr(m, "CorruptionError", py::handle(CorruptionError));

  BusyError = PyErr_NewException("prestige.BusyError", PrestigeError, nullptr);
  py::setattr(m, "BusyError", py::handle(BusyError));

  TimedOutError =
      PyErr_NewException("prestige.TimedOutError", PrestigeError, nullptr);
  py::setattr(m, "TimedOutError", py::handle(TimedOutError));
}

void CheckStatus(const rocksdb::Status& status) {
  if (status.ok()) {
    return;
  }

  const std::string msg = status.ToString();

  if (status.IsNotFound()) {
    PyErr_SetString(NotFoundError, msg.c_str());
    throw py::error_already_set();
  }
  if (status.IsInvalidArgument()) {
    PyErr_SetString(InvalidArgumentError, msg.c_str());
    throw py::error_already_set();
  }
  if (status.IsIOError()) {
    PyErr_SetString(IOError_, msg.c_str());
    throw py::error_already_set();
  }
  if (status.IsCorruption()) {
    PyErr_SetString(CorruptionError, msg.c_str());
    throw py::error_already_set();
  }
  if (status.IsBusy() || status.IsTryAgain()) {
    PyErr_SetString(BusyError, msg.c_str());
    throw py::error_already_set();
  }
  if (status.IsTimedOut()) {
    PyErr_SetString(TimedOutError, msg.c_str());
    throw py::error_already_set();
  }

  // Generic error for other cases
  PyErr_SetString(PrestigeError, msg.c_str());
  throw py::error_already_set();
}

void CheckStatusNotFound(const rocksdb::Status& status,
                         const std::string& key) {
  if (status.ok()) {
    return;
  }

  if (status.IsNotFound()) {
    std::string msg = "Key not found: " + key;
    PyErr_SetString(NotFoundError, msg.c_str());
    throw py::error_already_set();
  }

  // Fall back to generic handling
  CheckStatus(status);
}

}  // namespace prestige::python
