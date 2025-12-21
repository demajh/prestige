cmake_minimum_required(VERSION 3.16)
project(prestige_uvs
  VERSION 0.1.0
  DESCRIPTION "Content-deduplicated key-value store with exact and semantic dedup"
  HOMEPAGE_URL "https://github.com/demajh/prestige"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate position-independent code for shared library builds
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Suppress C++20 extension warnings from RocksDB headers
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  add_compile_options(-Wno-c++20-extensions)
endif()

option(PRESTIGE_BUILD_TOOLS "Build CLI/tools" ON)
option(PRESTIGE_BUILD_EXAMPLES "Build examples" ON)
option(PRESTIGE_ENABLE_SEMANTIC "Enable semantic deduplication (requires ONNX Runtime)" OFF)
option(PRESTIGE_USE_FAISS "Enable FAISS backend for semantic dedup (requires FAISS)" OFF)

# --- Find RocksDB ---
# This is intentionally simple. In production, you might prefer pkg-config or a dedicated FindRocksDB.cmake.
find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h)
find_library(ROCKSDB_LIBRARY rocksdb)

if(NOT ROCKSDB_INCLUDE_DIR OR NOT ROCKSDB_LIBRARY)
  message(FATAL_ERROR "Could not find RocksDB. Set ROCKSDB_INCLUDE_DIR and ROCKSDB_LIBRARY or install RocksDB dev packages.")
endif()

# --- Semantic dedup dependencies (optional) ---
if(PRESTIGE_ENABLE_SEMANTIC)
  # Find ONNX Runtime
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      /usr/local/include/onnxruntime
      /opt/homebrew/include/onnxruntime
      $ENV{ONNXRUNTIME_DIR}/include
  )
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS
      /usr/local/lib
      /opt/homebrew/lib
      $ENV{ONNXRUNTIME_DIR}/lib
  )

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "PRESTIGE_ENABLE_SEMANTIC is ON but ONNX Runtime not found. "
                        "Set ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY or install ONNX Runtime.")
  endif()

  message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")

  # Fetch hnswlib (header-only library)
  include(FetchContent)
  FetchContent_Declare(hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG v0.8.0
  )
  FetchContent_MakeAvailable(hnswlib)

  message(STATUS "hnswlib fetched: ${hnswlib_SOURCE_DIR}")

  # Optional FAISS support
  if(PRESTIGE_USE_FAISS)
    find_package(faiss REQUIRED)
    message(STATUS "FAISS found")
  endif()
endif()

# --- Library sources ---
set(PRESTIGE_SOURCES src/store.cpp)

if(PRESTIGE_ENABLE_SEMANTIC)
  list(APPEND PRESTIGE_SOURCES
    src/embedder.cpp
    src/vector_index.cpp
  )
endif()

add_library(prestige_uvs ${PRESTIGE_SOURCES})

target_include_directories(prestige_uvs
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ROCKSDB_INCLUDE_DIR}
)

target_link_libraries(prestige_uvs
  PUBLIC
    ${ROCKSDB_LIBRARY}
    pthread
)

# Semantic dedup includes and libraries
if(PRESTIGE_ENABLE_SEMANTIC)
  target_include_directories(prestige_uvs
    PUBLIC
      ${ONNXRUNTIME_INCLUDE_DIR}
      ${hnswlib_SOURCE_DIR}
  )

  target_link_libraries(prestige_uvs
    PUBLIC
      ${ONNXRUNTIME_LIBRARY}
  )

  target_compile_definitions(prestige_uvs
    PUBLIC
      PRESTIGE_ENABLE_SEMANTIC=1
  )

  if(PRESTIGE_USE_FAISS)
    target_link_libraries(prestige_uvs PUBLIC faiss)
    target_compile_definitions(prestige_uvs PUBLIC PRESTIGE_USE_FAISS=1)
  endif()
endif()

# Some distros require these explicitly; harmless elsewhere.
find_library(DL_LIBRARY dl)
if(DL_LIBRARY)
  target_link_libraries(prestige_uvs PUBLIC ${DL_LIBRARY})
endif()

if(PRESTIGE_BUILD_EXAMPLES)
  add_executable(prestige_example_basic examples/basic.cpp)
  target_link_libraries(prestige_example_basic PRIVATE prestige_uvs)
  add_executable(prestige_example_observability examples/observability.cpp)
  target_link_libraries(prestige_example_observability PRIVATE prestige_uvs)

  # Semantic example only when semantic dedup is enabled
  if(PRESTIGE_ENABLE_SEMANTIC)
    add_executable(prestige_example_semantic examples/semantic.cpp)
    target_link_libraries(prestige_example_semantic PRIVATE prestige_uvs)
  endif()
endif()

if(PRESTIGE_BUILD_TOOLS)
  add_executable(prestige_cli tools/prestige_cli.cpp)
  target_link_libraries(prestige_cli PRIVATE prestige_uvs)
endif()

# --- Installation ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Set library properties
set_target_properties(prestige_uvs PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/prestige/store.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/prestige/internal.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/prestige/version.hpp"
)

# Install library
install(TARGETS prestige_uvs
  EXPORT prestige_uvsTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/prestige
)

# Install additional headers
install(DIRECTORY include/prestige
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)

# Install CLI tool
if(PRESTIGE_BUILD_TOOLS)
  install(TARGETS prestige_cli
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

# Generate and install CMake package config
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/prestige_uvsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prestige_uvsConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/prestige_uvsConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/prestige_uvs
)

install(EXPORT prestige_uvsTargets
  FILE prestige_uvsTargets.cmake
  NAMESPACE prestige::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/prestige_uvs
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/prestige_uvsConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/prestige_uvsConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/prestige_uvs
)

# Generate pkg-config file
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prestige_uvs.pc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/prestige_uvs.pc"
  @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/prestige_uvs.pc"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
