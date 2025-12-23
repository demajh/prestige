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

# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include GNUInstallDirs early for CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)

# Suppress C++20 extension warnings from RocksDB headers
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
  add_compile_options(-Wno-c++20-extensions)
endif()

option(PRESTIGE_BUILD_TOOLS "Build CLI/tools" ON)
option(PRESTIGE_BUILD_EXAMPLES "Build examples" ON)
option(PRESTIGE_BUILD_TESTS "Build unit tests" OFF)
option(PRESTIGE_BUILD_BENCHMARKS "Build performance benchmarks" OFF)
option(PRESTIGE_ENABLE_COVERAGE "Enable code coverage" OFF)
option(PRESTIGE_ENABLE_SEMANTIC "Enable semantic deduplication (requires ONNX Runtime)" OFF)
option(PRESTIGE_USE_FAISS "Enable FAISS backend for semantic dedup (requires FAISS)" OFF)

# --- Find RocksDB ---
# This is intentionally simple. In production, you might prefer pkg-config or a dedicated FindRocksDB.cmake.
find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h)
find_library(ROCKSDB_LIBRARY rocksdb)

if(NOT ROCKSDB_INCLUDE_DIR OR NOT ROCKSDB_LIBRARY)
  message(FATAL_ERROR "Could not find RocksDB. Set ROCKSDB_INCLUDE_DIR and ROCKSDB_LIBRARY or install RocksDB dev packages.")
endif()

# --- Find OpenSSL (required for SHA-256) ---
find_package(OpenSSL REQUIRED)
message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")

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
set(PRESTIGE_SOURCES
  src/store.cpp
  src/normalize.cpp
  src/shutdown.cpp
)

if(PRESTIGE_ENABLE_SEMANTIC)
  list(APPEND PRESTIGE_SOURCES
    src/embedder.cpp
    src/tokenizer.cpp
    src/vector_index.cpp
  )
endif()

add_library(prestige_uvs ${PRESTIGE_SOURCES})

target_include_directories(prestige_uvs
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${ROCKSDB_INCLUDE_DIR}
)

target_link_libraries(prestige_uvs
  PUBLIC
    ${ROCKSDB_LIBRARY}
    OpenSSL::Crypto
    pthread
)

# Semantic dedup includes and libraries
if(PRESTIGE_ENABLE_SEMANTIC)
  target_include_directories(prestige_uvs
    PUBLIC
      ${ONNXRUNTIME_INCLUDE_DIR}
      $<BUILD_INTERFACE:${hnswlib_SOURCE_DIR}>
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
include(CMakePackageConfigHelpers)

# Set library properties
set_target_properties(prestige_uvs PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/prestige/store.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/prestige/internal.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/prestige/version.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/prestige/normalize.hpp"
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

# --- Testing ---
if(PRESTIGE_BUILD_TESTS)
  enable_testing()

  # Fetch Google Test
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  include(GoogleTest)

  # Coverage flags
  if(PRESTIGE_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
      add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
      add_link_options(--coverage)
    endif()
  endif()

  # Unit tests
  add_executable(prestige_tests
    tests/internal_test.cpp
    tests/normalize_test.cpp
    tests/store_test.cpp
    tests/cache_test.cpp
    tests/integration_test.cpp
  )

  target_link_libraries(prestige_tests
    PRIVATE
      prestige_uvs
      GTest::gtest
      GTest::gtest_main
  )

  target_include_directories(prestige_tests
    PRIVATE
      ${ROCKSDB_INCLUDE_DIR}
  )

  gtest_discover_tests(prestige_tests
    DISCOVERY_TIMEOUT 60
    PROPERTIES
      TIMEOUT 300
  )

  # Add coverage target
  if(PRESTIGE_ENABLE_COVERAGE)
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)

    if(LCOV_PATH AND GENHTML_PATH)
      add_custom_target(coverage
        COMMAND ${LCOV_PATH} --directory . --zerocounters
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage.info
        COMMAND ${LCOV_PATH} --remove coverage.info '/usr/*' '*/googletest/*' '*/tests/*' --output-file coverage.info.cleaned
        COMMAND ${GENHTML_PATH} -o coverage coverage.info.cleaned
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report..."
      )
    endif()
  endif()
endif()

# --- Benchmarks ---
if(PRESTIGE_BUILD_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)

  add_executable(prestige_benchmarks
    benchmarks/store_benchmark.cpp
  )

  target_link_libraries(prestige_benchmarks
    PRIVATE
      prestige_uvs
      benchmark::benchmark
      benchmark::benchmark_main
  )

  target_include_directories(prestige_benchmarks
    PRIVATE
      ${ROCKSDB_INCLUDE_DIR}
  )

  # Benchmark comparison target
  add_custom_target(benchmark_baseline
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/prestige_benchmarks
            --benchmark_out=${CMAKE_CURRENT_BINARY_DIR}/benchmark_baseline.json
            --benchmark_out_format=json
    COMMENT "Running benchmarks and saving baseline..."
  )

  add_custom_target(benchmark_compare
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/prestige_benchmarks
            --benchmark_out=${CMAKE_CURRENT_BINARY_DIR}/benchmark_current.json
            --benchmark_out_format=json
    COMMENT "Running benchmarks for comparison..."
  )
endif()

# --- Code Quality Targets ---
find_program(CLANG_FORMAT_PATH clang-format)
find_program(CLANG_TIDY_PATH clang-tidy)

if(CLANG_FORMAT_PATH)
  file(GLOB_RECURSE ALL_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp
  )

  add_custom_target(format
    COMMAND ${CLANG_FORMAT_PATH} -i ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on all source files..."
  )

  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_PATH} --dry-run --Werror ${ALL_SOURCE_FILES}
    COMMENT "Checking code formatting..."
  )
endif()

if(CLANG_TIDY_PATH)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY_PATH}
            -p ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    COMMENT "Running clang-tidy on source files..."
  )
endif()
