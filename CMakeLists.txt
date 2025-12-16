cmake_minimum_required(VERSION 3.16)
project(prestige_uvs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PRESTIGE_BUILD_TOOLS "Build CLI/tools" ON)
option(PRESTIGE_BUILD_EXAMPLES "Build examples" ON)

# --- Find RocksDB ---
# This is intentionally simple. In production, you might prefer pkg-config or a dedicated FindRocksDB.cmake.
find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h)
find_library(ROCKSDB_LIBRARY rocksdb)

if(NOT ROCKSDB_INCLUDE_DIR OR NOT ROCKSDB_LIBRARY)
  message(FATAL_ERROR "Could not find RocksDB. Set ROCKSDB_INCLUDE_DIR and ROCKSDB_LIBRARY or install RocksDB dev packages.")
endif()

add_library(prestige_uvs
  src/store.cpp
)

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

# Some distros require these explicitly; harmless elsewhere.
find_library(DL_LIBRARY dl)
if(DL_LIBRARY)
  target_link_libraries(prestige_uvs PUBLIC ${DL_LIBRARY})
endif()

if(PRESTIGE_BUILD_EXAMPLES)
  add_executable(prestige_example_basic examples/basic.cpp)
  target_link_libraries(prestige_example_basic PRIVATE prestige_uvs)
endif()

if(PRESTIGE_BUILD_TOOLS)
  add_executable(prestige_cli tools/prestige_cli.cpp)
  target_link_libraries(prestige_cli PRIVATE prestige_uvs)
endif()
