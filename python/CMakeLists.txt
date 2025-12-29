# CMakeLists.txt for Prestige Python bindings

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Try to find pybind11 via different methods
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Try to find via Python module
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        # Fetch pybind11 if not found
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

message(STATUS "pybind11 found: ${pybind11_VERSION}")

# Collect binding sources
set(PRESTIGE_PYTHON_SOURCES
    src/bindings.cpp
    src/exceptions.cpp
    src/options_bindings.cpp
    src/store_bindings.cpp
)

# Conditionally add optional binding sources
if(PRESTIGE_ENABLE_SEMANTIC OR PRESTIGE_BUILD_SERVER)
    # Observability bindings (for metrics sink used by both)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/observability_bindings.cpp)
        list(APPEND PRESTIGE_PYTHON_SOURCES src/observability_bindings.cpp)
    endif()
endif()

if(PRESTIGE_BUILD_SERVER)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/server_bindings.cpp)
        list(APPEND PRESTIGE_PYTHON_SOURCES src/server_bindings.cpp)
    endif()
endif()

# Create pybind11 module
pybind11_add_module(_prestige MODULE ${PRESTIGE_PYTHON_SOURCES})

# Link against prestige library
target_link_libraries(_prestige PRIVATE prestige_uvs)

# Include directories
target_include_directories(_prestige PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# C++17 required
target_compile_features(_prestige PRIVATE cxx_std_17)

# Compile definitions for feature detection
if(PRESTIGE_ENABLE_SEMANTIC)
    target_compile_definitions(_prestige PRIVATE PRESTIGE_ENABLE_SEMANTIC=1)
endif()

if(PRESTIGE_BUILD_SERVER)
    target_compile_definitions(_prestige PRIVATE PRESTIGE_BUILD_SERVER=1)
    target_link_libraries(_prestige PRIVATE prestige_server)
endif()

# Platform-specific settings
if(APPLE)
    # Don't add rpath, we'll bundle dependencies
    set_target_properties(_prestige PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif(UNIX)
    set_target_properties(_prestige PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Set output directory for development builds
# Use CMAKE_LIBRARY_OUTPUT_DIRECTORY if specified (e.g., by setup.py),
# otherwise default to source directory for development
if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    # Let setup.py control the output directory
    set_target_properties(_prestige PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
else()
    # Development build - output to source directory
    set_target_properties(_prestige PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/prestige
    )
endif()

# Install target
install(TARGETS _prestige
    LIBRARY DESTINATION prestige
)
