"""Setup script for prestige Python bindings."""

import os
import sys
import platform
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.1.0"

# Detect platform
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


class CMakeExtension(Extension):
    """CMake-based extension module."""

    def __init__(self, name: str, cmake_lists_dir: str = ".."):
        super().__init__(name, sources=[])
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    """Custom build command that uses CMake."""

    def build_extension(self, ext: CMakeExtension):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        build_temp = Path(self.build_temp).absolute()
        build_temp.mkdir(parents=True, exist_ok=True)

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
            "-DPRESTIGE_BUILD_PYTHON=ON",
            "-DPRESTIGE_BUILD_TESTS=OFF",
            "-DPRESTIGE_BUILD_EXAMPLES=OFF",
            "-DPRESTIGE_BUILD_TOOLS=OFF",
            "-DPRESTIGE_BUILD_BENCHMARKS=OFF",
        ]

        # Handle optional features from environment
        if os.environ.get("PRESTIGE_ENABLE_SEMANTIC", "").lower() in ("1", "on", "true"):
            cmake_args.append("-DPRESTIGE_ENABLE_SEMANTIC=ON")
            if onnx_path := os.environ.get("ONNXRUNTIME_DIR"):
                cmake_args.append(f"-DONNXRUNTIME_DIR={onnx_path}")

        if os.environ.get("PRESTIGE_BUILD_SERVER", "").lower() in ("1", "on", "true"):
            cmake_args.append("-DPRESTIGE_BUILD_SERVER=ON")

        # Platform-specific settings
        build_args = ["--config", "Debug" if self.debug else "Release"]

        if IS_WINDOWS:
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{'DEBUG' if self.debug else 'RELEASE'}={ext_dir}",
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            build_args += ["--", f"-j{os.cpu_count() or 1}"]

        if IS_MAC:
            # macOS: handle universal2 builds
            archs = os.environ.get("CMAKE_OSX_ARCHITECTURES", "")
            if archs:
                cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={archs}")

            # Set deployment target
            cmake_args.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=10.14")

        # Configure
        print(f"CMake configure: {' '.join(cmake_args)}")
        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=build_temp
        )

        # Build
        print(f"CMake build: {' '.join(build_args)}")
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_prestige"] + build_args,
            cwd=build_temp,
        )


def get_long_description():
    """Read README for long description."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    # Fall back to parent README
    readme_path = Path(__file__).parent.parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Prestige: Content-deduplicated key-value store"


setup(
    name="prestige",
    version=__version__,
    author="Prestige Authors",
    description="Content-deduplicated key-value store with optional semantic deduplication",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/demajh/prestige",
    packages=["prestige", "prestige.dataloaders"],
    ext_modules=[CMakeExtension("prestige._prestige")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Database",
    ],
)
