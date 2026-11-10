# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build the spacemit_avatar wheel with CMake, or from binaries an SDK build already produced."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        prebuilt = os.environ.get("AVATAR_PREBUILT_DIR")
        if prebuilt:
            # The SDK build already compiled the extension and avatar_worker; package them unchanged.
            source = Path(prebuilt).resolve()
            destination = Path(self.get_ext_fullpath(ext.name)).resolve().parent
            files = [path for path in source.rglob("*") if path.is_file()]
            if not files:
                raise FileNotFoundError(f"No prebuilt files in {source}")
            for item in files:
                target = destination / item.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
            return
        build = Path(self.build_temp).resolve() / ext.name.replace(".", "_")
        build.mkdir(parents=True, exist_ok=True)
        args = [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DAVATAR_WHEEL_LAYOUT=ON",
            "-DAVATAR_BUILD_WHEEL=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_INSTALL_PREFIX={Path(self.build_lib).resolve()}",
        ]
        for name in ["ONNXRUNTIME_INCLUDE_DIR", "OPENCV_INCLUDE_DIR", "pybind11_DIR"]:
            if os.environ.get(name):
                args.append(f"-D{name}={os.environ[name]}")
        if not os.environ.get("pybind11_DIR"):
            import pybind11

            args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2")
        subprocess.run(args, check=True)
        subprocess.run(["cmake", "--build", str(build), "--parallel", jobs], check=True)
        subprocess.run(["cmake", "--install", str(build)], check=True)


setup(
    version=(ROOT / "VERSION").read_text(encoding="utf-8").strip(),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=[Extension("spacemit_avatar._spacemit_avatar", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    package_data={"spacemit_avatar": ["*.so*", "bin/*"]},
)
