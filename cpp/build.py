"""编译 C++ 扩展模块的脚本。

用法：
    python cpp/build.py

原理：
    1. 用 CMake 配置构建系统
    2. 用 CMake 编译生成 _minisklearn_fast.pyd（Windows）或 .so（Linux）
    3. 输出文件直接放到 minisklearn/_fast/ 目录
    4. Python 可通过 `from minisklearn._fast import _minisklearn_fast` 导入
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def find_cmake():
    """查找 cmake 可执行文件。"""
    # 优先用 venv 中的 cmake
    venv_cmake = Path(sys.executable).parent / "cmake.exe"
    if venv_cmake.exists():
        return str(venv_cmake)
    # 回退到 PATH 中的 cmake
    return "cmake"


def get_pybind11_cmake_dir():
    """获取 pybind11 的 CMake 配置目录。"""
    result = subprocess.run(
        [sys.executable, "-c", "import pybind11; print(pybind11.get_cmake_dir())"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def main():
    cpp_dir = Path(__file__).parent
    build_dir = cpp_dir / "build"

    cmake = find_cmake()
    pybind11_dir = get_pybind11_cmake_dir()
    print(f"cmake: {cmake}")
    print(f"pybind11 CMake 目录: {pybind11_dir}")

    # 设置环境（MinGW 在 PATH 中）
    env = os.environ.copy()
    if sys.platform == "win32":
        mingw = r"C:\mingw64\bin"
        if Path(mingw).exists():
            env["PATH"] = mingw + os.pathsep + env["PATH"]

    # 清除旧构建
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # CMake 配置
    print("=== CMake 配置 ===")
    cmake_args = [
        cmake,
        "-S", str(cpp_dir),
        "-B", str(build_dir),
        f"-Dpybind11_DIR={pybind11_dir}",
    ]
    if sys.platform == "win32":
        cmake_args.extend(["-G", "MinGW Makefiles"])
    subprocess.run(cmake_args, check=True, env=env)

    # CMake 构建
    print("=== CMake 构建 ===")
    subprocess.run(
        [cmake, "--build", str(build_dir), "--config", "Release"],
        check=True, env=env
    )

    # 验证输出
    fast_dir = cpp_dir.parent / "minisklearn" / "_fast"
    ext_files = list(fast_dir.glob("_minisklearn_fast.*"))

    if ext_files:
        print(f"\n编译成功！输出文件: {ext_files[0]}")
    else:
        print("\n警告：未找到输出文件")
        print("请检查 CMake 构建输出。")


if __name__ == "__main__":
    main()
