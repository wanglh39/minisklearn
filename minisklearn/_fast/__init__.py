"""minisklearn._fast —— C++ 加速模块。

本模块封装 C++ 编译的 _minisklearn_fast 扩展，提供与纯 Python 实现等价
但更快的核心计算函数。

使用前需要先编译 C++ 扩展：
    python cpp/build.py

如果未编译，import 会失败但不会影响 minisklearn 其他功能。

Windows 注意事项：
    MinGW 编译的 .pyd 依赖 MinGW 运行时 DLL（libstdc++ 等），
    需要将 MinGW\\bin 加入 PATH 或用 os.add_dll_directory 加载。
"""

import os
import sys


def _load_extension():
    """尝试加载 C++ 扩展，处理 Windows DLL 依赖。"""
    try:
        from ._minisklearn_fast import (
            euclidean_distances,
            knn_neighbors,
            kmeans_assign,
            kmeans_update,
            kmeans_inertia,
        )
        return {
            "euclidean_distances": euclidean_distances,
            "knn_neighbors": knn_neighbors,
            "kmeans_assign": kmeans_assign,
            "kmeans_update": kmeans_update,
            "kmeans_inertia": kmeans_inertia,
        }
    except ImportError:
        pass

    # Windows: 尝试添加 MinGW bin 到 DLL 搜索路径
    if sys.platform == "win32":
        mingw_paths = [
            r"C:\mingw64\bin",
            os.path.join(os.environ.get("MINGW_PREFIX", ""), "bin"),
        ]
        for path in mingw_paths:
            if os.path.isdir(path):
                try:
                    os.add_dll_directory(path)
                except (OSError, AttributeError):
                    pass
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

        try:
            from ._minisklearn_fast import (
                euclidean_distances,
                knn_neighbors,
                kmeans_assign,
                kmeans_update,
                kmeans_inertia,
            )
            return {
                "euclidean_distances": euclidean_distances,
                "knn_neighbors": knn_neighbors,
                "kmeans_assign": kmeans_assign,
                "kmeans_update": kmeans_update,
                "kmeans_inertia": kmeans_inertia,
            }
        except ImportError:
            pass

    return None


_ext = _load_extension()
_available = _ext is not None

if _available:
    euclidean_distances = _ext["euclidean_distances"]
    knn_neighbors = _ext["knn_neighbors"]
    kmeans_assign = _ext["kmeans_assign"]
    kmeans_update = _ext["kmeans_update"]
    kmeans_inertia = _ext["kmeans_inertia"]
else:
    def _not_available(*args, **kwargs):
        raise ImportError(
            "C++ 扩展未编译或加载失败。请运行: python cpp/build.py"
        )

    euclidean_distances = _not_available
    knn_neighbors = _not_available
    kmeans_assign = _not_available
    kmeans_update = _not_available
    kmeans_inertia = _not_available


def is_available():
    """检查 C++ 扩展是否可用。"""
    return _available


__all__ = [
    "euclidean_distances",
    "knn_neighbors",
    "kmeans_assign",
    "kmeans_update",
    "kmeans_inertia",
    "is_available",
]
