cmake_minimum_required(VERSION 3.15)
project(minisklearn_fast LANGUAGES CXX)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 释放优化（-O3 / /O2）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 查找 pybind11
find_package(pybind11 REQUIRED)

# 编译扩展模块
pybind11_add_module(_minisklearn_fast
    src/distances.cpp
    src/kmeans.cpp
    src/bindings.cpp
)

# 输出目录设为 minisklearn/_fast/
set_target_properties(_minisklearn_fast PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../minisklearn/_fast
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../minisklearn/_fast
)