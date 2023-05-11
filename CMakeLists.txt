cmake_minimum_required(VERSION 3.18.0)

project(oneflow-cambricon CXX)
option(BUILD_PYTHON "Option to build python module" ON)

set(THIRD_PARTY_MIRROR "" CACHE STRING "")

include(cmake/third_party.cmake)
include(cmake/oneflow.cmake)
if(BUILD_PYTHON)
  include(cmake/pybind11.cmake)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(INSTALL_DIR "${PROJECT_BINARY_DIR}/install" CACHE STRING "")
add_subdirectory(oneflow_mlu)
