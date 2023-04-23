find_package(OpenCL REQUIRED)
message(STATUS "OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL_LIBRARIES = ${OpenCL_LIBRARIES}")

file(GLOB_RECURSE ONEFLOW_OCL_SRCS *.cpp *.cc)
add_library(oneflow_ocl SHARED ${ONEFLOW_OCL_SRCS})
add_dependencies(oneflow_ocl oneflow)
target_include_directories(oneflow_ocl PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(oneflow_ocl oneflow ${OpenCL_LIBRARIES})

set_property(GLOBAL APPEND PROPERTY ONEFLOW_BACKEND_LIBS oneflow_ocl)
