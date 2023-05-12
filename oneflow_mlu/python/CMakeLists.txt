set(ONEFLOW_MLU_PYTHON_DIR ${PROJECT_SOURCE_DIR}/python)

pybind11_add_module(oneflow_mlu_internal SHARED stub.cpp)
set_target_properties(oneflow_mlu_internal PROPERTIES CXX_VISIBILITY_PRESET "default")
set_target_properties(oneflow_mlu_internal PROPERTIES PREFIX "_")
set_target_properties(oneflow_mlu_internal
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${ONEFLOW_MLU_PYTHON_DIR}/oneflow_mlu")
add_dependencies(oneflow_mlu_internal oneflow_mlu)
target_include_directories(oneflow_mlu_internal PRIVATE
  ${Python_INCLUDE_DIRS} ${ONEFLOW_MLU_INCLUDE_DIRS} ${ONEFLOW_INCLUDE_DIR})
target_link_libraries(oneflow_mlu_internal PUBLIC -Wl,--no-as-needed oneflow_mlu)
