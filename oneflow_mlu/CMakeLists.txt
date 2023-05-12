include(cmake/mlu.cmake)

set(ONEFLOW_MLU_INCLUDE_DIRS ${NEUWARE_INCLUDE_DIRS})
set(ONEFLOW_MLU_LIBS ${NEUWARE_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB_RECURSE ONEFLOW_MLU_BANG_SRCS *.mlu)
if(ONEFLOW_MLU_BANG_SRCS)
  include(cmake/bang.cmake)
  bang_add_library(custom_bangc STATIC ${ONEFLOW_MLU_BANG_SRCS})
  list(APPEND ONEFLOW_MLU_LIBS custom_bangc)
endif()

file(
  GLOB_RECURSE
  ONEFLOW_MLU_SRCS
  cnnl/*.cpp
  collective_communication/*.cpp
  common/*.cpp
  ep/*.cpp
  graph/*.cpp
  kernels/*.cpp)
add_library(oneflow_mlu SHARED ${ONEFLOW_MLU_SRCS})
add_dependencies(oneflow_mlu ${ONEFLOW_THIRD_PARTY_LIBRARIES})
target_include_directories(oneflow_mlu PRIVATE ${ONEFLOW_MLU_INCLUDE_DIRS} ${ONEFLOW_INCLUDE_DIR})
target_link_libraries(oneflow_mlu oneflow ${ONEFLOW_MLU_LIBS} ${ONEFLOW_THIRD_PARTY_LIBRARIES})
#    -Wl,--version-script ${PROJECT_SOURCE_DIR}/oneflow_mlu/version_script.lds)

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()

set(ONEFLOW_MLU_INSTALL_TARGETS oneflow_mlu)

install(
  TARGETS ${ONEFLOW_MLU_INSTALL_TARGETS}
  COMPONENT oneflow_mlu_libs
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

add_custom_target(
  install_oneflow_mlu ALL
  COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=oneflow_mlu_libs
          -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
  DEPENDS ${ONEFLOW_MLU_INSTALL_TARGETS})
