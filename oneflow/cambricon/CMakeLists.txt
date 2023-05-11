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

file(GLOB_RECURSE ONEFLOW_MLU_SRCS *.cpp *.cc)
add_library(oneflow_mlu SHARED ${ONEFLOW_MLU_SRCS})
add_dependencies(oneflow_mlu ${ONEFLOW_THIRD_PARTY_LIBRARIES})
target_include_directories(oneflow_mlu PRIVATE ${ONEFLOW_MLU_INCLUDE_DIRS} ${ONEFLOW_INCLUDE_DIR})
target_link_libraries(oneflow_mlu oneflow ${ONEFLOW_MLU_LIBS} ${ONEFLOW_THIRD_PARTY_LIBRARIES})
