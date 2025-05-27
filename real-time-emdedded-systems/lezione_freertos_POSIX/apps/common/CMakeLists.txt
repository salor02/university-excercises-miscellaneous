add_library(common_utils STATIC utils.c)
target_include_directories(common_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})