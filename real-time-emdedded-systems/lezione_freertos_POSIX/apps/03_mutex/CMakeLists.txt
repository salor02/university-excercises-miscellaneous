# Define the executable for student_app1
add_executable(03_mutex main.c)

# Include directories
target_include_directories(03_mutex PUBLIC
    ${CMAKE_SOURCE_DIR}/apps/common  # Include common_utils headers
)

# Link against common dependencies
target_link_libraries(03_mutex FreeRTOS pthread)

# Optional: Link against shared utilities if needed
target_link_libraries(03_mutex common_utils)
