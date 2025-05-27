# Define the executable for student_app1
add_executable(02_queues main.c)

# Include directories
target_include_directories(02_queues PUBLIC
    ${CMAKE_SOURCE_DIR}/apps/common  # Include common_utils headers
)

# Link against common dependencies
target_link_libraries(02_queues FreeRTOS pthread)

# Optional: Link against shared utilities if needed
target_link_libraries(02_queues common_utils)
