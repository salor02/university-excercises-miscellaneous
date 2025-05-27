# Define the executable for student_app1
add_executable(01_tasks main.c)

# Include directories
target_include_directories(01_tasks PUBLIC
    ${CMAKE_SOURCE_DIR}/apps/common  # Include common_utils headers
)

# Link against common dependencies
target_link_libraries(01_tasks FreeRTOS pthread)

# Optional: Link against shared utilities if needed
target_link_libraries(01_tasks common_utils)
