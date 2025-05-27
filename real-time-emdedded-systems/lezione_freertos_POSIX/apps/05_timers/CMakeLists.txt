# Define the executable for student_app1
add_executable(05_timers main.c)

# Include directories
target_include_directories(05_timers PUBLIC
    ${CMAKE_SOURCE_DIR}/apps/common  # Include common_utils headers
)

# Link against common dependencies
target_link_libraries(05_timers FreeRTOS pthread)

# Optional: Link against shared utilities if needed
target_link_libraries(05_timers common_utils)
