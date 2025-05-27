# Define the executable for student_app1
add_executable(04_semaphore_isr main.c port_simulated_interrupt.c)

# Include directories
target_include_directories(04_semaphore_isr PUBLIC
    ${CMAKE_SOURCE_DIR}/apps/common  # Include common_utils headers
)

# Link against common dependencies
target_link_libraries(04_semaphore_isr FreeRTOS pthread)

# Optional: Link against shared utilities if needed
target_link_libraries(04_semaphore_isr common_utils)
