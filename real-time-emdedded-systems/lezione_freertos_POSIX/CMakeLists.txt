cmake_minimum_required(VERSION 3.12)
project(MyFreeRTOSApp C CXX)

# Set language standards.
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Set a global output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Define a global compile definition.
add_compile_definitions(PLATFORM_NAME_FREERTOS)

# Define FreeRTOS paths.
set(FREERTOS_KERNEL_PATH "${CMAKE_SOURCE_DIR}/FreeRTOS-Kernel")
set(FREERTOS_PORT_PATH "${FREERTOS_KERNEL_PATH}/portable/ThirdParty/GCC/Posix")
set(FREERTOS_HEAP_PATH "${FREERTOS_KERNEL_PATH}/portable/MemMang")

# Include FreeRTOS header directories.
include_directories(
    ${CMAKE_SOURCE_DIR}  # This ensures FreeRTOSConfig.h is found
    ${FREERTOS_KERNEL_PATH}/include
    ${FREERTOS_PORT_PATH}
    ${FREERTOS_PORT_PATH}/utils
)

add_compile_options(-DprojCOVERAGE_TEST=0)
add_compile_options(-DprojENABLE_TRACING=0)

# Gather FreeRTOS kernel sources.
file(GLOB FREERTOS_KERNEL_SOURCES
    "${FREERTOS_KERNEL_PATH}/*.c"
    "${FREERTOS_HEAP_PATH}/heap_4.c"         # Change heap implementation if needed.
    "${FREERTOS_PORT_PATH}/*.c"
    "${FREERTOS_PORT_PATH}/utils/*.c"
)

# Create a static library for FreeRTOS.
add_library(FreeRTOS STATIC ${FREERTOS_KERNEL_SOURCES})

# Expose include directories for consumers.
target_include_directories(FreeRTOS PUBLIC
        ${FREERTOS_KERNEL_PATH}/include
        ${FREERTOS_PORT_PATH}/include
        ${CMAKE_CURRENT_LIST_DIR}
    )

# Set additional compiler flags (optional).
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")

# Add user applications
add_subdirectory(apps/common)  # Shared utilities
add_subdirectory(apps/01_tasks)
add_subdirectory(apps/02_queues)
add_subdirectory(apps/03_mutex)
add_subdirectory(apps/04_semaphore_isr)
add_subdirectory(apps/05_timers)
