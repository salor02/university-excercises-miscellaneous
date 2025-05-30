# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build

# Include any dependencies generated for this target.
include apps/03_mutex/CMakeFiles/03_mutex.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include apps/03_mutex/CMakeFiles/03_mutex.dir/compiler_depend.make

# Include the progress variables for this target.
include apps/03_mutex/CMakeFiles/03_mutex.dir/progress.make

# Include the compile flags for this target's objects.
include apps/03_mutex/CMakeFiles/03_mutex.dir/flags.make

apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.o: apps/03_mutex/CMakeFiles/03_mutex.dir/flags.make
apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.o: /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/03_mutex/main.c
apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.o: apps/03_mutex/CMakeFiles/03_mutex.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.o"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/03_mutex && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.o -MF CMakeFiles/03_mutex.dir/main.c.o.d -o CMakeFiles/03_mutex.dir/main.c.o -c /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/03_mutex/main.c

apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/03_mutex.dir/main.c.i"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/03_mutex && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/03_mutex/main.c > CMakeFiles/03_mutex.dir/main.c.i

apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/03_mutex.dir/main.c.s"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/03_mutex && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/03_mutex/main.c -o CMakeFiles/03_mutex.dir/main.c.s

# Object files for target 03_mutex
03_mutex_OBJECTS = \
"CMakeFiles/03_mutex.dir/main.c.o"

# External object files for target 03_mutex
03_mutex_EXTERNAL_OBJECTS =

03_mutex: apps/03_mutex/CMakeFiles/03_mutex.dir/main.c.o
03_mutex: apps/03_mutex/CMakeFiles/03_mutex.dir/build.make
03_mutex: libFreeRTOS.a
03_mutex: apps/common/libcommon_utils.a
03_mutex: apps/03_mutex/CMakeFiles/03_mutex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../../03_mutex"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/03_mutex && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/03_mutex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/03_mutex/CMakeFiles/03_mutex.dir/build: 03_mutex
.PHONY : apps/03_mutex/CMakeFiles/03_mutex.dir/build

apps/03_mutex/CMakeFiles/03_mutex.dir/clean:
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/03_mutex && $(CMAKE_COMMAND) -P CMakeFiles/03_mutex.dir/cmake_clean.cmake
.PHONY : apps/03_mutex/CMakeFiles/03_mutex.dir/clean

apps/03_mutex/CMakeFiles/03_mutex.dir/depend:
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/03_mutex /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/03_mutex /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/03_mutex/CMakeFiles/03_mutex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/03_mutex/CMakeFiles/03_mutex.dir/depend

