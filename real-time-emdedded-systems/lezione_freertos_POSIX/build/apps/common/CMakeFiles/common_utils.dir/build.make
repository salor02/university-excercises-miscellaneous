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
include apps/common/CMakeFiles/common_utils.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include apps/common/CMakeFiles/common_utils.dir/compiler_depend.make

# Include the progress variables for this target.
include apps/common/CMakeFiles/common_utils.dir/progress.make

# Include the compile flags for this target's objects.
include apps/common/CMakeFiles/common_utils.dir/flags.make

apps/common/CMakeFiles/common_utils.dir/utils.c.o: apps/common/CMakeFiles/common_utils.dir/flags.make
apps/common/CMakeFiles/common_utils.dir/utils.c.o: /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/common/utils.c
apps/common/CMakeFiles/common_utils.dir/utils.c.o: apps/common/CMakeFiles/common_utils.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object apps/common/CMakeFiles/common_utils.dir/utils.c.o"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT apps/common/CMakeFiles/common_utils.dir/utils.c.o -MF CMakeFiles/common_utils.dir/utils.c.o.d -o CMakeFiles/common_utils.dir/utils.c.o -c /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/common/utils.c

apps/common/CMakeFiles/common_utils.dir/utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/common_utils.dir/utils.c.i"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/common/utils.c > CMakeFiles/common_utils.dir/utils.c.i

apps/common/CMakeFiles/common_utils.dir/utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/common_utils.dir/utils.c.s"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/common/utils.c -o CMakeFiles/common_utils.dir/utils.c.s

# Object files for target common_utils
common_utils_OBJECTS = \
"CMakeFiles/common_utils.dir/utils.c.o"

# External object files for target common_utils
common_utils_EXTERNAL_OBJECTS =

apps/common/libcommon_utils.a: apps/common/CMakeFiles/common_utils.dir/utils.c.o
apps/common/libcommon_utils.a: apps/common/CMakeFiles/common_utils.dir/build.make
apps/common/libcommon_utils.a: apps/common/CMakeFiles/common_utils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libcommon_utils.a"
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common && $(CMAKE_COMMAND) -P CMakeFiles/common_utils.dir/cmake_clean_target.cmake
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/common_utils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/common/CMakeFiles/common_utils.dir/build: apps/common/libcommon_utils.a
.PHONY : apps/common/CMakeFiles/common_utils.dir/build

apps/common/CMakeFiles/common_utils.dir/clean:
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common && $(CMAKE_COMMAND) -P CMakeFiles/common_utils.dir/cmake_clean.cmake
.PHONY : apps/common/CMakeFiles/common_utils.dir/clean

apps/common/CMakeFiles/common_utils.dir/depend:
	cd /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/apps/common /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common /data/Documents/university-excercises-miscellaneous/real-time-emdedded-systems/RTES_freertos_POSIX/build/apps/common/CMakeFiles/common_utils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/common/CMakeFiles/common_utils.dir/depend

