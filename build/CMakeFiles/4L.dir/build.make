# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dancoeks/Kuliah/DSEC/NN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dancoeks/Kuliah/DSEC/NN/build

# Include any dependencies generated for this target.
include CMakeFiles/4L.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/4L.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/4L.dir/flags.make

CMakeFiles/4L.dir/src/4L/main.cpp.o: CMakeFiles/4L.dir/flags.make
CMakeFiles/4L.dir/src/4L/main.cpp.o: ../src/4L/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dancoeks/Kuliah/DSEC/NN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/4L.dir/src/4L/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/4L.dir/src/4L/main.cpp.o -c /home/dancoeks/Kuliah/DSEC/NN/src/4L/main.cpp

CMakeFiles/4L.dir/src/4L/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/4L.dir/src/4L/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dancoeks/Kuliah/DSEC/NN/src/4L/main.cpp > CMakeFiles/4L.dir/src/4L/main.cpp.i

CMakeFiles/4L.dir/src/4L/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/4L.dir/src/4L/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dancoeks/Kuliah/DSEC/NN/src/4L/main.cpp -o CMakeFiles/4L.dir/src/4L/main.cpp.s

# Object files for target 4L
4L_OBJECTS = \
"CMakeFiles/4L.dir/src/4L/main.cpp.o"

# External object files for target 4L
4L_EXTERNAL_OBJECTS =

4L: CMakeFiles/4L.dir/src/4L/main.cpp.o
4L: CMakeFiles/4L.dir/build.make
4L: /usr/lib/x86_64-linux-gnu/libpython3.8.so
4L: CMakeFiles/4L.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dancoeks/Kuliah/DSEC/NN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 4L"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/4L.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/4L.dir/build: 4L

.PHONY : CMakeFiles/4L.dir/build

CMakeFiles/4L.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/4L.dir/cmake_clean.cmake
.PHONY : CMakeFiles/4L.dir/clean

CMakeFiles/4L.dir/depend:
	cd /home/dancoeks/Kuliah/DSEC/NN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dancoeks/Kuliah/DSEC/NN /home/dancoeks/Kuliah/DSEC/NN /home/dancoeks/Kuliah/DSEC/NN/build /home/dancoeks/Kuliah/DSEC/NN/build /home/dancoeks/Kuliah/DSEC/NN/build/CMakeFiles/4L.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/4L.dir/depend

