# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_COMMAND = /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/cmake-3.21.4-nvydniyy3o7epsmkovhbb2r2gabeijca/bin/cmake

# The command to remove a file.
RM = /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/cmake-3.21.4-nvydniyy3o7epsmkovhbb2r2gabeijca/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build

# Include any dependencies generated for this target.
include data_flow/CMakeFiles/flow.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include data_flow/CMakeFiles/flow.dir/compiler_depend.make

# Include the progress variables for this target.
include data_flow/CMakeFiles/flow.dir/progress.make

# Include the compile flags for this target's objects.
include data_flow/CMakeFiles/flow.dir/flags.make

data_flow/CMakeFiles/flow.dir/flow.cpp.o: data_flow/CMakeFiles/flow.dir/flags.make
data_flow/CMakeFiles/flow.dir/flow.cpp.o: ../data_flow/flow.cpp
data_flow/CMakeFiles/flow.dir/flow.cpp.o: data_flow/CMakeFiles/flow.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object data_flow/CMakeFiles/flow.dir/flow.cpp.o"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/data_flow && /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT data_flow/CMakeFiles/flow.dir/flow.cpp.o -MF CMakeFiles/flow.dir/flow.cpp.o.d -o CMakeFiles/flow.dir/flow.cpp.o -c /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/data_flow/flow.cpp

data_flow/CMakeFiles/flow.dir/flow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/flow.dir/flow.cpp.i"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/data_flow && /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/data_flow/flow.cpp > CMakeFiles/flow.dir/flow.cpp.i

data_flow/CMakeFiles/flow.dir/flow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/flow.dir/flow.cpp.s"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/data_flow && /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/data_flow/flow.cpp -o CMakeFiles/flow.dir/flow.cpp.s

# Object files for target flow
flow_OBJECTS = \
"CMakeFiles/flow.dir/flow.cpp.o"

# External object files for target flow
flow_EXTERNAL_OBJECTS =

data_flow/libflow.so: data_flow/CMakeFiles/flow.dir/flow.cpp.o
data_flow/libflow.so: data_flow/CMakeFiles/flow.dir/build.make
data_flow/libflow.so: data_flow/CMakeFiles/flow.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module libflow.so"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/data_flow && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/flow.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
data_flow/CMakeFiles/flow.dir/build: data_flow/libflow.so
.PHONY : data_flow/CMakeFiles/flow.dir/build

data_flow/CMakeFiles/flow.dir/clean:
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/data_flow && $(CMAKE_COMMAND) -P CMakeFiles/flow.dir/cmake_clean.cmake
.PHONY : data_flow/CMakeFiles/flow.dir/clean

data_flow/CMakeFiles/flow.dir/depend:
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/data_flow /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/data_flow /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/data_flow/CMakeFiles/flow.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : data_flow/CMakeFiles/flow.dir/depend

