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
include Final_Gen/CMakeFiles/GEN.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Final_Gen/CMakeFiles/GEN.dir/compiler_depend.make

# Include the progress variables for this target.
include Final_Gen/CMakeFiles/GEN.dir/progress.make

# Include the compile flags for this target's objects.
include Final_Gen/CMakeFiles/GEN.dir/flags.make

Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.o: Final_Gen/CMakeFiles/GEN.dir/flags.make
Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.o: ../Final_Gen/Gen.cpp
Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.o: Final_Gen/CMakeFiles/GEN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.o"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/Final_Gen && /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.o -MF CMakeFiles/GEN.dir/Gen.cpp.o.d -o CMakeFiles/GEN.dir/Gen.cpp.o -c /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/Final_Gen/Gen.cpp

Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GEN.dir/Gen.cpp.i"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/Final_Gen && /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/Final_Gen/Gen.cpp > CMakeFiles/GEN.dir/Gen.cpp.i

Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GEN.dir/Gen.cpp.s"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/Final_Gen && /mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-x86_64_v3/gcc-4.8.5/gcc-8.5.0-r653aprkmt2xeqxzz4locto6dy3hxl47/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/Final_Gen/Gen.cpp -o CMakeFiles/GEN.dir/Gen.cpp.s

# Object files for target GEN
GEN_OBJECTS = \
"CMakeFiles/GEN.dir/Gen.cpp.o"

# External object files for target GEN
GEN_EXTERNAL_OBJECTS =

Final_Gen/libGEN.so: Final_Gen/CMakeFiles/GEN.dir/Gen.cpp.o
Final_Gen/libGEN.so: Final_Gen/CMakeFiles/GEN.dir/build.make
Final_Gen/libGEN.so: Final_Gen/CMakeFiles/GEN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module libGEN.so"
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/Final_Gen && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GEN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Final_Gen/CMakeFiles/GEN.dir/build: Final_Gen/libGEN.so
.PHONY : Final_Gen/CMakeFiles/GEN.dir/build

Final_Gen/CMakeFiles/GEN.dir/clean:
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/Final_Gen && $(CMAKE_COMMAND) -P CMakeFiles/GEN.dir/cmake_clean.cmake
.PHONY : Final_Gen/CMakeFiles/GEN.dir/clean

Final_Gen/CMakeFiles/GEN.dir/depend:
	cd /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/Final_Gen /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/Final_Gen /mnt/sdb1/home/mzw/HIPCC_LLVM_4.3.1/include/llvm/Transforms/build/Final_Gen/CMakeFiles/GEN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Final_Gen/CMakeFiles/GEN.dir/depend
