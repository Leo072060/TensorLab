# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_SOURCE_DIR = /home/twh/TensorLab

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/twh/TensorLab/test

# Include any dependencies generated for this target.
include CMakeFiles/ML.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ML.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ML.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ML.dir/flags.make

CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o: CMakeFiles/ML.dir/flags.make
CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o: ../src/ML/classificationEvaluation.cpp
CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o: CMakeFiles/ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o -MF CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o.d -o CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o -c /home/twh/TensorLab/src/ML/classificationEvaluation.cpp

CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/twh/TensorLab/src/ML/classificationEvaluation.cpp > CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.i

CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/twh/TensorLab/src/ML/classificationEvaluation.cpp -o CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.s

CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o: CMakeFiles/ML.dir/flags.make
CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o: ../src/ML/decisionTree.cpp
CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o: CMakeFiles/ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o -MF CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o.d -o CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o -c /home/twh/TensorLab/src/ML/decisionTree.cpp

CMakeFiles/ML.dir/src/ML/decisionTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML.dir/src/ML/decisionTree.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/twh/TensorLab/src/ML/decisionTree.cpp > CMakeFiles/ML.dir/src/ML/decisionTree.cpp.i

CMakeFiles/ML.dir/src/ML/decisionTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML.dir/src/ML/decisionTree.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/twh/TensorLab/src/ML/decisionTree.cpp -o CMakeFiles/ML.dir/src/ML/decisionTree.cpp.s

CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o: CMakeFiles/ML.dir/flags.make
CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o: ../src/ML/linearRegression.cpp
CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o: CMakeFiles/ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o -MF CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o.d -o CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o -c /home/twh/TensorLab/src/ML/linearRegression.cpp

CMakeFiles/ML.dir/src/ML/linearRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML.dir/src/ML/linearRegression.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/twh/TensorLab/src/ML/linearRegression.cpp > CMakeFiles/ML.dir/src/ML/linearRegression.cpp.i

CMakeFiles/ML.dir/src/ML/linearRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML.dir/src/ML/linearRegression.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/twh/TensorLab/src/ML/linearRegression.cpp -o CMakeFiles/ML.dir/src/ML/linearRegression.cpp.s

CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o: CMakeFiles/ML.dir/flags.make
CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o: ../src/ML/logisticRegression.cpp
CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o: CMakeFiles/ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o -MF CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o.d -o CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o -c /home/twh/TensorLab/src/ML/logisticRegression.cpp

CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/twh/TensorLab/src/ML/logisticRegression.cpp > CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.i

CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/twh/TensorLab/src/ML/logisticRegression.cpp -o CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.s

CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o: CMakeFiles/ML.dir/flags.make
CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o: ../src/ML/regressionEvalution.cpp
CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o: CMakeFiles/ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o -MF CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o.d -o CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o -c /home/twh/TensorLab/src/ML/regressionEvalution.cpp

CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/twh/TensorLab/src/ML/regressionEvalution.cpp > CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.i

CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/twh/TensorLab/src/ML/regressionEvalution.cpp -o CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.s

CMakeFiles/ML.dir/src/_internal/managed.cpp.o: CMakeFiles/ML.dir/flags.make
CMakeFiles/ML.dir/src/_internal/managed.cpp.o: ../src/_internal/managed.cpp
CMakeFiles/ML.dir/src/_internal/managed.cpp.o: CMakeFiles/ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/ML.dir/src/_internal/managed.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ML.dir/src/_internal/managed.cpp.o -MF CMakeFiles/ML.dir/src/_internal/managed.cpp.o.d -o CMakeFiles/ML.dir/src/_internal/managed.cpp.o -c /home/twh/TensorLab/src/_internal/managed.cpp

CMakeFiles/ML.dir/src/_internal/managed.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML.dir/src/_internal/managed.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/twh/TensorLab/src/_internal/managed.cpp > CMakeFiles/ML.dir/src/_internal/managed.cpp.i

CMakeFiles/ML.dir/src/_internal/managed.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML.dir/src/_internal/managed.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/twh/TensorLab/src/_internal/managed.cpp -o CMakeFiles/ML.dir/src/_internal/managed.cpp.s

CMakeFiles/ML.dir/src/main.cpp.o: CMakeFiles/ML.dir/flags.make
CMakeFiles/ML.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/ML.dir/src/main.cpp.o: CMakeFiles/ML.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/ML.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ML.dir/src/main.cpp.o -MF CMakeFiles/ML.dir/src/main.cpp.o.d -o CMakeFiles/ML.dir/src/main.cpp.o -c /home/twh/TensorLab/src/main.cpp

CMakeFiles/ML.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ML.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/twh/TensorLab/src/main.cpp > CMakeFiles/ML.dir/src/main.cpp.i

CMakeFiles/ML.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ML.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/twh/TensorLab/src/main.cpp -o CMakeFiles/ML.dir/src/main.cpp.s

# Object files for target ML
ML_OBJECTS = \
"CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o" \
"CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o" \
"CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o" \
"CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o" \
"CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o" \
"CMakeFiles/ML.dir/src/_internal/managed.cpp.o" \
"CMakeFiles/ML.dir/src/main.cpp.o"

# External object files for target ML
ML_EXTERNAL_OBJECTS =

ML: CMakeFiles/ML.dir/src/ML/classificationEvaluation.cpp.o
ML: CMakeFiles/ML.dir/src/ML/decisionTree.cpp.o
ML: CMakeFiles/ML.dir/src/ML/linearRegression.cpp.o
ML: CMakeFiles/ML.dir/src/ML/logisticRegression.cpp.o
ML: CMakeFiles/ML.dir/src/ML/regressionEvalution.cpp.o
ML: CMakeFiles/ML.dir/src/_internal/managed.cpp.o
ML: CMakeFiles/ML.dir/src/main.cpp.o
ML: CMakeFiles/ML.dir/build.make
ML: CMakeFiles/ML.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/twh/TensorLab/test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX executable ML"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ML.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ML.dir/build: ML
.PHONY : CMakeFiles/ML.dir/build

CMakeFiles/ML.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ML.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ML.dir/clean

CMakeFiles/ML.dir/depend:
	cd /home/twh/TensorLab/test && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/twh/TensorLab /home/twh/TensorLab /home/twh/TensorLab/test /home/twh/TensorLab/test /home/twh/TensorLab/test/CMakeFiles/ML.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ML.dir/depend

