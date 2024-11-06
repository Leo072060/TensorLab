cmake_minimum_required(VERSION 3.2)
project(machineLearning)


add_definitions(-DDEBUG)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(ML ${SRC})