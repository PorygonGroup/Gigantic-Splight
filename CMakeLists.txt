cmake_minimum_required(VERSION 3.16)
project(Gigantic_Splight)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SRC "src/*.cpp")

include_directories(include src)

set(CMAKE_CXX_FLAGS "-g -O0")

add_executable(splight ${SRC})
