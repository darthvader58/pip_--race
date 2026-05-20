cmake_minimum_required(VERSION 3.16)
project(pip_race_native LANGUAGES CXX)

add_library(pip_race_native SHARED pip_race_native.cpp)
target_compile_features(pip_race_native PRIVATE cxx_std_17)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_options(pip_race_native PRIVATE -O3 -march=native -ffast-math -fvisibility=hidden)
endif()

set_target_properties(
  pip_race_native
  PROPERTIES
    OUTPUT_NAME "pip_race_native"
    POSITION_INDEPENDENT_CODE ON
)
