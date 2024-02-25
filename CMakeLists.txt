cmake_minimum_required (VERSION 3.8)
project ("pstringmatching" CXX)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (WIN32)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
  add_definitions(-DBOOST_ALL_NO_LIB=1)
else ()
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif ()

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

include_directories(csrc/include)
pybind11_add_module(pstringmatching csrc/main.cpp)
