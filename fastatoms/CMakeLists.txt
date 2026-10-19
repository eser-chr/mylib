cmake_minimum_required(VERSION 3.18)
project(fastatoms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FASTATOMS_USE_OPENMP "Parse/format frames in parallel with OpenMP" ON)
option(FASTATOMS_USE_FAST_FLOAT "Use fast_float for number parsing (downloaded at configure time)" ON)
option(FASTATOMS_NATIVE "Compile with -march=native (not portable)" OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
  src/bindings.cpp
  src/atoms.cpp
  src/elements.cpp
  src/extxyz.cpp
)
target_include_directories(_core PRIVATE src)

if(FASTATOMS_USE_FAST_FLOAT)
  include(FetchContent)
  FetchContent_Declare(
    fast_float
    GIT_REPOSITORY https://github.com/fastfloat/fast_float.git
    GIT_TAG v6.1.6
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(fast_float)
  target_link_libraries(_core PRIVATE fast_float)
  target_compile_definitions(_core PRIVATE FASTATOMS_HAVE_FAST_FLOAT)
endif()

if(FASTATOMS_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE FASTATOMS_HAVE_OPENMP)
    message(STATUS "fastatoms: OpenMP enabled")
  else()
    message(WARNING "fastatoms: OpenMP not found, building single-threaded")
  endif()
endif()

if(MSVC)
  target_compile_options(_core PRIVATE /O2 /W3)
else()
  target_compile_options(_core PRIVATE -O3 -Wall -Wextra)
  if(FASTATOMS_NATIVE)
    target_compile_options(_core PRIVATE -march=native)
  endif()
endif()

install(TARGETS _core LIBRARY DESTINATION fastatoms)
