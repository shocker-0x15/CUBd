cmake_minimum_required(VERSION 3.18)
project(
    cubd_cmake
    VERSION 1.0
    DESCRIPTION "A simple library (example) to isolate CUB include"
    LANGUAGES C CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(CUDAToolkit 12.0 REQUIRED)

option(CUBD_BUILD_SAMPLES "Build sample projects for cubd." ON)
option(CUBD_BUILD_DYNAMIC_LIB "Build dynamic-link cubd library." ON)
option(CUBD_BUILD_STATIC_LIB "Build static-link cubd library." ON)

# c++20 has a compilation issue with VS 2022 17.10.0 and CUDA 12.5.
set(CUBD_CPP_VER "c++20" CACHE STRING "C++ version the host code uses")
set_property(
    CACHE CUBD_CPP_VER PROPERTY STRINGS
    "c++17" "c++20")

add_library(cubd_fakelib INTERFACE)
target_compile_definitions(
    cubd_fakelib INTERFACE
    "$<$<CONFIG:Debug>:_DEBUG=1>"
)
target_compile_options(
    cubd_fakelib INTERFACE
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/MP);
    "$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>"
    # if (compilerID == MSVC && compilerLanguage != CUDA) set(/Zc:__cplusplus);
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/Zc:__cplusplus>"
    # if (compilerID == MSVC && compilerLanguage == CUDA) set(-Xcompiler "/wd 4819 /Zc:__cplusplus");
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/wd4819 /Zc:__cplusplus>"
)

if(CUBD_BUILD_DYNAMIC_LIB)
    add_subdirectory(libcubd)
endif()
if(CUBD_BUILD_STATIC_LIB)
    add_subdirectory(libcubd_static)
endif()
if (CUBD_BUILD_SAMPLES)
    add_subdirectory(direct_use)
    if(CUBD_BUILD_DYNAMIC_LIB)
        add_subdirectory(dynamic_link)
    endif()
    if(CUBD_BUILD_STATIC_LIB)
        add_subdirectory(static_link)
    endif()
endif()
