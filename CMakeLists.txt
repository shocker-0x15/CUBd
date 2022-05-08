cmake_minimum_required(VERSION 3.18)
project(
    cubd_cmake
    VERSION 1.0
    DESCRIPTION "A simple library (example) to isolate CUB include"
    LANGUAGES C CXX CUDA)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(CUDAToolkit 11.6 REQUIRED)

option(CUBD_BUILD_SAMPLES "Build sample projects for cubd." ON)
option(CUBD_BUILD_DYNAMIC_LIB "Build dynamic-link cubd library." ON)
option(CUBD_BUILD_STATIC_LIB "Build static-link cubd library." ON)

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
