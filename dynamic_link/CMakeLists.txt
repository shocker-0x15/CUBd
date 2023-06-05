set(TARGET_NAME "dynamic_link")

set(
    SOURCES
    "../cubd.h"
    "../cuda_util.h"
    "../cuda_util.cpp"
    "../main.cpp"
)

add_executable(
    "${TARGET_NAME}"
    ${SOURCES}
)
if(${CUBD_CPP_VER} STREQUAL "c++17")
    target_compile_features(
        "${TARGET_NAME}" PRIVATE
        cxx_std_17
    )
elseif(${CUBD_CPP_VER} STREQUAL "c++20")
    target_compile_features(
        "${TARGET_NAME}" PRIVATE
        cxx_std_20
    )
else()
    message(FATAL_ERROR "Invalid C++ standard.")
endif()
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
target_compile_definitions("${TARGET_NAME}" PRIVATE USE_CUBD_DLL)
target_compile_definitions(
    "${TARGET_NAME}" PRIVATE
    "CUDA_UTIL_DONT_USE_GL_INTEROP"
)
target_link_libraries(
    "${TARGET_NAME}"
    cubd_fakelib
    CUDA::cuda_driver
    libcubd
)
