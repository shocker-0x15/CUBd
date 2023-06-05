set(TARGET_NAME "libcubd")

set(
    SOURCES
    "../cubd.h"
    "../cubd.cu"
)

add_library(
    "${TARGET_NAME}" SHARED
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
target_compile_definitions("${TARGET_NAME}" PRIVATE CUBD_API_EXPORTS)
target_link_libraries(
    "${TARGET_NAME}"
    cubd_fakelib
    CUDA::cudart
)
