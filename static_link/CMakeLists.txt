set(TARGET_NAME "static_link")

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
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
target_compile_definitions("${TARGET_NAME}" PRIVATE USE_CUBD_LIB)
target_link_libraries(
    "${TARGET_NAME}"
    cubd_fakelib
    CUDA::cuda_driver
    libcubd_static
)
