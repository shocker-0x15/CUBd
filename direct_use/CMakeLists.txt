set(TARGET_NAME "direct_use")

set(
    SOURCES
    "../cubd.h"
    "../cubd.cu"
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
target_link_libraries(
    "${TARGET_NAME}"
    fakelib
    CUDA::cuda_driver
)
