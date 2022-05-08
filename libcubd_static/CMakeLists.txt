set(TARGET_NAME "libcubd_static")

set(
    SOURCES
    "../cubd.h"
    "../cubd.cu"
)

add_library(
    "${TARGET_NAME}" STATIC
    ${SOURCES}
)
target_compile_features("${TARGET_NAME}" PRIVATE cxx_std_20)
set_target_properties("${TARGET_NAME}" PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(
    "${TARGET_NAME}"
    cubd_fakelib
    CUDA::cudart
)
