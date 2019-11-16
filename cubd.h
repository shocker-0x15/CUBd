#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// JP: このヘッダーはCUBのヘッダーを見せていないため、
//     このヘッダーをincludeするファイルは通常のC++コードとしてコンパイルできる。
// EN: This header doesn't expose the CUB headers to make it possible to compile
//     files including this header as ordinary C++ code.

namespace cubd {
    template <typename T>
    struct DoubleBuffer {
        T* d_buffers[2];
        int selector;

        DoubleBuffer() {
            selector = 0;
            d_buffers[0] = NULL;
            d_buffers[1] = NULL;
        }
        DoubleBuffer(T* d_current, T* d_alternate) {
            selector = 0;
            d_buffers[0] = d_current;
            d_buffers[1] = d_alternate;
        }

        T* Current() { return d_buffers[selector]; }
        T* Alternate() { return d_buffers[selector ^ 1]; }
    };

    struct DeviceReduce {
        template <typename InputIteratorT, typename OutputIteratorT>
        static cudaError_t Sum(void* d_temp_storage, size_t &temp_storage_bytes,
                               InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                               cudaStream_t stream = 0, bool debug_synchronous = false);
    };

    struct DeviceScan {
        template <typename InputIteratorT, typename OutputIteratorT>
        static cudaError_t ExclusiveSum(void* d_temp_storage, size_t &temp_storage_bytes,
                                        InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                                        cudaStream_t stream = 0, bool debug_synchronous = false);
    };

    struct DeviceRadixSort {
        template <typename KeyT, typename ValueT>
        static cudaError_t SortPairs(void* d_temp_storage, size_t &temp_storage_bytes,
                                     DoubleBuffer<KeyT> &d_keys, DoubleBuffer<ValueT> &d_values, int num_items,
                                     int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                     cudaStream_t stream = 0, bool debug_synchronous = false);
    };
}
