#include "cubd.h"

#include <iterator>

// Intellisense is broken if it reads this header.
#ifndef __INTELLISENSE__
#include "cub/cub.cuh"
#endif

// JP: CUBの関数を使用するために、このファイルは
//     NVCCコンパイルタイプ"Generate hybrid object file (--compile)"
//     としてコンパイルする必要がある。
// EN: This file needs to be compiled as 
//     NVCC compilation type "Generate hybrid object file (--compile)".
//     to use CUB functions.

// JP: テンプレートの明示的インスタンス化を使って必要な定義を足してください。
// EN: Add necessary definitions using explicit template instanciation.

namespace cubd {
#define ARGXXX_KEY_VALUE_PAIR(ValueType) KeyValuePair<int32_t, ValueType>



#define DEVICE_REDUCE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::Sum(void* d_temp_storage, size_t &temp_storage_bytes, \
                      InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
                      cudaStream_t stream, bool debug_synchronous)

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, num_items,
                                      stream, debug_synchronous);
    }

    template cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(int32_t*, int32_t*);
    template cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(const int32_t*, int32_t*);
    template cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(uint32_t*, uint32_t*);
    template cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(const uint32_t*, uint32_t*);
    template cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(float*, float*);
    template cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(const float*, float*);



#define DEVICE_REDUCE_MIN_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::Min(void* d_temp_storage, size_t &temp_storage_bytes, \
                      InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
                      cudaStream_t stream, bool debug_synchronous)

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, num_items,
                                      stream, debug_synchronous);
    }

    template cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(int32_t*, int32_t*);
    template cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(const int32_t*, int32_t*);
    template cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(uint32_t*, uint32_t*);
    template cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(const uint32_t*, uint32_t*);
    template cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(float*, float*);
    template cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(const float*, float*);



#define DEVICE_REDUCE_ARGMIN_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::ArgMin(void* d_temp_storage, size_t &temp_storage_bytes, \
                         InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
                         cudaStream_t stream, bool debug_synchronous)

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_ARGMIN_SIGNATURE(InputIteratorT, OutputIteratorT) {
        typedef typename std::iterator_traits<OutputIteratorT>::value_type OutputValueType;
        typedef typename cub::KeyValuePair<typename OutputValueType::Key, typename OutputValueType::Value> CubPairType;
        static_assert(sizeof(OutputValueType) == sizeof(CubPairType),
                      "Sizes of KeyValuePair: Not match");
        CubPairType* cub_d_out = reinterpret_cast<CubPairType*>(d_out);
        return cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes,
                                         d_in, cub_d_out, num_items,
                                         stream, debug_synchronous);
    }

    template cudaError_t DEVICE_REDUCE_ARGMIN_SIGNATURE(int32_t*, ARGXXX_KEY_VALUE_PAIR(int32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMIN_SIGNATURE(const int32_t*, ARGXXX_KEY_VALUE_PAIR(int32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMIN_SIGNATURE(uint32_t*, ARGXXX_KEY_VALUE_PAIR(uint32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMIN_SIGNATURE(const uint32_t*, ARGXXX_KEY_VALUE_PAIR(uint32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMIN_SIGNATURE(float*, ARGXXX_KEY_VALUE_PAIR(float)*);
    template cudaError_t DEVICE_REDUCE_ARGMIN_SIGNATURE(const float*, ARGXXX_KEY_VALUE_PAIR(float)*);



#define DEVICE_REDUCE_MAX_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::Max(void* d_temp_storage, size_t &temp_storage_bytes, \
                      InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
                      cudaStream_t stream, bool debug_synchronous)

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, num_items,
                                      stream, debug_synchronous);
    }

    template cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(int32_t*, int32_t*);
    template cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(const int32_t*, int32_t*);
    template cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(uint32_t*, uint32_t*);
    template cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(const uint32_t*, uint32_t*);
    template cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(float*, float*);
    template cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(const float*, float*);



#define DEVICE_REDUCE_ARGMAX_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::ArgMax(void* d_temp_storage, size_t &temp_storage_bytes, \
                         InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
                         cudaStream_t stream, bool debug_synchronous)

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(InputIteratorT, OutputIteratorT) {
        typedef typename std::iterator_traits<OutputIteratorT>::value_type OutputValueType;
        typedef typename cub::KeyValuePair<typename OutputValueType::Key, typename OutputValueType::Value> cubPairType;
        cubPairType* cub_d_out = reinterpret_cast<cubPairType*>(d_out);
        return cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes,
                                         d_in, cub_d_out, num_items,
                                         stream, debug_synchronous);
    }

    template cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(int32_t*, ARGXXX_KEY_VALUE_PAIR(int32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(const int32_t*, ARGXXX_KEY_VALUE_PAIR(int32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(uint32_t*, ARGXXX_KEY_VALUE_PAIR(uint32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(const uint32_t*, ARGXXX_KEY_VALUE_PAIR(uint32_t)*);
    template cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(float*, ARGXXX_KEY_VALUE_PAIR(float)*);
    template cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(const float*, ARGXXX_KEY_VALUE_PAIR(float)*);



#define DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceScan::ExclusiveSum(void* d_temp_storage, size_t &temp_storage_bytes, \
                             InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
                             cudaStream_t stream, bool debug_synchronous)

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_in, d_out, num_items,
                                             stream, debug_synchronous);
    }

    template cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(int32_t*, int32_t*);
    template cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const int32_t*, int32_t*);
    template cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(uint32_t*, uint32_t*);
    template cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const uint32_t*, uint32_t*);
    template cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(float*, float*);
    template cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const float*, float*);



#define DEVICE_RADIX_SORT_SORT_PAIRS_SIGNATURE(KeyT, ValueT) \
    DeviceRadixSort::SortPairs(void* d_temp_storage, size_t &temp_storage_bytes, \
                               DoubleBuffer<KeyT> &d_keys, DoubleBuffer<ValueT> &d_values, int num_items, \
                               int begin_bit, int end_bit, \
                               cudaStream_t stream, bool debug_synchronous)
    
    template <typename KeyT, typename ValueT>
    cudaError_t DEVICE_RADIX_SORT_SORT_PAIRS_SIGNATURE(KeyT, ValueT) {
        static_assert(sizeof(cubd::DoubleBuffer<KeyT>) == sizeof(cub::DoubleBuffer<KeyT>) &&
                      sizeof(cubd::DoubleBuffer<ValueT>) == sizeof(cub::DoubleBuffer<ValueT>),
                      "Sizes of DoubleBuffer: Not match");
        cub::DoubleBuffer<KeyT> cub_d_keys = *reinterpret_cast<cub::DoubleBuffer<KeyT>*>(&d_keys);
        cub::DoubleBuffer<ValueT> cub_d_values = *reinterpret_cast<cub::DoubleBuffer<ValueT>*>(&d_values);
        cudaError_t res = cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                                          cub_d_keys, cub_d_values, num_items,
                                                          begin_bit, end_bit,
                                                          stream, debug_synchronous);
        d_keys = *reinterpret_cast<cubd::DoubleBuffer<KeyT>*>(&cub_d_keys);
        d_values = *reinterpret_cast<cubd::DoubleBuffer<ValueT>*>(&cub_d_values);
        return res;
    }

    // JP: RadixSortの値には任意の型が使用可能。
    // EN: Value can be an arbitrary type.
    template cudaError_t DEVICE_RADIX_SORT_SORT_PAIRS_SIGNATURE(uint64_t, uint32_t);
}
