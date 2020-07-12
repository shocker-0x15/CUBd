#include "cubd.h"

#include <iterator>

#include <cub/cub.cuh>

// JP: CUBの関数を使用するために、このファイルは
//     NVCCコンパイルタイプ"Generate hybrid object file (--compile)"
//     としてコンパイルする必要がある。
// EN: This file needs to be compiled as 
//     NVCC compilation type "Generate hybrid object file (--compile)".
//     to use CUB functions.

namespace cubd {
    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, num_items,
                                      stream, debug_synchronous);
    }



    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_MIN_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, num_items,
                                      stream, debug_synchronous);
    }



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



    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_MAX_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, num_items,
                                      stream, debug_synchronous);
    }



    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_REDUCE_ARGMAX_SIGNATURE(InputIteratorT, OutputIteratorT) {
        typedef typename std::iterator_traits<OutputIteratorT>::value_type OutputValueType;
        typedef typename cub::KeyValuePair<typename OutputValueType::Key, typename OutputValueType::Value> cubPairType;
        cubPairType* cub_d_out = reinterpret_cast<cubPairType*>(d_out);
        return cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes,
                                         d_in, cub_d_out, num_items,
                                         stream, debug_synchronous);
    }



    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_in, d_out, num_items,
                                             stream, debug_synchronous);
    }



    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        return cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_in, d_out, num_items,
                                             stream, debug_synchronous);
    }



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



    template <typename KeyT>
    cudaError_t DEVICE_RADIX_SORT_SORT_KEYS_SIGNATURE(KeyT) {
        static_assert(sizeof(cubd::DoubleBuffer<KeyT>) == sizeof(cub::DoubleBuffer<KeyT>),
                      "Sizes of DoubleBuffer: Not match");
        cub::DoubleBuffer<KeyT> cub_d_keys = *reinterpret_cast<cub::DoubleBuffer<KeyT>*>(&d_keys);
        cudaError_t res = cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
                                                         cub_d_keys, num_items,
                                                         begin_bit, end_bit,
                                                         stream, debug_synchronous);
        d_keys = *reinterpret_cast<cubd::DoubleBuffer<KeyT>*>(&cub_d_keys);
        return res;
    }
}
