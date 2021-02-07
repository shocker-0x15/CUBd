/*

   Copyright 2021 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

#include "cubd.h"

#include <cuda_runtime.h>
#include <iterator>

#if !defined(__INTELLISENSE__)
#include <cub/cub.cuh>
#endif

// JP: CUBの関数を使用するために、このファイルは
//     NVCCコンパイルタイプ"Generate hybrid object file (--compile)"
//     としてコンパイルする必要がある。
// EN: This file needs to be compiled as 
//     NVCC compilation type "Generate hybrid object file (--compile)".
//     to use CUB functions.

namespace cubd {
    template <typename CubdKeyValuePair>
    using CubKeyValuePairType = cub::KeyValuePair<typename CubdKeyValuePair::Key, typename CubdKeyValuePair::Value>;



    static inline CUresult cudaError_t_to_CUresult(cudaError_t cudaError) {
        // JP: Driver APIとRuntime APIのエラーコード間で対応関係のあるものは同じ数値が設定されているようなので、
        //     エラーコードの変換は直接の代入に頼る。
        // EN: Converting the error code relies on direct substitution because
        //     corresponding error codes in Driver API and Runtime API seem to have the same value.
        return static_cast<CUresult>(cudaError);
    }



    template <typename InputIteratorT, typename OutputIteratorT>
    CUresult DEVICE_REDUCE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        cudaError_t cudaError = cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items,
            stream, debug_synchronous);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    CUresult DEVICE_REDUCE_MIN_SIGNATURE(InputIteratorT, OutputIteratorT) {
        cudaError_t cudaError = cub::DeviceReduce::Min(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items,
            stream, debug_synchronous);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(InputIteratorT, OutputIteratorT) {
        using CubdKeyValuePair = typename std::iterator_traits<OutputIteratorT>::value_type;
        using CubKeyValuePair = CubKeyValuePairType<CubdKeyValuePair>;
        static_assert(sizeof(CubdKeyValuePair) == sizeof(CubKeyValuePair),
                      "Sizes of KeyValuePair: Not match");

        auto cub_d_out = reinterpret_cast<CubKeyValuePair*>(d_out);
        cudaError_t cudaError = cub::DeviceReduce::ArgMin(
            d_temp_storage, temp_storage_bytes,
            d_in, cub_d_out, num_items,
            stream, debug_synchronous);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    CUresult DEVICE_REDUCE_MAX_SIGNATURE(InputIteratorT, OutputIteratorT) {
        cudaError_t cudaError = cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items,
            stream, debug_synchronous);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(InputIteratorT, OutputIteratorT) {
        using CubdKeyValuePair = typename std::iterator_traits<OutputIteratorT>::value_type;
        using CubKeyValuePair = CubKeyValuePairType<CubdKeyValuePair>;
        static_assert(sizeof(CubdKeyValuePair) == sizeof(CubKeyValuePair),
                      "Sizes of KeyValuePair: Not match");

        auto cub_d_out = reinterpret_cast<CubKeyValuePair*>(d_out);
        cudaError_t cudaError = cub::DeviceReduce::ArgMax(
            d_temp_storage, temp_storage_bytes,
            d_in, cub_d_out, num_items,
            stream, debug_synchronous);
        return cudaError_t_to_CUresult(cudaError);
    }



    template <typename InputIteratorT, typename OutputIteratorT>
    CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        cudaError_t cudaError = cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items,
            stream, debug_synchronous);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename InputIteratorT, typename OutputIteratorT>
    CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) {
        cudaError_t cudaError = cub::DeviceScan::InclusiveSum(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items,
            stream, debug_synchronous);
        return cudaError_t_to_CUresult(cudaError);
    }



    template <typename KeyT, typename ValueT>
    CUresult DEVICE_RADIX_SORT_SORT_PAIRS_SIGNATURE(KeyT, ValueT) {
        static_assert(sizeof(cubd::DoubleBuffer<KeyT>) == sizeof(cub::DoubleBuffer<KeyT>) &&
                      sizeof(cubd::DoubleBuffer<ValueT>) == sizeof(cub::DoubleBuffer<ValueT>),
                      "Sizes of DoubleBuffer: Not match");

        auto cub_d_keys = reinterpret_cast<cub::DoubleBuffer<KeyT> &>(d_keys);
        auto cub_d_values = reinterpret_cast<cub::DoubleBuffer<ValueT> &>(d_values);
        cudaError_t cudaError = cub::DeviceRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            cub_d_keys, cub_d_values, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous);
        d_keys = reinterpret_cast<cubd::DoubleBuffer<KeyT> &>(cub_d_keys);
        d_values = reinterpret_cast<cubd::DoubleBuffer<ValueT> &>(cub_d_values);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename KeyT, typename ValueT>
    CUresult DEVICE_RADIX_SORT_SORT_PAIRS_DESCENDING_SIGNATURE(KeyT, ValueT) {
        static_assert(sizeof(cubd::DoubleBuffer<KeyT>) == sizeof(cub::DoubleBuffer<KeyT>) &&
                      sizeof(cubd::DoubleBuffer<ValueT>) == sizeof(cub::DoubleBuffer<ValueT>),
                      "Sizes of DoubleBuffer: Not match");

        auto cub_d_keys = reinterpret_cast<cub::DoubleBuffer<KeyT> &>(d_keys);
        auto cub_d_values = reinterpret_cast<cub::DoubleBuffer<ValueT> &>(d_values);
        cudaError_t cudaError = cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            cub_d_keys, cub_d_values, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous);
        d_keys = reinterpret_cast<cubd::DoubleBuffer<KeyT> &>(cub_d_keys);
        d_values = reinterpret_cast<cubd::DoubleBuffer<ValueT> &>(cub_d_values);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename KeyT>
    CUresult DEVICE_RADIX_SORT_SORT_KEYS_SIGNATURE(KeyT) {
        static_assert(sizeof(cubd::DoubleBuffer<KeyT>) == sizeof(cub::DoubleBuffer<KeyT>),
                      "Sizes of DoubleBuffer: Not match");

        auto cub_d_keys = reinterpret_cast<cub::DoubleBuffer<KeyT> &>(d_keys);
        cudaError_t cudaError = cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            cub_d_keys, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous);
        d_keys = reinterpret_cast<cubd::DoubleBuffer<KeyT> &>(cub_d_keys);
        return cudaError_t_to_CUresult(cudaError);
    }

    template <typename KeyT>
    CUresult DEVICE_RADIX_SORT_SORT_KEYS_DESCENDING_SIGNATURE(KeyT) {
        static_assert(sizeof(cubd::DoubleBuffer<KeyT>) == sizeof(cub::DoubleBuffer<KeyT>),
                      "Sizes of DoubleBuffer: Not match");

        auto cub_d_keys = reinterpret_cast<cub::DoubleBuffer<KeyT> &>(d_keys);
        cudaError_t cudaError = cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            cub_d_keys, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous);
        d_keys = reinterpret_cast<cubd::DoubleBuffer<KeyT> &>(cub_d_keys);
        return cudaError_t_to_CUresult(cudaError);
    }
}
