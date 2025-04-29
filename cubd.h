/*

   Copyright 2025 Shin Watanabe

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

#pragma once

// JP: このヘッダーはCUBのヘッダーを見せていないため、
//     このヘッダーをincludeするファイルは通常のC++コードとしてコンパイルできる。
//     静的リンクする際は"USE_CUBD_LIB"、動的リンクする場合は"USE_CUBD_DLL"の定義が必要。
// EN: This header doesn't expose the CUB headers to make it possible to compile
//     files including this header as ordinary C++ code.
//     It is necessarry to define "USE_CUBD_LIB" for static linking and "USE_CUBD_DLL" for dynamic linking.

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define CUBD_API_Platform_Windows
#    if defined(_MSC_VER)
#        define CUBD_API_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define CUBD_API_Platform_macOS
#endif

#if defined(CUBD_API_Platform_Windows_MSVC)
#   if defined(CUBD_API_EXPORTS)
#       define CUBD_EXTERN
#       define CUBD_API __declspec(dllexport)
#   elif defined(USE_CUBD_DLL)
#       define CUBD_EXTERN extern
#       define CUBD_API __declspec(dllimport)
#   elif defined(USE_CUBD_LIB)
#       define CUBD_EXTERN extern
#       define CUBD_API
#   else
#       define CUBD_EXTERN
#       define CUBD_API
#   endif
#else
#   define CUBD_EXTERN
#   define CUBD_API
#endif

#include <cstdint>
#include <cuda.h>

// JP: テンプレートの明示的インスタンス化を使って必要な定義を足してください。
// EN: Add necessary definitions using explicit template instantiation.

namespace cubd {
    template <typename _Key, typename _Value>
    struct CUBD_API KeyValuePair {
        using Key = _Key;
        using Value = _Value;

        Key key;
        Value value;
    };

    template <typename _Value>
    using ArgXxxKeyValuePair = KeyValuePair<int32_t, _Value>;



    template <typename T>
    struct CUBD_API DoubleBuffer {
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

    CUBD_EXTERN template struct CUBD_API DoubleBuffer<uint32_t>;
    CUBD_EXTERN template struct CUBD_API DoubleBuffer<uint64_t>;



    struct CUBD_API DeviceReduce {
        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult Sum(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);

        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult Min(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);

        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult ArgMin(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);

        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult Max(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);

        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult ArgMax(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);
    };

#define DEVICE_REDUCE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::Sum( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(const int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(const uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(const float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(const int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(uint64_t*, uint64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_SUM_SIGNATURE(const uint64_t*, uint64_t*);

#define DEVICE_REDUCE_MIN_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::Min( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(const int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(const uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(const float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(const int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(uint64_t*, uint64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MIN_SIGNATURE(const uint64_t*, uint64_t*);

#define DEVICE_REDUCE_ARGMIN_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::ArgMin( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(int32_t*, ArgXxxKeyValuePair<int32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(const int32_t*, ArgXxxKeyValuePair<int32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(uint32_t*, ArgXxxKeyValuePair<uint32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(const uint32_t*, ArgXxxKeyValuePair<uint32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(float*, ArgXxxKeyValuePair<float>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(const float*, ArgXxxKeyValuePair<float>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(int64_t*, ArgXxxKeyValuePair<int64_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(const int64_t*, ArgXxxKeyValuePair<int64_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(uint64_t*, ArgXxxKeyValuePair<uint64_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMIN_SIGNATURE(const uint64_t*, ArgXxxKeyValuePair<uint64_t>*);

#define DEVICE_REDUCE_MAX_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::Max( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(const int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(const uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(const float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(const int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(uint64_t*, uint64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_MAX_SIGNATURE(const uint64_t*, uint64_t*);

#define DEVICE_REDUCE_ARGMAX_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceReduce::ArgMax( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(int32_t*, ArgXxxKeyValuePair<int32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(const int32_t*, ArgXxxKeyValuePair<int32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(uint32_t*, ArgXxxKeyValuePair<uint32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(const uint32_t*, ArgXxxKeyValuePair<uint32_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(float*, ArgXxxKeyValuePair<float>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(const float*, ArgXxxKeyValuePair<float>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(int64_t*, ArgXxxKeyValuePair<int64_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(const int64_t*, ArgXxxKeyValuePair<int64_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(uint64_t*, ArgXxxKeyValuePair<uint64_t>*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_REDUCE_ARGMAX_SIGNATURE(const uint64_t*, ArgXxxKeyValuePair<uint64_t>*);



    struct CUBD_API DeviceScan {
        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult ExclusiveSum(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);

        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult InclusiveSum(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);

        template <typename InputIteratorT, typename OutputIteratorT, typename InitValueT>
        static CUresult ExclusiveMax(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, InitValueT init_value, int num_items,
            CUstream stream = 0);

        template <typename InputIteratorT, typename OutputIteratorT>
        static CUresult InclusiveMax(
            void* d_temp_storage, size_t &temp_storage_bytes,
            InputIteratorT d_in, OutputIteratorT d_out, int num_items,
            CUstream stream = 0);
    };

#define DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceScan::ExclusiveSum( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(uint64_t*, uint64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_SUM_SIGNATURE(const uint64_t*, uint64_t*);

#define DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceScan::InclusiveSum( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(const int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(const uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(const float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(const int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(uint64_t*, uint64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_SUM_SIGNATURE(const uint64_t*, uint64_t*);

#define DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(InputIteratorT, OutputIteratorT, InitValueT) \
    DeviceScan::ExclusiveMax( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, InitValueT init_value, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(int32_t*, int32_t*, int32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(const int32_t*, int32_t*, int32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(uint32_t*, uint32_t*, uint32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(const uint32_t*, uint32_t*, uint32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(float*, float*, float);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(const float*, float*, float);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(int64_t*, int64_t*, int64_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(const int64_t*, int64_t*, int64_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(uint64_t*, uint64_t*, uint64_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_EXCLUSIVE_MAX_SIGNATURE(const uint64_t*, uint64_t*, uint64_t);

#define DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(InputIteratorT, OutputIteratorT) \
    DeviceScan::InclusiveMax( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(const int32_t*, int32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(const uint32_t*, uint32_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(const float*, float*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(const int64_t*, int64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(uint64_t*, uint64_t*);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_SCAN_INCLUSIVE_MAX_SIGNATURE(const uint64_t*, uint64_t*);



    struct CUBD_API DeviceRadixSort {
        template <typename KeyT, typename ValueT>
        static CUresult SortPairs(
            void* d_temp_storage, size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys, DoubleBuffer<ValueT> &d_values, int num_items,
            int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
            CUstream stream = 0);

        template <typename KeyT, typename ValueT>
        static CUresult SortPairsDescending(
            void* d_temp_storage, size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys, DoubleBuffer<ValueT> &d_values, int num_items,
            int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
            CUstream stream = 0);

        template <typename KeyT>
        static CUresult SortKeys(
            void* d_temp_storage, size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys, int num_items,
            int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
            CUstream stream = 0);

        template <typename KeyT>
        static CUresult SortKeysDescending(
            void* d_temp_storage, size_t &temp_storage_bytes,
            DoubleBuffer<KeyT> &d_keys, int num_items,
            int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
            CUstream stream = 0);
    };

#define DEVICE_RADIX_SORT_SORT_PAIRS_SIGNATURE(KeyT, ValueT) \
    DeviceRadixSort::SortPairs( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        DoubleBuffer<KeyT> &d_keys, DoubleBuffer<ValueT> &d_values, int num_items, \
        int begin_bit, int end_bit, \
        CUstream stream)

    // JP: RadixSortの値として任意の型を定義可能。
    // EN: It is possible to define an arbitrary type as value for radix sort.
    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_PAIRS_SIGNATURE(uint32_t, uint32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_PAIRS_SIGNATURE(uint64_t, uint32_t);

#define DEVICE_RADIX_SORT_SORT_PAIRS_DESCENDING_SIGNATURE(KeyT, ValueT) \
    DeviceRadixSort::SortPairsDescending( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        DoubleBuffer<KeyT> &d_keys, DoubleBuffer<ValueT> &d_values, int num_items, \
        int begin_bit, int end_bit, \
        CUstream stream)

    // JP: RadixSortの値として任意の型を定義可能。
    // EN: It is possible to define an arbitrary type as value for radix sort.
    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_PAIRS_DESCENDING_SIGNATURE(uint32_t, uint32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_PAIRS_DESCENDING_SIGNATURE(uint64_t, uint32_t);

#define DEVICE_RADIX_SORT_SORT_KEYS_SIGNATURE(KeyT) \
    DeviceRadixSort::SortKeys( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        DoubleBuffer<KeyT> &d_keys, int num_items, \
        int begin_bit, int end_bit, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_KEYS_SIGNATURE(uint32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_KEYS_SIGNATURE(uint64_t);

#define DEVICE_RADIX_SORT_SORT_KEYS_DESCENDING_SIGNATURE(KeyT) \
    DeviceRadixSort::SortKeysDescending( \
        void* d_temp_storage, size_t &temp_storage_bytes, \
        DoubleBuffer<KeyT> &d_keys, int num_items, \
        int begin_bit, int end_bit, \
        CUstream stream)

    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_KEYS_DESCENDING_SIGNATURE(uint32_t);
    CUBD_EXTERN template CUBD_API CUresult DEVICE_RADIX_SORT_SORT_KEYS_DESCENDING_SIGNATURE(uint64_t);
}
