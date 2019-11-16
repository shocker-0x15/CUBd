#include "cubd.h"

// Intellisense is broken if it reads this header.
#ifndef __INTELLISENSE__
#include "cub/cub.cuh"
#endif

// JP: CUB�̊֐����g�p���邽�߂ɁA���̃t�@�C����
//     NVCC�R���p�C���^�C�v"Generate hybrid object file (--compile)"
//     �Ƃ��ăR���p�C������K�v������B
// EN: This file needs to be compiled as 
//     NVCC compilation type "Generate hybrid object file (--compile)".
//     to use CUB functions.

// JP: �e���v���[�g�̖����I�C���X�^���X�����g���ĕK�v�Ȓ�`�𑫂��Ă��������B
// EN: Add necessary definitions using explicit template instanciation.

namespace cubd {
#define DEVICE_REDUCE_SUM_ARGUMENTS(InputIteratorT, OutputIteratorT) \
    void* d_temp_storage, size_t &temp_storage_bytes, \
    InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
    cudaStream_t stream, bool debug_synchronous

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DeviceReduce::Sum(DEVICE_REDUCE_SUM_ARGUMENTS(InputIteratorT, OutputIteratorT)) {
        return cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      d_in, d_out, num_items,
                                      stream, debug_synchronous);
    }

    template cudaError_t DeviceReduce::Sum(DEVICE_REDUCE_SUM_ARGUMENTS(int32_t*, int32_t*));
    template cudaError_t DeviceReduce::Sum(DEVICE_REDUCE_SUM_ARGUMENTS(const int32_t*, int32_t*));
    template cudaError_t DeviceReduce::Sum(DEVICE_REDUCE_SUM_ARGUMENTS(uint32_t*, uint32_t*));
    template cudaError_t DeviceReduce::Sum(DEVICE_REDUCE_SUM_ARGUMENTS(const uint32_t*, uint32_t*));
    template cudaError_t DeviceReduce::Sum(DEVICE_REDUCE_SUM_ARGUMENTS(float*, float*));
    template cudaError_t DeviceReduce::Sum(DEVICE_REDUCE_SUM_ARGUMENTS(const float*, float*));



#define DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(InputIteratorT, OutputIteratorT) \
    void* d_temp_storage, size_t &temp_storage_bytes, \
    InputIteratorT d_in, OutputIteratorT d_out, int num_items, \
    cudaStream_t stream, bool debug_synchronous

    template <typename InputIteratorT, typename OutputIteratorT>
    cudaError_t DeviceScan::ExclusiveSum(DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(InputIteratorT, OutputIteratorT)) {
        return cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                             d_in, d_out, num_items,
                                             stream, debug_synchronous);
    }

    template cudaError_t DeviceScan::ExclusiveSum(DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(int32_t*, int32_t*));
    template cudaError_t DeviceScan::ExclusiveSum(DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(const int32_t*, int32_t*));
    template cudaError_t DeviceScan::ExclusiveSum(DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(uint32_t*, uint32_t*));
    template cudaError_t DeviceScan::ExclusiveSum(DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(const uint32_t*, uint32_t*));
    template cudaError_t DeviceScan::ExclusiveSum(DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(float*, float*));
    template cudaError_t DeviceScan::ExclusiveSum(DEVICE_SCAN_EXCLUSIVE_SUM_ARGUMENTS(const float*, float*));



#define DEVICE_RADIX_SORT_SORT_PAIRS_ARGUMENTS(KeyT, ValueT) \
    void* d_temp_storage, size_t &temp_storage_bytes, \
    DoubleBuffer<KeyT> &d_keys, DoubleBuffer<ValueT> &d_values, int num_items, \
    int begin_bit, int end_bit, \
    cudaStream_t stream, bool debug_synchronous
    
    template <typename KeyT, typename ValueT>
    cudaError_t DeviceRadixSort::SortPairs(DEVICE_RADIX_SORT_SORT_PAIRS_ARGUMENTS(KeyT, ValueT)) {
        cub::DoubleBuffer<KeyT> cub_d_keys(d_keys.Current(), d_keys.Alternate());
        cub::DoubleBuffer<ValueT> cub_d_values(d_values.Current(), d_values.Alternate());
        return cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                               cub_d_keys, cub_d_values, num_items,
                                               begin_bit, end_bit,
                                               stream, debug_synchronous);
        d_keys.selector = cub_d_keys.Current() == d_keys.Alternate();
        d_values.selector = cub_d_values.Current() == d_values.Alternate();
    }

    // JP: RadixSort�̒l�ɂ͔C�ӂ̌^���g�p�\�B
    // EN: Value can be an arbitrary type.
    template cudaError_t DeviceRadixSort::SortPairs(DEVICE_RADIX_SORT_SORT_PAIRS_ARGUMENTS(uint64_t, uint32_t));
}
