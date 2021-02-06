#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <exception>
#include <random>
#include <limits>

#include "cubd.h"
#include "cuda_util.h"

// JP: このファイルは通常のC++コードとしてコンパイルできる。
// EN: This file can be compiled as an ordinary C++ code.

#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

static bool test_sum_int32_t();
static bool test_sum_uint32_t();
static bool test_sum_float();
static bool test_min_int32_t();
static bool test_min_uint32_t();
static bool test_min_float();
static bool test_max_int32_t();
static bool test_max_uint32_t();
static bool test_max_float();
static bool test_argmin_int32_t();
static bool test_argmin_uint32_t();
static bool test_argmin_float();
static bool test_argmax_int32_t();
static bool test_argmax_uint32_t();
static bool test_argmax_float();
static bool test_exclusive_sum_int32_t();
static bool test_exclusive_sum_uint32_t();
static bool test_exclusive_sum_float();
static bool test_inclusive_sum_int32_t();
static bool test_inclusive_sum_uint32_t();
static bool test_inclusive_sum_float();
static bool test_radix_sort_uint64_t_key_uint32_t_value();
static bool test_radix_sort_uint64_t_key();

static CUcontext cuContext;
static CUstream cuStream;
static cudau::BufferType bufferType = cudau::BufferType::Device;

int32_t main(int32_t argc, const char* argv[]) {
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    std::mt19937 rng(194712984);

    printf("Start tests.\n");

    bool success = true;

    success &= test_sum_int32_t();
    success &= test_sum_uint32_t();
    success &= test_sum_float();

    success &= test_min_int32_t();
    success &= test_min_uint32_t();
    success &= test_min_float();

    success &= test_max_int32_t();
    success &= test_max_uint32_t();
    success &= test_max_float();

    success &= test_argmin_int32_t();
    success &= test_argmin_uint32_t();
    success &= test_argmin_float();

    success &= test_argmax_int32_t();
    success &= test_argmax_uint32_t();
    success &= test_argmax_float();

    success &= test_exclusive_sum_int32_t();
    success &= test_exclusive_sum_uint32_t();
    success &= test_exclusive_sum_float();

    success &= test_inclusive_sum_int32_t();
    success &= test_inclusive_sum_uint32_t();
    success &= test_inclusive_sum_float();

    success &= test_radix_sort_uint64_t_key_uint32_t_value();

    success &= test_radix_sort_uint64_t_key();

    if (success)
        printf("All Success!\n");
    else
        printf("Something went wrong...\n");

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}



std::mt19937_64 rng(194712984);

constexpr uint32_t NumTests = 10;

static bool test_sum_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-100, 100);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> sum;
    sum.initialize(cuContext, bufferType, 1);
    sum.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            values.getDevicePointer(), sum.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Sum, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refSum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        values.unmap();

        sum.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), sum.getDevicePointer(), numElements);

        ValueType sumOnHost;
        sum.read(&sumOnHost, 1);

        printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, sumOnHost, refSum,
               sumOnHost == refSum ? "" : " NG");

        allSuccess &= sumOnHost == refSum;
    }
    printf("\n");

    tempStorage.finalize();
    sum.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_sum_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 100);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> sum;
    sum.initialize(cuContext, bufferType, 1);
    sum.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            values.getDevicePointer(), sum.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Sum, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refSum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        values.unmap();

        sum.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), sum.getDevicePointer(), numElements);

        ValueType sumOnHost;
        sum.read(&sumOnHost, 1);

        printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, sumOnHost, refSum,
               sumOnHost == refSum ? "" : " NG");

        allSuccess &= sumOnHost == refSum;
    }
    printf("\n");

    tempStorage.finalize();
    sum.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_sum_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> sum;
    sum.initialize(cuContext, bufferType, 1);
    sum.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            values.getDevicePointer(), sum.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Sum, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        double refSum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        values.unmap();

        sum.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), sum.getDevicePointer(), numElements);

        ValueType sumOnHost;
        sum.read(&sumOnHost, 1);

        ValueType error = (sumOnHost - refSum) / refSum;
        bool success = std::fabs(error) < 0.001f;
        printf("  N: %5u, %g (ref: %g), error: %.2f%%%s\n", numElements, sumOnHost, refSum, error * 100,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    sum.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_min_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> minValue;
    minValue.initialize(cuContext, bufferType, 1);
    minValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            values.getDevicePointer(), minValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Min, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMin = std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMin = std::min(refMin, value);
        }
        values.unmap();

        minValue.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Min(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), minValue.getDevicePointer(), numElements);

        ValueType minOnHost;
        minValue.read(&minOnHost, 1);

        printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, minOnHost, refMin,
               minOnHost == refMin ? "" : " NG");

        allSuccess &= minOnHost == refMin;
    }
    printf("\n");

    tempStorage.finalize();
    minValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_min_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> minValue;
    minValue.initialize(cuContext, bufferType, 1);
    minValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Min(nullptr, tempStorageSize,
                            values.getDevicePointer(), minValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Min, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMin = std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMin = std::min(refMin, value);
        }
        values.unmap();

        minValue.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Min(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), minValue.getDevicePointer(), numElements);

        ValueType minOnHost;
        minValue.read(&minOnHost, 1);

        printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, minOnHost, refMin,
               minOnHost == refMin ? "" : " NG");

        allSuccess &= minOnHost == refMin;
    }
    printf("\n");

    tempStorage.finalize();
    minValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_min_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> minValue;
    minValue.initialize(cuContext, bufferType, 1);
    minValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Min(nullptr, tempStorageSize,
                            values.getDevicePointer(), minValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Min, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMin = std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMin = std::min(refMin, value);
        }
        values.unmap();

        minValue.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Min(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), minValue.getDevicePointer(), numElements);

        ValueType minOnHost;
        minValue.read(&minOnHost, 1);

        printf("  N: %5u, %g (ref: %g)%s\n", numElements, minOnHost, refMin,
               minOnHost == refMin ? "" : " NG");

        allSuccess &= minOnHost == refMin;
    }
    printf("\n");

    tempStorage.finalize();
    minValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_max_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> maxValue;
    maxValue.initialize(cuContext, bufferType, 1);
    maxValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                            values.getDevicePointer(), maxValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Max, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMax = std::numeric_limits<ValueType>::min();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMax = std::max(refMax, value);
        }
        values.unmap();

        maxValue.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Max(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), maxValue.getDevicePointer(), numElements);

        ValueType maxOnHost;
        maxValue.read(&maxOnHost, 1);

        printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, maxOnHost, refMax,
               maxOnHost == refMax ? "" : " NG");

        allSuccess &= maxOnHost == refMax;
    }
    printf("\n");

    tempStorage.finalize();
    maxValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_max_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> maxValue;
    maxValue.initialize(cuContext, bufferType, 1);
    maxValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                            values.getDevicePointer(), maxValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Max, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMax = std::numeric_limits<ValueType>::min();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMax = std::max(refMax, value);
        }
        values.unmap();

        maxValue.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Max(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), maxValue.getDevicePointer(), numElements);

        ValueType maxOnHost;
        maxValue.read(&maxOnHost, 1);

        printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, maxOnHost, refMax,
               maxOnHost == refMax ? "" : " NG");

        allSuccess &= maxOnHost == refMax;
    }
    printf("\n");

    tempStorage.finalize();
    maxValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_max_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> maxValue;
    maxValue.initialize(cuContext, bufferType, 1);
    maxValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                            values.getDevicePointer(), maxValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Max, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMax = std::numeric_limits<ValueType>::min();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMax = std::max(refMax, value);
        }
        values.unmap();

        maxValue.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Max(tempStorage.getDevicePointer(), tempStorageSize,
                                values.getDevicePointer(), maxValue.getDevicePointer(), numElements);

        ValueType maxOnHost;
        maxValue.read(&maxOnHost, 1);

        printf("  N: %5u, %g (ref: %g)%s\n", numElements, maxOnHost, refMax,
               maxOnHost == refMax ? "" : " NG");

        allSuccess &= maxOnHost == refMax;
    }
    printf("\n");

    tempStorage.finalize();
    maxValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_argmin_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<cubd::KeyValuePair<int32_t, ValueType>> minValue;
    minValue.initialize(cuContext, bufferType, 1);
    minValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                               values.getDevicePointer(), minValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::ArgMin, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMin = std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value < refMin) {
                refIdx = i;
                refMin = value;
            }
        }
        values.unmap();

        minValue.fill(cubd::KeyValuePair<int32_t, ValueType>{0, 0});

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMin(tempStorage.getDevicePointer(), tempStorageSize,
                                   values.getDevicePointer(), minValue.getDevicePointer(), numElements);

        cubd::KeyValuePair<int32_t, ValueType> minOnHost;
        minValue.read(&minOnHost, 1);

        bool success = minOnHost.key == refIdx && minOnHost.value == refMin;
        printf("  N:%5u, %8d at %6d (ref: %8d at %6d)%s\n", numElements,
               minOnHost.value, minOnHost.key, refMin, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    minValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_argmin_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<cubd::KeyValuePair<int32_t, ValueType>> minValue;
    minValue.initialize(cuContext, bufferType, 1);
    minValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                               values.getDevicePointer(), minValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::ArgMin, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMin = std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value < refMin) {
                refIdx = i;
                refMin = value;
            }
        }
        values.unmap();

        minValue.fill(cubd::KeyValuePair<int32_t, ValueType>{0, 0});

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMin(tempStorage.getDevicePointer(), tempStorageSize,
                                   values.getDevicePointer(), minValue.getDevicePointer(), numElements);

        cubd::KeyValuePair<int32_t, ValueType> minOnHost;
        minValue.read(&minOnHost, 1);

        bool success = minOnHost.key == refIdx && minOnHost.value == refMin;
        printf("  N:%5u, %8u at %6d (ref: %8u at %6d)%s\n", numElements,
               minOnHost.value, minOnHost.key, refMin, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    minValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_argmin_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<cubd::KeyValuePair<int32_t, ValueType>> minValue;
    minValue.initialize(cuContext, bufferType, 1);
    minValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                               values.getDevicePointer(), minValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::ArgMin, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMin = std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value < refMin) {
                refIdx = i;
                refMin = value;
            }
        }
        values.unmap();

        minValue.fill(cubd::KeyValuePair<int32_t, ValueType>{0, 0});

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMin(tempStorage.getDevicePointer(), tempStorageSize,
                                   values.getDevicePointer(), minValue.getDevicePointer(), numElements);

        cubd::KeyValuePair<int32_t, ValueType> minOnHost;
        minValue.read(&minOnHost, 1);

        bool success = minOnHost.key == refIdx && minOnHost.value == refMin;
        printf("  N:%5u, %g at %6d (ref: %g at %6d)%s\n", numElements,
               minOnHost.value, minOnHost.key, refMin, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    minValue.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_argmax_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<cubd::KeyValuePair<int32_t, ValueType>> maxValue;
    maxValue.initialize(cuContext, bufferType, 1);
    maxValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                               values.getDevicePointer(), maxValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::ArgMax, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMax = std::numeric_limits<ValueType>::min();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value > refMax) {
                refIdx = i;
                refMax = value;
            }
        }
        values.unmap();

        maxValue.fill(cubd::KeyValuePair<int32_t, ValueType>{0, 0});

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMax(tempStorage.getDevicePointer(), tempStorageSize,
                                   values.getDevicePointer(), maxValue.getDevicePointer(), numElements);

        cubd::KeyValuePair<int32_t, ValueType> maxOnHost;
        maxValue.read(&maxOnHost, 1);

        bool success = maxOnHost.key == refIdx && maxOnHost.value == refMax;
        printf("  N:%5u, %8d at %6d (ref: %8d at %6d)%s\n", numElements,
               maxOnHost.value, maxOnHost.key, refMax, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    maxValue.finalize();
    values.finalize();


    return allSuccess;
}

static bool test_argmax_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<cubd::KeyValuePair<int32_t, ValueType>> maxValue;
    maxValue.initialize(cuContext, bufferType, 1);
    maxValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                               values.getDevicePointer(), maxValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::ArgMax, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMax = std::numeric_limits<ValueType>::min();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value > refMax) {
                refIdx = i;
                refMax = value;
            }
        }
        values.unmap();

        maxValue.fill(cubd::KeyValuePair<int32_t, ValueType>{0, 0});

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMax(tempStorage.getDevicePointer(), tempStorageSize,
                                   values.getDevicePointer(), maxValue.getDevicePointer(), numElements);

        cubd::KeyValuePair<int32_t, ValueType> maxOnHost;
        maxValue.read(&maxOnHost, 1);

        bool success = maxOnHost.key == refIdx && maxOnHost.value == refMax;
        printf("  N:%5u, %8u at %6d (ref: %8u at %6d)%s\n", numElements,
               maxOnHost.value, maxOnHost.key, refMax, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    maxValue.finalize();
    values.finalize();


    return allSuccess;
}

static bool test_argmax_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<cubd::KeyValuePair<int32_t, ValueType>> maxValue;
    maxValue.initialize(cuContext, bufferType, 1);
    maxValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                               values.getDevicePointer(), maxValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::ArgMax, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMax = std::numeric_limits<ValueType>::min();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value > refMax) {
                refIdx = i;
                refMax = value;
            }
        }
        values.unmap();

        maxValue.fill(cubd::KeyValuePair<int32_t, ValueType>{0, 0});

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMax(tempStorage.getDevicePointer(), tempStorageSize,
                                   values.getDevicePointer(), maxValue.getDevicePointer(), numElements);

        cubd::KeyValuePair<int32_t, ValueType> maxOnHost;
        maxValue.read(&maxOnHost, 1);

        bool success = maxOnHost.key == refIdx && maxOnHost.value == refMax;
        printf("  N:%5u, %g at %6d (ref: %g at %6d)%s\n", numElements,
               maxOnHost.value, maxOnHost.key, refMax, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    maxValue.finalize();
    values.finalize();


    return allSuccess;
}

static bool test_exclusive_sum_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-100, 100);

    constexpr uint32_t MaxNumElements = 100000;

    std::vector<ValueType> refPrefixSums(MaxNumElements);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> prefixSums;
    prefixSums.initialize(cuContext, bufferType, MaxNumElements);
    prefixSums.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::ExclusiveSum(nullptr, tempStorageSize,
                                   values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceScan::ExclusiveSum, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType sum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        values.unmap();

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);

        ValueType* prefixSumsOnHost = prefixSums.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            success &= prefixSumsOnHost[i] == refPrefixSums[i];
            if (!success)
                break;
        }
        prefixSums.unmap();
        printf("  N:%5u, value at the end: %8d (ref: %8d)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();

    prefixSums.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_exclusive_sum_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 100);

    constexpr uint32_t MaxNumElements = 100000;
    
    std::vector<ValueType> refPrefixSums(MaxNumElements);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> prefixSums;
    prefixSums.initialize(cuContext, bufferType, MaxNumElements);
    prefixSums.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::ExclusiveSum(nullptr, tempStorageSize,
                                   values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceScan::ExclusiveSum, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType sum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        values.unmap();

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);

        ValueType* prefixSumsOnHost = prefixSums.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            success &= prefixSumsOnHost[i] == refPrefixSums[i];
            if (!success)
                break;
        }
        prefixSums.unmap();
        printf("  N:%5u, value at the end: %8u (ref: %8u)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();

    prefixSums.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_exclusive_sum_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    
    std::vector<ValueType> refPrefixSums(MaxNumElements);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> prefixSums;
    prefixSums.initialize(cuContext, bufferType, MaxNumElements);
    prefixSums.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::ExclusiveSum(nullptr, tempStorageSize,
                                   values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceScan::ExclusiveSum, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        double sum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        values.unmap();

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);

        ValueType* prefixSumsOnHost = prefixSums.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            ValueType error = (prefixSumsOnHost[i] - refPrefixSums[i]) / refPrefixSums[i];
            if (refPrefixSums[i] != 0)
                success &= std::fabs(error) < 0.001f;
            else
                ;
            if (!success)
                break;
        }
        prefixSums.unmap();
        printf("  N:%5u, value at the end: %g (ref: %g)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();

    prefixSums.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_inclusive_sum_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-100, 100);

    constexpr uint32_t MaxNumElements = 100000;
    
    std::vector<ValueType> refPrefixSums(MaxNumElements);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> prefixSums;
    prefixSums.initialize(cuContext, bufferType, MaxNumElements);
    prefixSums.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::InclusiveSum(nullptr, tempStorageSize,
                                   values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceScan::InclusiveSum, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType sum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            sum += value;
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
        }
        values.unmap();

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::InclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);

        ValueType* prefixSumsOnHost = prefixSums.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            success &= prefixSumsOnHost[i] == refPrefixSums[i];
            if (!success)
                break;
        }
        prefixSums.unmap();
        printf("  N:%5u, value at the end: %8d (ref: %8d)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();

    prefixSums.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_inclusive_sum_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 100);

    constexpr uint32_t MaxNumElements = 100000;
    
    std::vector<ValueType> refPrefixSums(MaxNumElements);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> prefixSums;
    prefixSums.initialize(cuContext, bufferType, MaxNumElements);
    prefixSums.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::InclusiveSum(nullptr, tempStorageSize,
                                   values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceScan::InclusiveSum, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType sum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            sum += value;
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
        }
        values.unmap();

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::InclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);

        ValueType* prefixSumsOnHost = prefixSums.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            success &= prefixSumsOnHost[i] == refPrefixSums[i];
            if (!success)
                break;
        }
        prefixSums.unmap();
        printf("  N:%5u, value at the end: %8u (ref: %8u)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();

    prefixSums.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_inclusive_sum_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    
    std::vector<ValueType> refPrefixSums(MaxNumElements);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> prefixSums;
    prefixSums.initialize(cuContext, bufferType, MaxNumElements);
    prefixSums.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::InclusiveSum(nullptr, tempStorageSize,
                                   values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceScan::InclusiveSum, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        double sum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            sum += value;
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
        }
        values.unmap();

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::InclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);

        ValueType* prefixSumsOnHost = prefixSums.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            ValueType error = (prefixSumsOnHost[i] - refPrefixSums[i]) / refPrefixSums[i];
            if (refPrefixSums[i] != 0)
                success &= std::fabs(error) < 0.001f;
            else
                ;
            if (!success)
                break;
        }
        prefixSums.unmap();
        printf("  N:%5u, value at the end: %g (ref: %g)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();

    prefixSums.finalize();
    values.finalize();

    return allSuccess;
}

static bool test_radix_sort_uint64_t_key_uint32_t_value() {
    using KeyType = uint64_t;
    using ValueType = uint32_t;

    std::uniform_int_distribution<KeyType> dist(0, 59237535202341);

    constexpr uint32_t MaxNumElements = 100000;

    std::vector<std::pair<KeyType, ValueType>> refKeyValuePairs(MaxNumElements);

    cudau::TypedBuffer<KeyType> keysA;
    cudau::TypedBuffer<KeyType> keysB;
    cudau::TypedBuffer<ValueType> valuesA;
    cudau::TypedBuffer<ValueType> valuesB;
    keysA.initialize(cuContext, bufferType, MaxNumElements);
    keysB.initialize(cuContext, bufferType, MaxNumElements);
    valuesA.initialize(cuContext, bufferType, MaxNumElements);
    valuesB.initialize(cuContext, bufferType, MaxNumElements);
    keysA.setMappedMemoryPersistent(true);
    keysB.setMappedMemoryPersistent(true);
    valuesA.setMappedMemoryPersistent(true);
    valuesB.setMappedMemoryPersistent(true);

    cubd::DoubleBuffer<KeyType> keys(keysA.getDevicePointer(), keysB.getDevicePointer());
    cubd::DoubleBuffer<ValueType> values(valuesA.getDevicePointer(), valuesB.getDevicePointer());

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceRadixSort::SortPairs(nullptr, tempStorageSize,
                                     keys, values, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceRadixSort::SortPairs, uint64_t / uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        cudau::TypedBuffer<KeyType> &curKeys = keys.selector ? keysB : keysA;
        cudau::TypedBuffer<ValueType> &curValues = values.selector ? valuesB : valuesA;
        KeyType* keysOnHost = curKeys.map();
        ValueType* valuesOnHost = curValues.map();
        for (int i = 0; i < numElements; ++i) {
            KeyType key = dist(rng);
            keysOnHost[i] = key;
            valuesOnHost[i] = i;
            refKeyValuePairs[i] = std::make_pair(key, i);
        }
        curValues.unmap();
        curKeys.unmap();
        std::stable_sort(refKeyValuePairs.begin(), refKeyValuePairs.begin() + numElements,
                         [](const std::pair<KeyType, ValueType> &pairA, const std::pair<KeyType, ValueType> &pairB) {
                             return pairA.first < pairB.first;
                         });

        // JP: ソートの実行。
        // EN: perform sort.
        cubd::DeviceRadixSort::SortPairs(tempStorage.getDevicePointer(), tempStorageSize,
                                         keys, values, numElements);

        cudau::TypedBuffer<KeyType> &sortedKeys = keys.selector ? keysB : keysA;
        cudau::TypedBuffer<ValueType> &sortedValues = values.selector ? valuesB : valuesA;
        KeyType* sortedKeysOnHost = sortedKeys.map();
        ValueType* sortedValuesOnHost = sortedValues.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            const std::pair<KeyType, ValueType> &refPair = refKeyValuePairs[i];
            success &= sortedKeysOnHost[i] == refPair.first && sortedValuesOnHost[i] == refPair.second;
            if (!success)
                break;
        }
        sortedValues.unmap();
        sortedKeys.unmap();
        printf("  N:%5u%s\n", numElements, success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    valuesB.finalize();
    valuesA.finalize();
    keysB.finalize();
    keysA.finalize();

    return allSuccess;
}

static bool test_radix_sort_uint64_t_key() {
    using KeyType = uint64_t;

    std::uniform_int_distribution<KeyType> dist(0, 59237535202341);

    constexpr uint32_t MaxNumElements = 100000;

    std::vector<KeyType> refKeys(MaxNumElements);

    cudau::TypedBuffer<KeyType> keysA;
    cudau::TypedBuffer<KeyType> keysB;
    keysA.initialize(cuContext, bufferType, MaxNumElements);
    keysB.initialize(cuContext, bufferType, MaxNumElements);
    keysA.setMappedMemoryPersistent(true);
    keysB.setMappedMemoryPersistent(true);

    cubd::DoubleBuffer<KeyType> keys(keysA.getDevicePointer(), keysB.getDevicePointer());

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceRadixSort::SortKeys(nullptr, tempStorageSize,
                                    keys, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceRadixSort::SortKeys, uint64_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        cudau::TypedBuffer<KeyType> &curKeys = keys.selector ? keysB : keysA;
        KeyType* keysOnHost = curKeys.map();
        for (int i = 0; i < numElements; ++i) {
            KeyType key = dist(rng);
            keysOnHost[i] = key;
            refKeys[i] = key;
        }
        curKeys.unmap();
        std::stable_sort(refKeys.begin(), refKeys.begin() + numElements);

        // JP: ソートの実行。
        // EN: perform sort.
        cubd::DeviceRadixSort::SortKeys(tempStorage.getDevicePointer(), tempStorageSize,
                                        keys, numElements);

        cudau::TypedBuffer<KeyType> &sortedKeys = keys.selector ? keysB : keysA;
        KeyType* sortedKeysOnHost = sortedKeys.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            success &= sortedKeysOnHost[i] == keysOnHost[i];
            if (!success)
                break;
        }
        sortedKeys.unmap();
        printf("  N:%5u%s\n", numElements, success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    keysB.finalize();
    keysA.finalize();

    return allSuccess;
}
