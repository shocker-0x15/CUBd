#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <random>
#include <limits>

#include "cubd.h"

// JP: このファイルは通常のC++コードとしてコンパイルできる。
// EN: This file can be compiled as an ordinary C++ code.

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
static bool test_exclusive_scan_int32_t();
static bool test_exclusive_scan_uint32_t();
static bool test_exclusive_scan_float();
static bool test_radix_sort_uint64_t_key_uint32_t_value();

int32_t main(int32_t argc, const char* argv[]) {
    cudaError_t err;

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

    success &= test_exclusive_scan_int32_t();
    success &= test_exclusive_scan_uint32_t();
    success &= test_exclusive_scan_float();

    success &= test_radix_sort_uint64_t_key_uint32_t_value();

    if (success)
        printf("All Success!\n");
    else
        printf("Something went wrong...\n");

    return 0;
}



std::mt19937_64 rng(194712984);

constexpr uint32_t NumTests = 10;

static bool test_sum_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-100, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* sumOnDevice;
    cudaMalloc(&sumOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            valuesOnDevice, sumOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Sum, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refSum = 0;
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(sumOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                valuesOnDevice, sumOnDevice, numElements);

        ValueType sumOnHost;
        cudaMemcpy(&sumOnHost, sumOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, sumOnHost, refSum,
               sumOnHost == refSum ? "" : " NG");

        allSuccess &= sumOnHost == refSum;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(sumOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_sum_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* sumOnDevice;
    cudaMalloc(&sumOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            valuesOnDevice, sumOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Sum, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refSum = 0;
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(sumOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                valuesOnDevice, sumOnDevice, numElements);

        ValueType sumOnHost;
        cudaMemcpy(&sumOnHost, sumOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, sumOnHost, refSum,
               sumOnHost == refSum ? "" : " NG");

        allSuccess &= sumOnHost == refSum;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(sumOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_sum_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* sumOnDevice;
    cudaMalloc(&sumOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            valuesOnDevice, sumOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Sum, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        double refSum = 0;
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(sumOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                valuesOnDevice, sumOnDevice, numElements);

        ValueType sumOnHost;
        cudaMemcpy(&sumOnHost, sumOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        ValueType error = (sumOnHost - refSum) / refSum;
        bool success = std::fabs(error) < 0.001f;
        printf("  N: %5u, %g (ref: %g), error: %.2f%%%s\n", numElements, sumOnHost, refSum, error * 100,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(sumOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_min_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* minOnDevice;
    cudaMalloc(&minOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                            valuesOnDevice, minOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Min, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMin = std::numeric_limits<ValueType>::max();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMin = std::min(refMin, value);
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(minOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Min(tempStorage, tempStorageSize,
                                valuesOnDevice, minOnDevice, numElements);

        ValueType minOnHost;
        cudaMemcpy(&minOnHost, minOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, minOnHost, refMin,
               minOnHost == refMin ? "" : " NG");

        allSuccess &= minOnHost == refMin;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(minOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_min_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* minOnDevice;
    cudaMalloc(&minOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Min(nullptr, tempStorageSize,
                            valuesOnDevice, minOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Min, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMin = std::numeric_limits<ValueType>::max();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMin = std::min(refMin, value);
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(minOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Min(tempStorage, tempStorageSize,
                                valuesOnDevice, minOnDevice, numElements);

        ValueType minOnHost;
        cudaMemcpy(&minOnHost, minOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, minOnHost, refMin,
               minOnHost == refMin ? "" : " NG");

        allSuccess &= minOnHost == refMin;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(minOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_min_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* minOnDevice;
    cudaMalloc(&minOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Min(nullptr, tempStorageSize,
                            valuesOnDevice, minOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Min, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMin = std::numeric_limits<ValueType>::max();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMin = std::min(refMin, value);
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(minOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Min(tempStorage, tempStorageSize,
                                valuesOnDevice, minOnDevice, numElements);

        ValueType minOnHost;
        cudaMemcpy(&minOnHost, minOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N: %5u, %g (ref: %g)%s\n", numElements, minOnHost, refMin,
               minOnHost == refMin ? "" : " NG");

        allSuccess &= minOnHost == refMin;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(minOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_max_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* maxOnDevice;
    cudaMalloc(&maxOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                            valuesOnDevice, maxOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Max, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMax = std::numeric_limits<ValueType>::min();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMax = std::max(refMax, value);
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(maxOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Max(tempStorage, tempStorageSize,
                                valuesOnDevice, maxOnDevice, numElements);

        ValueType maxOnHost;
        cudaMemcpy(&maxOnHost, maxOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, maxOnHost, refMax,
               maxOnHost == refMax ? "" : " NG");

        allSuccess &= maxOnHost == refMax;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(maxOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_max_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* maxOnDevice;
    cudaMalloc(&maxOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                            valuesOnDevice, maxOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Max, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMax = std::numeric_limits<ValueType>::min();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMax = std::max(refMax, value);
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(maxOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Max(tempStorage, tempStorageSize,
                                valuesOnDevice, maxOnDevice, numElements);

        ValueType maxOnHost;
        cudaMemcpy(&maxOnHost, maxOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, maxOnHost, refMax,
               maxOnHost == refMax ? "" : " NG");

        allSuccess &= maxOnHost == refMax;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(maxOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_max_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* maxOnDevice;
    cudaMalloc(&maxOnDevice, sizeof(ValueType));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                            valuesOnDevice, maxOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::Max, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refMax = std::numeric_limits<ValueType>::min();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refMax = std::max(refMax, value);
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(maxOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Max(tempStorage, tempStorageSize,
                                valuesOnDevice, maxOnDevice, numElements);

        ValueType maxOnHost;
        cudaMemcpy(&maxOnHost, maxOnDevice, sizeof(ValueType), cudaMemcpyDeviceToHost);

        printf("  N: %5u, %g (ref: %g)%s\n", numElements, maxOnHost, refMax,
               maxOnHost == refMax ? "" : " NG");

        allSuccess &= maxOnHost == refMax;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(maxOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_argmin_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    cubd::KeyValuePair<int32_t, ValueType>* minOnDevice;
    cudaMalloc(&minOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                               valuesOnDevice, minOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::ArgMin, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMin = std::numeric_limits<ValueType>::max();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value < refMin) {
                refIdx = i;
                refMin = value;
            }
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(minOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMin(tempStorage, tempStorageSize,
                                   valuesOnDevice, minOnDevice, numElements);

        cubd::KeyValuePair<int32_t, ValueType> minOnHost;
        cudaMemcpy(&minOnHost, minOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>), cudaMemcpyDeviceToHost);

        bool success = minOnHost.key == refIdx && minOnHost.value == refMin;
        printf("  N:%5u, %8d at %6d (ref: %8d at %6d)%s\n", numElements,
               minOnHost.value, minOnHost.key, refMin, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(minOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_argmin_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    cubd::KeyValuePair<int32_t, ValueType>* minOnDevice;
    cudaMalloc(&minOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                               valuesOnDevice, minOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::ArgMin, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMin = std::numeric_limits<ValueType>::max();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value < refMin) {
                refIdx = i;
                refMin = value;
            }
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(minOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMin(tempStorage, tempStorageSize,
                                   valuesOnDevice, minOnDevice, numElements);

        cubd::KeyValuePair<int32_t, ValueType> minOnHost;
        cudaMemcpy(&minOnHost, minOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>), cudaMemcpyDeviceToHost);

        bool success = minOnHost.key == refIdx && minOnHost.value == refMin;
        printf("  N:%5u, %8u at %6d (ref: %8u at %6d)%s\n", numElements,
               minOnHost.value, minOnHost.key, refMin, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(minOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_argmin_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    cubd::KeyValuePair<int32_t, ValueType>* minOnDevice;
    cudaMalloc(&minOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                               valuesOnDevice, minOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::ArgMin, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMin = std::numeric_limits<ValueType>::max();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value < refMin) {
                refIdx = i;
                refMin = value;
            }
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(minOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMin(tempStorage, tempStorageSize,
                                   valuesOnDevice, minOnDevice, numElements);

        cubd::KeyValuePair<int32_t, ValueType> minOnHost;
        cudaMemcpy(&minOnHost, minOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>), cudaMemcpyDeviceToHost);

        bool success = minOnHost.key == refIdx && minOnHost.value == refMin;
        printf("  N:%5u, %g at %6d (ref: %g at %6d)%s\n", numElements,
               minOnHost.value, minOnHost.key, refMin, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(minOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_argmax_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-1000000, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    cubd::KeyValuePair<int32_t, ValueType>* maxOnDevice;
    cudaMalloc(&maxOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                               valuesOnDevice, maxOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::ArgMax, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMax = std::numeric_limits<ValueType>::min();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value > refMax) {
                refIdx = i;
                refMax = value;
            }
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(maxOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMax(tempStorage, tempStorageSize,
                                   valuesOnDevice, maxOnDevice, numElements);

        cubd::KeyValuePair<int32_t, ValueType> maxOnHost;
        cudaMemcpy(&maxOnHost, maxOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>), cudaMemcpyDeviceToHost);

        bool success = maxOnHost.key == refIdx && maxOnHost.value == refMax;
        printf("  N:%5u, %8d at %6d (ref: %8d at %6d)%s\n", numElements,
               maxOnHost.value, maxOnHost.key, refMax, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(maxOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_argmax_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 1000000);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    cubd::KeyValuePair<int32_t, ValueType>* maxOnDevice;
    cudaMalloc(&maxOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                               valuesOnDevice, maxOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::ArgMax, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMax = std::numeric_limits<ValueType>::min();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value > refMax) {
                refIdx = i;
                refMax = value;
            }
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(maxOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMax(tempStorage, tempStorageSize,
                                   valuesOnDevice, maxOnDevice, numElements);

        cubd::KeyValuePair<int32_t, ValueType> maxOnHost;
        cudaMemcpy(&maxOnHost, maxOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>), cudaMemcpyDeviceToHost);

        bool success = maxOnHost.key == refIdx && maxOnHost.value == refMax;
        printf("  N:%5u, %8u at %6d (ref: %8u at %6d)%s\n", numElements,
               maxOnHost.value, maxOnHost.key, refMax, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(maxOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_argmax_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    cubd::KeyValuePair<int32_t, ValueType>* maxOnDevice;
    cudaMalloc(&maxOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>));

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                               valuesOnDevice, maxOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceReduce::ArgMax, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refMax = std::numeric_limits<ValueType>::min();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (value > refMax) {
                refIdx = i;
                refMax = value;
            }
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        cudaMemset(maxOnDevice, 0, sizeof(ValueType));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::ArgMax(tempStorage, tempStorageSize,
                                   valuesOnDevice, maxOnDevice, numElements);

        cubd::KeyValuePair<int32_t, ValueType> maxOnHost;
        cudaMemcpy(&maxOnHost, maxOnDevice, sizeof(cubd::KeyValuePair<int32_t, ValueType>), cudaMemcpyDeviceToHost);

        bool success = maxOnHost.key == refIdx && maxOnHost.value == refMax;
        printf("  N:%5u, %g at %6d (ref: %g at %6d)%s\n", numElements,
               maxOnHost.value, maxOnHost.key, refMax, refIdx,
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);
    cudaFree(maxOnDevice);
    cudaFree(valuesOnDevice);

    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_exclusive_scan_int32_t() {
    using ValueType = int32_t;

    std::uniform_int_distribution<ValueType> dist(-100, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];
    auto refPrefixSums = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* prefixSumsOnDevice;
    cudaMalloc(&prefixSumsOnDevice, MaxNumElements * sizeof(ValueType));
    auto prefixSumsOnHost = new ValueType[MaxNumElements];

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::ExclusiveSum(nullptr, tempStorageSize,
                                   valuesOnDevice, prefixSumsOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceScan::ExclusiveSum, int32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType sum = 0;
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage, tempStorageSize,
                                       valuesOnDevice, prefixSumsOnDevice, numElements);

        cudaMemcpy(prefixSumsOnHost, prefixSumsOnDevice, sizeof(ValueType) * numElements, cudaMemcpyDeviceToHost);

        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            success &= prefixSumsOnHost[i] == refPrefixSums[i];
            if (!success)
                break;
        }
        printf("  N:%5u, value at the end: %8d (ref: %8d)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);

    delete[] prefixSumsOnHost;
    cudaFree(prefixSumsOnDevice);
    cudaFree(valuesOnDevice);

    delete[] refPrefixSums;
    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_exclusive_scan_uint32_t() {
    using ValueType = uint32_t;

    std::uniform_int_distribution<ValueType> dist(0, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];
    auto refPrefixSums = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* prefixSumsOnDevice;
    cudaMalloc(&prefixSumsOnDevice, MaxNumElements * sizeof(ValueType));
    auto prefixSumsOnHost = new ValueType[MaxNumElements];

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::ExclusiveSum(nullptr, tempStorageSize,
                                   valuesOnDevice, prefixSumsOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceScan::ExclusiveSum, uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType sum = 0;
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage, tempStorageSize,
                                       valuesOnDevice, prefixSumsOnDevice, numElements);

        cudaMemcpy(prefixSumsOnHost, prefixSumsOnDevice, sizeof(ValueType) * numElements, cudaMemcpyDeviceToHost);

        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            success &= prefixSumsOnHost[i] == refPrefixSums[i];
            if (!success)
                break;
        }
        printf("  N:%5u, value at the end: %8u (ref: %8u)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);

    delete[] prefixSumsOnHost;
    cudaFree(prefixSumsOnDevice);
    cudaFree(valuesOnDevice);

    delete[] refPrefixSums;
    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_exclusive_scan_float() {
    using ValueType = float;

    std::uniform_real_distribution<ValueType> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new ValueType[MaxNumElements];
    auto refPrefixSums = new ValueType[MaxNumElements];

    ValueType* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(ValueType));

    ValueType* prefixSumsOnDevice;
    cudaMalloc(&prefixSumsOnDevice, MaxNumElements * sizeof(ValueType));
    auto prefixSumsOnHost = new ValueType[MaxNumElements];

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceScan::ExclusiveSum(nullptr, tempStorageSize,
                                   valuesOnDevice, prefixSumsOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceScan::ExclusiveSum, float:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        double sum = 0;
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage, tempStorageSize,
                                       valuesOnDevice, prefixSumsOnDevice, numElements);

        cudaMemcpy(prefixSumsOnHost, prefixSumsOnDevice, sizeof(ValueType) * numElements, cudaMemcpyDeviceToHost);

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
        printf("  N:%5u, value at the end: %g (ref: %g)%s\n", numElements, prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
               success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    cudaFree(tempStorage);

    delete[] prefixSumsOnHost;
    cudaFree(prefixSumsOnDevice);
    cudaFree(valuesOnDevice);

    delete[] refPrefixSums;
    delete[] valuesOnHost;

    return allSuccess;
}

static bool test_radix_sort_uint64_t_key_uint32_t_value() {
    using KeyType = uint64_t;
    using ValueType = uint32_t;

    std::uniform_int_distribution<KeyType> dist(0, 59237535202341);

    constexpr uint32_t MaxNumElements = 100000;
    auto keysOnHost = new KeyType[MaxNumElements];
    auto valuesOnHost = new ValueType[MaxNumElements];
    auto refKeyValuePairsOnHost = new std::pair<KeyType, ValueType>[MaxNumElements];

    KeyType* keysOnDeviceA;
    KeyType* keysOnDeviceB;
    ValueType* valuesOnDeviceA;
    ValueType* valuesOnDeviceB;
    cudaMalloc(&keysOnDeviceA, MaxNumElements * sizeof(KeyType));
    cudaMalloc(&keysOnDeviceB, MaxNumElements * sizeof(KeyType));
    cudaMalloc(&valuesOnDeviceA, MaxNumElements * sizeof(ValueType));
    cudaMalloc(&valuesOnDeviceB, MaxNumElements * sizeof(ValueType));

    cubd::DoubleBuffer<KeyType> keysOnDevice(keysOnDeviceA, keysOnDeviceB);
    cubd::DoubleBuffer<ValueType> valuesOnDevice(valuesOnDeviceA, valuesOnDeviceB);
    auto sortedKeysOnHost = new KeyType[MaxNumElements];
    auto sortedValuesOnHost = new ValueType[MaxNumElements];

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    cubd::DeviceRadixSort::SortPairs(nullptr, tempStorageSize,
                                     keysOnDevice, valuesOnDevice, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    void* tempStorage;
    cudaMalloc(&tempStorage, tempStorageSize);

    printf("DeviceRadixSort::SortPairs, uint64_t / uint32_t:\n");
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        for (int i = 0; i < numElements; ++i) {
            KeyType key = dist(rng);
            keysOnHost[i] = key;
            valuesOnHost[i] = i;
            refKeyValuePairsOnHost[i] = std::make_pair(key, i);
        }
        std::stable_sort(refKeyValuePairsOnHost, refKeyValuePairsOnHost + numElements,
                         [](const std::pair<KeyType, ValueType> &pairA, const std::pair<KeyType, ValueType> &pairB) {
                             return pairA.first < pairB.first;
                         });
        cudaMemcpy(keysOnDevice.Current(), keysOnHost, numElements * sizeof(KeyType), cudaMemcpyHostToDevice);
        cudaMemcpy(valuesOnDevice.Current(), valuesOnHost, numElements * sizeof(ValueType), cudaMemcpyHostToDevice);

        // JP: ソートの実行。
        // EN: perform sort.
        cubd::DeviceRadixSort::SortPairs(tempStorage, tempStorageSize,
                                         keysOnDevice, valuesOnDevice, numElements);

        cudaMemcpy(sortedKeysOnHost, keysOnDevice.Current(), sizeof(KeyType) * numElements, cudaMemcpyDeviceToHost);
        cudaMemcpy(sortedValuesOnHost, valuesOnDevice.Current(), sizeof(ValueType) * numElements, cudaMemcpyDeviceToHost);

        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            const std::pair<KeyType, ValueType> refPair = refKeyValuePairsOnHost[i];
            success &= sortedKeysOnHost[i] == refPair.first && sortedValuesOnHost[i] == refPair.second;
            if (!success)
                break;
        }
        printf("  N:%5u%s\n", numElements, success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    delete[] sortedValuesOnHost;
    delete[] sortedKeysOnHost;

    cudaFree(tempStorage);
    cudaFree(valuesOnDeviceB);
    cudaFree(valuesOnDeviceA);
    cudaFree(keysOnDeviceB);
    cudaFree(keysOnDeviceA);

    delete[] refKeyValuePairsOnHost;
    delete[] valuesOnHost;
    delete[]  keysOnHost;

    return allSuccess;
}
