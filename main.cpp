#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <random>

#include "cubd.h"

// JP: このファイルは通常のC++コードとしてコンパイルできる。
// EN: This file can be compiled as an ordinary C++ code.

static bool test_sum_int32_t();
static bool test_sum_uint32_t();
static bool test_sum_float();
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
    std::uniform_int_distribution<int32_t> dist(-100, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new int32_t[MaxNumElements];

    int32_t* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(int32_t));

    int32_t* sumOnDevice;
    cudaMalloc(&sumOnDevice, sizeof(int32_t));

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
        int32_t refSum = 0;
        for (int i = 0; i < numElements; ++i) {
            int32_t value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(int32_t), cudaMemcpyHostToDevice);

        cudaMemset(sumOnDevice, 0, sizeof(int32_t));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                valuesOnDevice, sumOnDevice, numElements);

        int32_t sumOnHost;
        cudaMemcpy(&sumOnHost, sumOnDevice, sizeof(int32_t), cudaMemcpyDeviceToHost);

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
    std::uniform_int_distribution<uint32_t> dist(0, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new uint32_t[MaxNumElements];

    uint32_t* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(uint32_t));

    uint32_t* sumOnDevice;
    cudaMalloc(&sumOnDevice, sizeof(uint32_t));

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
        uint32_t refSum = 0;
        for (int i = 0; i < numElements; ++i) {
            uint32_t value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaMemset(sumOnDevice, 0, sizeof(uint32_t));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                valuesOnDevice, sumOnDevice, numElements);

        uint32_t sumOnHost;
        cudaMemcpy(&sumOnHost, sumOnDevice, sizeof(uint32_t), cudaMemcpyDeviceToHost);

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
    std::uniform_real_distribution<float> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new float[MaxNumElements];

    float* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(float));

    float* sumOnDevice;
    cudaMalloc(&sumOnDevice, sizeof(float));

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
            float value = dist(rng);
            valuesOnHost[i] = value;
            refSum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(sumOnDevice, 0, sizeof(float));

        // JP: リダクションの実行。
        // EN: perform reduction.
        cubd::DeviceReduce::Sum(tempStorage, tempStorageSize,
                                valuesOnDevice, sumOnDevice, numElements);

        float sumOnHost;
        cudaMemcpy(&sumOnHost, sumOnDevice, sizeof(float), cudaMemcpyDeviceToHost);

        float error = (sumOnHost - refSum) / refSum;
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

static bool test_exclusive_scan_int32_t() {
    std::uniform_int_distribution<int32_t> dist(-100, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new int32_t[MaxNumElements];
    auto refPrefixSums = new int32_t[MaxNumElements];

    int32_t* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(int32_t));

    int32_t* prefixSumsOnDevice;
    cudaMalloc(&prefixSumsOnDevice, MaxNumElements * sizeof(int32_t));
    auto prefixSumsOnHost = new int32_t[MaxNumElements];

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
        int32_t sum = 0;
        for (int i = 0; i < numElements; ++i) {
            int32_t value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(int32_t), cudaMemcpyHostToDevice);

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage, tempStorageSize,
                                       valuesOnDevice, prefixSumsOnDevice, numElements);

        cudaMemcpy(prefixSumsOnHost, prefixSumsOnDevice, sizeof(int32_t) * numElements, cudaMemcpyDeviceToHost);

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
    std::uniform_int_distribution<uint32_t> dist(0, 100);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new uint32_t[MaxNumElements];
    auto refPrefixSums = new uint32_t[MaxNumElements];

    uint32_t* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(uint32_t));

    uint32_t* prefixSumsOnDevice;
    cudaMalloc(&prefixSumsOnDevice, MaxNumElements * sizeof(uint32_t));
    auto prefixSumsOnHost = new uint32_t[MaxNumElements];

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
        uint32_t sum = 0;
        for (int i = 0; i < numElements; ++i) {
            uint32_t value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage, tempStorageSize,
                                       valuesOnDevice, prefixSumsOnDevice, numElements);

        cudaMemcpy(prefixSumsOnHost, prefixSumsOnDevice, sizeof(uint32_t) * numElements, cudaMemcpyDeviceToHost);

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
    std::uniform_real_distribution<float> dist(0, 1);

    constexpr uint32_t MaxNumElements = 100000;
    auto valuesOnHost = new float[MaxNumElements];
    auto refPrefixSums = new float[MaxNumElements];

    float* valuesOnDevice;
    cudaMalloc(&valuesOnDevice, MaxNumElements * sizeof(float));

    float* prefixSumsOnDevice;
    cudaMalloc(&prefixSumsOnDevice, MaxNumElements * sizeof(float));
    auto prefixSumsOnHost = new float[MaxNumElements];

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
            float value = dist(rng);
            valuesOnHost[i] = value;
            refPrefixSums[i] = sum;
            sum += value;
        }
        cudaMemcpy(valuesOnDevice, valuesOnHost, numElements * sizeof(float), cudaMemcpyHostToDevice);

        // JP: スキャンの実行。
        // EN: perform scan.
        cubd::DeviceScan::ExclusiveSum(tempStorage, tempStorageSize,
                                       valuesOnDevice, prefixSumsOnDevice, numElements);

        cudaMemcpy(prefixSumsOnHost, prefixSumsOnDevice, sizeof(float) * numElements, cudaMemcpyDeviceToHost);

        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            float error = (prefixSumsOnHost[i] - refPrefixSums[i]) / refPrefixSums[i];
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
    std::uniform_int_distribution<uint64_t> dist(0, 59237535202341);

    constexpr uint32_t MaxNumElements = 100000;
    auto keysOnHost = new uint64_t[MaxNumElements];
    auto valuesOnHost = new uint32_t[MaxNumElements];
    auto refKeyValuePairsOnHost = new std::pair<uint64_t, uint32_t>[MaxNumElements];

    uint64_t* keysOnDeviceA;
    uint64_t* keysOnDeviceB;
    uint32_t* valuesOnDeviceA;
    uint32_t* valuesOnDeviceB;
    cudaMalloc(&keysOnDeviceA, MaxNumElements * sizeof(uint64_t));
    cudaMalloc(&keysOnDeviceB, MaxNumElements * sizeof(uint64_t));
    cudaMalloc(&valuesOnDeviceA, MaxNumElements * sizeof(uint32_t));
    cudaMalloc(&valuesOnDeviceB, MaxNumElements * sizeof(uint32_t));

    cubd::DoubleBuffer<uint64_t> keysOnDevice(keysOnDeviceA, keysOnDeviceB);
    cubd::DoubleBuffer<uint32_t> valuesOnDevice(valuesOnDeviceA, valuesOnDeviceB);
    auto sortedKeysOnHost = new uint64_t[MaxNumElements];
    auto sortedValuesOnHost = new uint32_t[MaxNumElements];

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
            uint64_t key = dist(rng);
            keysOnHost[i] = key;
            valuesOnHost[i] = i;
            refKeyValuePairsOnHost[i] = std::make_pair(key, i);
        }
        std::stable_sort(refKeyValuePairsOnHost, refKeyValuePairsOnHost + numElements,
                         [](const std::pair<uint64_t, uint32_t> &pairA, const std::pair<uint64_t, uint32_t> &pairB) {
                             return pairA.first < pairB.first;
                         });
        cudaMemcpy(keysOnDevice.Current(), keysOnHost, numElements * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(valuesOnDevice.Current(), valuesOnHost, numElements * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // JP: ソートの実行。
        // EN: perform sort.
        cubd::DeviceRadixSort::SortPairs(tempStorage, tempStorageSize,
                                         keysOnDevice, valuesOnDevice, numElements);

        cudaMemcpy(sortedKeysOnHost, keysOnDevice.Current(), sizeof(uint64_t) * numElements, cudaMemcpyDeviceToHost);
        cudaMemcpy(sortedValuesOnHost, valuesOnDevice.Current(), sizeof(uint32_t) * numElements, cudaMemcpyDeviceToHost);

        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            const std::pair<uint64_t, uint32_t> refPair = refKeyValuePairsOnHost[i];
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
