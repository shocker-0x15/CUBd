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



template <typename RealType>
struct CompensatedSum {
    RealType result;
    RealType comp;
    CompensatedSum(const RealType &value) : result(value), comp(0.0) { };
    CompensatedSum &operator=(const RealType &value) {
        result = value;
        comp = 0;
        return *this;
    }
    CompensatedSum &operator+=(const RealType &value) {
        RealType cInput = value - comp;
        RealType sumTemp = result + cInput;
        comp = (sumTemp - result) - cInput;
        result = sumTemp;
        return *this;
    }
    operator RealType() const { return result; };
};

struct Int32Traits {
    using ValueType = int32_t;
    using SumValueType = int32_t;
    using DistributionType = std::uniform_int_distribution<int32_t>;
    static constexpr const char* s_keyword = "int32_t";
};
struct UInt32Traits {
    using ValueType = uint32_t;
    using SumValueType = uint32_t;
    using DistributionType = std::uniform_int_distribution<uint32_t>;
    static constexpr const char* s_keyword = "uint32_t";
};
struct Float32Traits {
    using ValueType = float;
    using SumValueType = CompensatedSum<float>;
    using DistributionType = std::uniform_real_distribution<float>;
    static constexpr const char* s_keyword = "float";
};



template <typename TypeTraits>
static bool test_sum(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax);
template <typename TypeTraits, bool maxOp>
static bool test_minMax(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax);
template <typename TypeTraits, bool maxOp>
static bool test_argMinMax(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax);
template <typename TypeTraits, bool inclusive>
static bool test_prefixSum(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax);
static bool test_radix_sort_uint64_t_key_uint32_t_value();
static bool test_radix_sort_uint64_t_key();

static CUcontext cuContext;
static CUstream cuStream;
static cudau::BufferType bufferType = cudau::BufferType::Device;
std::mt19937_64 rng(194712984);

int32_t main(int32_t argc, const char* argv[]) {
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    std::mt19937 rng(194712984);

    printf("Start tests.\n");

    bool success = true;

    success &= test_sum<Int32Traits>(100000, -100, 100);
    success &= test_sum<UInt32Traits>(100000, 0, 100);
    success &= test_sum<Float32Traits>(100000, 0, 1);

    success &= test_minMax<Int32Traits, false>(100000, -1000000, 1000000);
    success &= test_minMax<UInt32Traits, false>(100000, 0, 1000000);
    success &= test_minMax<Float32Traits, false>(100000, 0, 1);

    success &= test_minMax<Int32Traits, true>(100000, -1000000, 1000000);
    success &= test_minMax<UInt32Traits, true>(100000, 0, 1000000);
    success &= test_minMax<Float32Traits, true>(100000, 0, 1);

    success &= test_argMinMax<Int32Traits, false>(100000, -1000000, 1000000);
    success &= test_argMinMax<UInt32Traits, false>(100000, 0, 1000000);
    success &= test_argMinMax<Float32Traits, false>(100000, 0, 1);

    success &= test_argMinMax<Int32Traits, true>(100000, -1000000, 1000000);
    success &= test_argMinMax<UInt32Traits, true>(100000, 0, 1000000);
    success &= test_argMinMax<Float32Traits, true>(100000, 0, 1);

    success &= test_prefixSum<Int32Traits, false>(100000, -100, 100);
    success &= test_prefixSum<UInt32Traits, false>(100000, 0, 100);
    success &= test_prefixSum<Float32Traits, false>(100000, 0, 1);

    success &= test_prefixSum<Int32Traits, true>(100000, -100, 100);
    success &= test_prefixSum<UInt32Traits, true>(100000, 0, 100);
    success &= test_prefixSum<Float32Traits, true>(100000, 0, 1);

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



constexpr uint32_t NumTests = 10;

template <typename TypeTraits>
static bool test_sum(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax) {
    using ValueType = typename TypeTraits::ValueType;
    using SumValueType = typename TypeTraits::SumValueType;
    using DistributionType = typename TypeTraits::DistributionType;

    DistributionType dist(distMin, distMax);

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

    printf("DeviceReduce::Sum, %s:\n", TypeTraits::s_keyword);
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);

        ValueType* valuesOnHost = values.map();
        SumValueType refSum = 0;
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

        if constexpr (std::is_same<ValueType, int32_t>::value) {
            printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, sumOnHost, refSum,
                   sumOnHost == refSum ? "" : " NG");
            allSuccess &= sumOnHost == refSum;
        }
        else if constexpr (std::is_same<ValueType, uint32_t>::value) {
            printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, sumOnHost, refSum,
                   sumOnHost == refSum ? "" : " NG");
            allSuccess &= sumOnHost == refSum;
        }
        else if constexpr (std::is_same<ValueType, float>::value) {
            ValueType error = (sumOnHost - refSum) / refSum;
            bool success = std::fabs(error) < 0.001f;
            printf("  N: %5u, %g (ref: %g), error: %.2f%%%s\n", numElements,
                   sumOnHost, static_cast<double>(refSum), error * 100,
                   success ? "" : " NG");
            allSuccess &= success;
        }
    }
    printf("\n");

    tempStorage.finalize();
    sum.finalize();
    values.finalize();

    return allSuccess;
}

template <typename TypeTraits, bool maxOp>
static bool test_minMax(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax) {
    using ValueType = typename TypeTraits::ValueType;
    using SumValueType = typename TypeTraits::SumValueType;
    using DistributionType = typename TypeTraits::DistributionType;

    DistributionType dist(distMin, distMax);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> resultValue;
    resultValue.initialize(cuContext, bufferType, 1);
    resultValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    if constexpr (maxOp)
        cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                                values.getDevicePointer(), resultValue.getDevicePointer(), MaxNumElements);
    else
        cubd::DeviceReduce::Min(nullptr, tempStorageSize,
                                values.getDevicePointer(), resultValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::%s, %s:\n", maxOp ? "Max" : "Min", TypeTraits::s_keyword);
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        ValueType refResult = maxOp ?
            std::numeric_limits<ValueType>::lowest() :
            std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if constexpr (maxOp)
                refResult = std::max(refResult, value);
            else
                refResult = std::min(refResult, value);
        }
        values.unmap();

        resultValue.fill(0);

        // JP: リダクションの実行。
        // EN: perform reduction.
        if constexpr (maxOp)
            cubd::DeviceReduce::Max(tempStorage.getDevicePointer(), tempStorageSize,
                                    values.getDevicePointer(), resultValue.getDevicePointer(), numElements);
        else
            cubd::DeviceReduce::Min(tempStorage.getDevicePointer(), tempStorageSize,
                                    values.getDevicePointer(), resultValue.getDevicePointer(), numElements);

        ValueType resultOnHost;
        resultValue.read(&resultOnHost, 1);

        if constexpr (std::is_same<ValueType, int32_t>::value)
            printf("  N:%5u, %8d (ref: %8d)%s\n", numElements, resultOnHost, refResult,
                   resultOnHost == refResult ? "" : " NG");
        else if constexpr (std::is_same<ValueType, uint32_t>::value)
            printf("  N:%5u, %8u (ref: %8u)%s\n", numElements, resultOnHost, refResult,
                   resultOnHost == refResult ? "" : " NG");
        else if constexpr (std::is_same<ValueType, float>::value)
            printf("  N: %5u, %g (ref: %g)%s\n", numElements, resultOnHost, refResult,
                   resultOnHost == refResult ? "" : " NG");

        allSuccess &= resultOnHost == refResult;
    }
    printf("\n");

    tempStorage.finalize();
    resultValue.finalize();
    values.finalize();

    return allSuccess;
}

template <typename TypeTraits, bool maxOp>
static bool test_argMinMax(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax) {
    using ValueType = typename TypeTraits::ValueType;
    using SumValueType = typename TypeTraits::SumValueType;
    using DistributionType = typename TypeTraits::DistributionType;

    DistributionType dist(distMin, distMax);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<cubd::KeyValuePair<int32_t, ValueType>> resultValue;
    resultValue.initialize(cuContext, bufferType, 1);
    resultValue.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    if constexpr (maxOp)
        cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                                   values.getDevicePointer(), resultValue.getDevicePointer(), MaxNumElements);
    else
        cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                                   values.getDevicePointer(), resultValue.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::Arg%s, %s:\n", maxOp ? "Max" : "Min", TypeTraits::s_keyword);
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        int32_t refIdx = -1;
        ValueType refResult = maxOp ?
            std::numeric_limits<ValueType>::lowest() :
            std::numeric_limits<ValueType>::max();
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if (maxOp ? value > refResult : value < refResult) {
                refIdx = i;
                refResult = value;
            }
        }
        values.unmap();

        resultValue.fill(cubd::KeyValuePair<int32_t, ValueType>{0, 0});

        // JP: リダクションの実行。
        // EN: perform reduction.
        if constexpr (maxOp)
            cubd::DeviceReduce::ArgMax(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), resultValue.getDevicePointer(), numElements);
        else
            cubd::DeviceReduce::ArgMin(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), resultValue.getDevicePointer(), numElements);

        cubd::KeyValuePair<int32_t, ValueType> resultOnHost;
        resultValue.read(&resultOnHost, 1);

        bool success = resultOnHost.key == refIdx && resultOnHost.value == refResult;
        if constexpr (std::is_same<ValueType, int32_t>::value)
            printf("  N:%5u, %8d at %6d (ref: %8d at %6d)%s\n", numElements,
                   resultOnHost.value, resultOnHost.key, refResult, refIdx,
                   success ? "" : " NG");
        else if constexpr (std::is_same<ValueType, uint32_t>::value)
            printf("  N:%5u, %8u at %6d (ref: %8u at %6d)%s\n", numElements,
                   resultOnHost.value, resultOnHost.key, refResult, refIdx,
                   success ? "" : " NG");
        else if constexpr (std::is_same<ValueType, float>::value)
            printf("  N:%5u, %g at %6d (ref: %g at %6d)%s\n", numElements,
                   resultOnHost.value, resultOnHost.key, refResult, refIdx,
                   success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    resultValue.finalize();
    values.finalize();

    return allSuccess;
}

template <typename TypeTraits, bool inclusive>
static bool test_prefixSum(uint32_t MaxNumElements, typename TypeTraits::ValueType distMin, typename TypeTraits::ValueType distMax) {
    using ValueType = typename TypeTraits::ValueType;
    using SumValueType = typename TypeTraits::SumValueType;
    using DistributionType = typename TypeTraits::DistributionType;

    DistributionType dist(distMin, distMax);

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
    if constexpr (inclusive)
        cubd::DeviceScan::InclusiveSum(nullptr, tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);
    else
        cubd::DeviceScan::ExclusiveSum(nullptr, tempStorageSize,
                                       values.getDevicePointer(), prefixSums.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceScan::%s, %s:\n", inclusive ? "Inclusive" : "Exclusive", TypeTraits::s_keyword);
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        SumValueType sum = 0;
        ValueType* valuesOnHost = values.map();
        for (int i = 0; i < numElements; ++i) {
            ValueType value = dist(rng);
            valuesOnHost[i] = value;
            if constexpr (inclusive)
                sum += value;
            refPrefixSums[i] = sum;
            if constexpr (!inclusive)
                sum += value;
        }
        values.unmap();

        // JP: スキャンの実行。
        // EN: perform scan.
        if constexpr (inclusive)
            cubd::DeviceScan::InclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                           values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);
        else
            cubd::DeviceScan::ExclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                           values.getDevicePointer(), prefixSums.getDevicePointer(), numElements);

        ValueType* prefixSumsOnHost = prefixSums.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            if constexpr (std::is_same<TypeTraits, Float32Traits>::value) {
                ValueType error = (prefixSumsOnHost[i] - refPrefixSums[i]) / refPrefixSums[i];
                if (refPrefixSums[i] != 0)
                    success &= std::fabs(error) < 0.001f;
                else
                    ;
            }
            else {
                success &= prefixSumsOnHost[i] == refPrefixSums[i];
            }
            if (!success)
                break;
        }
        prefixSums.unmap();

        if constexpr (std::is_same<ValueType, int32_t>::value)
            printf("  N:%5u, value at the end: %8d (ref: %8d)%s\n", numElements,
                   prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
                   success ? "" : " NG");
        else if constexpr (std::is_same<ValueType, uint32_t>::value)
            printf("  N:%5u, value at the end: %8u (ref: %8u)%s\n", numElements,
                   prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
                   success ? "" : " NG");
        else if constexpr (std::is_same<ValueType, float>::value)
            printf("  N:%5u, value at the end: %g (ref: %g)%s\n", numElements,
                   prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
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
