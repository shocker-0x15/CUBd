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
    using Type = int32_t;
    using SumType = int32_t;
    using DistributionType = std::uniform_int_distribution<int32_t>;
    static constexpr const char* s_keyword = "int32_t";
};
struct UInt32Traits {
    using Type = uint32_t;
    using SumType = uint32_t;
    using DistributionType = std::uniform_int_distribution<uint32_t>;
    static constexpr const char* s_keyword = "uint32_t";
};
struct Float32Traits {
    using Type = float;
    using SumType = CompensatedSum<float>;
    using DistributionType = std::uniform_real_distribution<float>;
    static constexpr const char* s_keyword = "float";
};
struct Int64Traits {
    using Type = int64_t;
    using SumType = int64_t;
    using DistributionType = std::uniform_int_distribution<int64_t>;
    static constexpr const char* s_keyword = "int64_t";
};
struct UInt64Traits {
    using Type = uint64_t;
    using SumType = uint64_t;
    using DistributionType = std::uniform_int_distribution<uint64_t>;
    static constexpr const char* s_keyword = "uint64_t";
};



enum class ReduceOpType {
    Sum = 0,
    Min,
    Max,
    ArgMin,
    ArgMax,
};
static constexpr const char* reduceOpKeywords[] = {
    "Sum",
    "Min",
    "Max",
    "ArgMin",
    "ArgMax",
};

enum class RadixSortOpType {
    SortKeys = 0,
    SortKeysDescending,
    SortPairs,
    SortPairsDescending,
};
static constexpr const char* radixSortOpKeywords[] = {
    "SortKeys",
    "SortKeysDescending",
    "SortPairs",
    "SortPairsDescending",
};

template <typename TypeTraits, ReduceOpType opType>
static bool test_DeviceReduce(uint32_t MaxNumElements, typename TypeTraits::Type distMin, typename TypeTraits::Type distMax);
template <typename TypeTraits, bool inclusive>
static bool test_DeviceScan(uint32_t MaxNumElements, typename TypeTraits::Type distMin, typename TypeTraits::Type distMax);
template <typename TypeTraits, RadixSortOpType opType>
static bool test_DeviceRadixSort(uint32_t MaxNumElements, typename TypeTraits::Type distMin, typename TypeTraits::Type distMax);

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

    success &= test_DeviceReduce<Int32Traits, ReduceOpType::Sum>(100000, -100, 100);
    success &= test_DeviceReduce<UInt32Traits, ReduceOpType::Sum>(100000, 0, 100);
    success &= test_DeviceReduce<Float32Traits, ReduceOpType::Sum>(100000, 0, 1);
    success &= test_DeviceReduce<Int64Traits, ReduceOpType::Sum>(100000, -1000000, 1000000);
    success &= test_DeviceReduce<UInt64Traits, ReduceOpType::Sum>(100000, 0, 1000000);

    success &= test_DeviceReduce<Int32Traits, ReduceOpType::Min>(100000, -1000000, 1000000);
    success &= test_DeviceReduce<UInt32Traits, ReduceOpType::Min>(100000, 0, 1000000);
    success &= test_DeviceReduce<Float32Traits, ReduceOpType::Min>(100000, 0, 1);
    success &= test_DeviceReduce<Int64Traits, ReduceOpType::Min>(100000, -100000000000, 100000000000);
    success &= test_DeviceReduce<UInt64Traits, ReduceOpType::Min>(100000, 0, 100000000000);

    success &= test_DeviceReduce<Int32Traits, ReduceOpType::Max>(100000, -1000000, 1000000);
    success &= test_DeviceReduce<UInt32Traits, ReduceOpType::Max>(100000, 0, 1000000);
    success &= test_DeviceReduce<Float32Traits, ReduceOpType::Max>(100000, 0, 1);
    success &= test_DeviceReduce<Int64Traits, ReduceOpType::Max>(100000, -100000000000, 100000000000);
    success &= test_DeviceReduce<UInt64Traits, ReduceOpType::Max>(100000, 0, 100000000000);

    success &= test_DeviceReduce<Int32Traits, ReduceOpType::ArgMin>(100000, -1000000, 1000000);
    success &= test_DeviceReduce<UInt32Traits, ReduceOpType::ArgMin>(100000, 0, 1000000);
    success &= test_DeviceReduce<Float32Traits, ReduceOpType::ArgMin>(100000, 0, 1);
    success &= test_DeviceReduce<Int64Traits, ReduceOpType::ArgMin>(100000, -100000000000, 100000000000);
    success &= test_DeviceReduce<UInt64Traits, ReduceOpType::ArgMin>(100000, 0, 100000000000);

    success &= test_DeviceReduce<Int32Traits, ReduceOpType::ArgMax>(100000, -1000000, 1000000);
    success &= test_DeviceReduce<UInt32Traits, ReduceOpType::ArgMax>(100000, 0, 1000000);
    success &= test_DeviceReduce<Float32Traits, ReduceOpType::ArgMax>(100000, 0, 1);
    success &= test_DeviceReduce<Int64Traits, ReduceOpType::ArgMax>(100000, -100000000000, 100000000000);
    success &= test_DeviceReduce<UInt64Traits, ReduceOpType::ArgMax>(100000, 0, 100000000000);

    success &= test_DeviceScan<Int32Traits, false>(100000, -100, 100);
    success &= test_DeviceScan<UInt32Traits, false>(100000, 0, 100);
    success &= test_DeviceScan<Float32Traits, false>(100000, 0, 1);
    success &= test_DeviceScan<Int64Traits, false>(100000, -1000000, 1000000);
    success &= test_DeviceScan<UInt64Traits, false>(100000, 0, 1000000);

    success &= test_DeviceScan<Int32Traits, true>(100000, -100, 100);
    success &= test_DeviceScan<UInt32Traits, true>(100000, 0, 100);
    success &= test_DeviceScan<Float32Traits, true>(100000, 0, 1);
    success &= test_DeviceScan<Int64Traits, true>(100000, -1000000, 1000000);
    success &= test_DeviceScan<UInt64Traits, true>(100000, 0, 1000000);

    success &= test_DeviceRadixSort<UInt32Traits, RadixSortOpType::SortKeys>(100000, 0, std::numeric_limits<uint32_t>::max());
    success &= test_DeviceRadixSort<UInt64Traits, RadixSortOpType::SortKeys>(100000, 0, std::numeric_limits<uint64_t>::max());

    success &= test_DeviceRadixSort<UInt32Traits, RadixSortOpType::SortKeysDescending>(100000, 0, std::numeric_limits<uint32_t>::max());
    success &= test_DeviceRadixSort<UInt64Traits, RadixSortOpType::SortKeysDescending>(100000, 0, std::numeric_limits<uint64_t>::max());

    success &= test_DeviceRadixSort<UInt32Traits, RadixSortOpType::SortPairs>(100000, 0, std::numeric_limits<uint32_t>::max());
    success &= test_DeviceRadixSort<UInt64Traits, RadixSortOpType::SortPairs>(100000, 0, std::numeric_limits<uint64_t>::max());

    success &= test_DeviceRadixSort<UInt32Traits, RadixSortOpType::SortPairsDescending>(100000, 0, std::numeric_limits<uint32_t>::max());
    success &= test_DeviceRadixSort<UInt64Traits, RadixSortOpType::SortPairsDescending>(100000, 0, std::numeric_limits<uint64_t>::max());

    if (success)
        printf("All Success!\n");
    else
        printf("Something went wrong...\n");

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}



constexpr uint32_t NumTests = 10;

template <typename TypeTraits, ReduceOpType opType>
static bool test_DeviceReduce(uint32_t MaxNumElements, typename TypeTraits::Type distMin, typename TypeTraits::Type distMax) {
    using ValueType = typename TypeTraits::Type;
    using ResultType = typename std::conditional<
        opType == ReduceOpType::ArgMin || opType == ReduceOpType::ArgMax,
        cubd::KeyValuePair<int32_t, ValueType>,
        ValueType>::type;
    using SumValueType = typename TypeTraits::SumType;
    using DistributionType = typename TypeTraits::DistributionType;

    DistributionType dist(distMin, distMax);

    cudau::TypedBuffer<ValueType> values;
    values.initialize(cuContext, bufferType, MaxNumElements);
    values.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ResultType> result;
    result.initialize(cuContext, bufferType, 1);
    result.setMappedMemoryPersistent(true);

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    if constexpr (opType == ReduceOpType::Sum)
        cubd::DeviceReduce::Sum(nullptr, tempStorageSize,
                                values.getDevicePointer(), result.getDevicePointer(), MaxNumElements);
    else if constexpr (opType == ReduceOpType::Min)
        cubd::DeviceReduce::Min(nullptr, tempStorageSize,
                                values.getDevicePointer(), result.getDevicePointer(), MaxNumElements);
    else if constexpr (opType == ReduceOpType::Max)
        cubd::DeviceReduce::Max(nullptr, tempStorageSize,
                                values.getDevicePointer(), result.getDevicePointer(), MaxNumElements);
    else if constexpr (opType == ReduceOpType::ArgMin)
        cubd::DeviceReduce::ArgMin(nullptr, tempStorageSize,
                                   values.getDevicePointer(), result.getDevicePointer(), MaxNumElements);
    else if constexpr (opType == ReduceOpType::ArgMax)
        cubd::DeviceReduce::ArgMax(nullptr, tempStorageSize,
                                   values.getDevicePointer(), result.getDevicePointer(), MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceReduce::%s, %s:\n", reduceOpKeywords[static_cast<uint32_t>(opType)], TypeTraits::s_keyword);
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);

        ValueType* valuesOnHost = values.map();
        int32_t refIdx = -1;
        ValueType refValue = 0;
        if constexpr (opType == ReduceOpType::Sum) {
            SumValueType sum = 0;
            for (int i = 0; i < numElements; ++i) {
                ValueType value = dist(rng);
                valuesOnHost[i] = value;
                sum += value;
            }
            refValue = sum;
        }
        else {
            refValue = (opType == ReduceOpType::Max || opType == ReduceOpType::ArgMax) ?
                std::numeric_limits<ValueType>::lowest() :
                std::numeric_limits<ValueType>::max();
            for (int i = 0; i < numElements; ++i) {
                ValueType value = dist(rng);
                valuesOnHost[i] = value;
                if ((opType == ReduceOpType::Max || opType == ReduceOpType::ArgMax) ? value > refValue : value < refValue) {
                    refIdx = i;
                    refValue = value;
                }
            }
        }
        values.unmap();

        ResultType fillValue;
        if constexpr (opType == ReduceOpType::ArgMin ||
                      opType == ReduceOpType::ArgMax)
            fillValue = ResultType{ 0, 0 };
        else
            fillValue = 0;
        result.fill(fillValue);

        // JP: リダクションの実行。
        // EN: perform reduction.
        if constexpr (opType == ReduceOpType::Sum)
            cubd::DeviceReduce::Sum(tempStorage.getDevicePointer(), tempStorageSize,
                                    values.getDevicePointer(), result.getDevicePointer(), numElements,
                                    cuStream);
        else if constexpr (opType == ReduceOpType::Min)
            cubd::DeviceReduce::Min(tempStorage.getDevicePointer(), tempStorageSize,
                                    values.getDevicePointer(), result.getDevicePointer(), numElements,
                                    cuStream);
        else if constexpr (opType == ReduceOpType::Max)
            cubd::DeviceReduce::Max(tempStorage.getDevicePointer(), tempStorageSize,
                                    values.getDevicePointer(), result.getDevicePointer(), numElements,
                                    cuStream);
        else if constexpr (opType == ReduceOpType::ArgMin)
            cubd::DeviceReduce::ArgMin(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), result.getDevicePointer(), numElements,
                                       cuStream);
        else if constexpr (opType == ReduceOpType::ArgMax)
            cubd::DeviceReduce::ArgMax(tempStorage.getDevicePointer(), tempStorageSize,
                                       values.getDevicePointer(), result.getDevicePointer(), numElements,
                                       cuStream);

        ResultType resultOnHost;
        result.read(&resultOnHost, 1, cuStream);

        CUDADRV_CHECK(cuStreamSynchronize(cuStream));

        bool success = true;
        if constexpr (opType == ReduceOpType::ArgMin ||
                      opType == ReduceOpType::ArgMax) {
            success = resultOnHost.key == refIdx && resultOnHost.value == refValue;
            if constexpr (std::is_same<TypeTraits, Int32Traits>::value)
                printf("  N:%5u, %8d at %6d (ref: %8d at %6d)%s\n", numElements,
                       resultOnHost.value, resultOnHost.key, refValue, refIdx,
                       success ? "" : " NG");
            else if constexpr (std::is_same<TypeTraits, UInt32Traits>::value)
                printf("  N:%5u, %8u at %6d (ref: %8u at %6d)%s\n", numElements,
                       resultOnHost.value, resultOnHost.key, refValue, refIdx,
                       success ? "" : " NG");
            else if constexpr (std::is_same<TypeTraits, Float32Traits>::value)
                printf("  N:%5u, %g at %6d (ref: %g at %6d)%s\n", numElements,
                       resultOnHost.value, resultOnHost.key, refValue, refIdx,
                       success ? "" : " NG");
            else if constexpr (std::is_same<TypeTraits, Int64Traits>::value)
                printf("  N:%5u, %16lld at %6d (ref: %16lld at %6d)%s\n", numElements,
                       resultOnHost.value, resultOnHost.key, refValue, refIdx,
                       success ? "" : " NG");
            else if constexpr (std::is_same<TypeTraits, UInt64Traits>::value)
                printf("  N:%5u, %16llu at %6d (ref: %16llu at %6d)%s\n", numElements,
                       resultOnHost.value, resultOnHost.key, refValue, refIdx,
                       success ? "" : " NG");
        }
        else { // Sum, Min, Max
            if constexpr (std::is_same<TypeTraits, Int32Traits>::value) {
                success = resultOnHost == refValue;
                printf("  N:%5u, %8d (ref: %8d)%s\n", numElements,
                       resultOnHost, refValue,
                       success ? "" : " NG");
            }
            else if constexpr (std::is_same<TypeTraits, UInt32Traits>::value) {
                success = resultOnHost == refValue;
                printf("  N:%5u, %8u (ref: %8u)%s\n", numElements,
                       resultOnHost, refValue,
                       success ? "" : " NG");
            }
            else if constexpr (std::is_same<TypeTraits, Float32Traits>::value) {
                if constexpr (opType == ReduceOpType::Sum) {
                    ValueType error = (resultOnHost - refValue) / refValue;
                    success = std::fabs(error) < 0.001f;
                    printf("  N: %5u, %g (ref: %g), error: %.2f%%%s\n", numElements,
                           resultOnHost, static_cast<double>(refValue), error * 100,
                           success ? "" : " NG");
                }
                else {
                    success = resultOnHost == refValue;
                    printf("  N: %5u, %g (ref: %g)%s\n", numElements, resultOnHost, refValue,
                           success ? "" : " NG");
                }
            }
            else if constexpr (std::is_same<TypeTraits, Int64Traits>::value) {
                success = resultOnHost == refValue;
                printf("  N:%5u, %16lld (ref: %16lld)%s\n", numElements,
                       resultOnHost, refValue,
                       success ? "" : " NG");
            }
            else if constexpr (std::is_same<TypeTraits, UInt64Traits>::value) {
                success = resultOnHost == refValue;
                printf("  N:%5u, %16llu (ref: %16llu)%s\n", numElements,
                       resultOnHost, refValue,
                       success ? "" : " NG");
            }
        }

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    result.finalize();
    values.finalize();

    return allSuccess;
}

template <typename TypeTraits, bool inclusive>
static bool test_DeviceScan(uint32_t MaxNumElements, typename TypeTraits::Type distMin, typename TypeTraits::Type distMax) {
    using ValueType = typename TypeTraits::Type;
    using SumValueType = typename TypeTraits::SumType;
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
                                           values.getDevicePointer(), prefixSums.getDevicePointer(), numElements,
                                           cuStream);
        else
            cubd::DeviceScan::ExclusiveSum(tempStorage.getDevicePointer(), tempStorageSize,
                                           values.getDevicePointer(), prefixSums.getDevicePointer(), numElements,
                                           cuStream);

        CUDADRV_CHECK(cuStreamSynchronize(cuStream));

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

        if constexpr (std::is_same<TypeTraits, Int32Traits>::value)
            printf("  N:%5u, value at the end: %8d (ref: %8d)%s\n", numElements,
                   prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
                   success ? "" : " NG");
        else if constexpr (std::is_same<TypeTraits, UInt32Traits>::value)
            printf("  N:%5u, value at the end: %8u (ref: %8u)%s\n", numElements,
                   prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
                   success ? "" : " NG");
        else if constexpr (std::is_same<TypeTraits, Float32Traits>::value)
            printf("  N:%5u, value at the end: %g (ref: %g)%s\n", numElements,
                   prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
                   success ? "" : " NG");
        else if constexpr (std::is_same<TypeTraits, Int64Traits>::value)
            printf("  N:%5u, value at the end: %16lld (ref: %16lld)%s\n", numElements,
                   prefixSumsOnHost[numElements - 1], refPrefixSums[numElements - 1],
                   success ? "" : " NG");
        else if constexpr (std::is_same<TypeTraits, UInt64Traits>::value)
            printf("  N:%5u, value at the end: %16llu (ref: %16llu)%s\n", numElements,
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

template <typename TypeTraits, RadixSortOpType opType>
static bool test_DeviceRadixSort(uint32_t MaxNumElements, typename TypeTraits::Type distMin, typename TypeTraits::Type distMax) {
    using KeyType = typename TypeTraits::Type;
    using DistributionType = typename TypeTraits::DistributionType;
    using ValueType = uint32_t;
    constexpr bool hasValues = opType == RadixSortOpType::SortPairs || opType == RadixSortOpType::SortPairsDescending;

    DistributionType dist(distMin, distMax);

    std::vector<std::pair<KeyType, ValueType>> refKeyValuePairs(MaxNumElements);

    cudau::TypedBuffer<KeyType> keysA;
    cudau::TypedBuffer<KeyType> keysB;
    keysA.initialize(cuContext, bufferType, MaxNumElements);
    keysB.initialize(cuContext, bufferType, MaxNumElements);
    keysA.setMappedMemoryPersistent(true);
    keysB.setMappedMemoryPersistent(true);

    cudau::TypedBuffer<ValueType> valuesA;
    cudau::TypedBuffer<ValueType> valuesB;
    if constexpr (hasValues) {
        valuesA.initialize(cuContext, bufferType, MaxNumElements);
        valuesB.initialize(cuContext, bufferType, MaxNumElements);
        valuesA.setMappedMemoryPersistent(true);
        valuesB.setMappedMemoryPersistent(true);
    }

    cubd::DoubleBuffer<KeyType> keys(keysA.getDevicePointer(), keysB.getDevicePointer());
    cubd::DoubleBuffer<ValueType> values(valuesA.getDevicePointer(), valuesB.getDevicePointer());

    // JP: 作業バッファーの最大サイズを得る。
    // EN: query the maximum size of working buffer.
    size_t tempStorageSize;
    if constexpr (opType == RadixSortOpType::SortKeys)
        cubd::DeviceRadixSort::SortKeys(nullptr, tempStorageSize,
                                        keys, MaxNumElements);
    else if constexpr (opType == RadixSortOpType::SortKeysDescending)
        cubd::DeviceRadixSort::SortKeysDescending(nullptr, tempStorageSize,
                                                  keys, MaxNumElements);
    else if constexpr (opType == RadixSortOpType::SortPairs)
        cubd::DeviceRadixSort::SortPairs(nullptr, tempStorageSize,
                                         keys, values, MaxNumElements);
    else if constexpr (opType == RadixSortOpType::SortPairsDescending)
        cubd::DeviceRadixSort::SortPairsDescending(nullptr, tempStorageSize,
                                                   keys, values, MaxNumElements);

    // JP: 作業バッファーの確保。
    // EN: allocate the working buffer.
    cudau::Buffer tempStorage;
    tempStorage.initialize(cuContext, bufferType, tempStorageSize, 1);

    printf("DeviceRadixSort::%s, %s / uint32_t:\n", radixSortOpKeywords[static_cast<uint32_t>(opType)], TypeTraits::s_keyword);
    bool allSuccess = true;
    for (int testIdx = 0; testIdx < NumTests; ++testIdx) {
        // JP: 値のセットとリファレンスとしての答えの計算。
        // EN: set values and calculate the reference answer.
        const uint32_t numElements = rng() % (MaxNumElements + 1);
        cudau::TypedBuffer<KeyType> &curKeys = keys.selector ? keysB : keysA;
        cudau::TypedBuffer<ValueType> &curValues = values.selector ? valuesB : valuesA;
        KeyType* keysOnHost = curKeys.map();
        ValueType* valuesOnHost;
        if constexpr (hasValues)
            valuesOnHost = curValues.map();
        for (int i = 0; i < numElements; ++i) {
            KeyType key = dist(rng);
            keysOnHost[i] = key;
            if constexpr (hasValues)
                valuesOnHost[i] = i;
            refKeyValuePairs[i] = std::make_pair(key, i);
        }
        if constexpr (hasValues)
            curValues.unmap();
        curKeys.unmap();
        const auto compareFunc = (opType == RadixSortOpType::SortKeys ||
                                  opType == RadixSortOpType::SortPairs) ?
            [](const std::pair<KeyType, ValueType> &pairA, const std::pair<KeyType, ValueType> &pairB) {
            return pairA.first < pairB.first;
        } :
            [](const std::pair<KeyType, ValueType> &pairA, const std::pair<KeyType, ValueType> &pairB) {
            return pairA.first > pairB.first;
        };
        std::stable_sort(refKeyValuePairs.begin(), refKeyValuePairs.begin() + numElements,
                         compareFunc);

        // JP: ソートの実行。
        // EN: perform sort.
        if constexpr (opType == RadixSortOpType::SortKeys)
            cubd::DeviceRadixSort::SortKeys(tempStorage.getDevicePointer(), tempStorageSize,
                                            keys, numElements, 0, sizeof(KeyType) * 8,
                                            cuStream);
        else if constexpr (opType == RadixSortOpType::SortKeysDescending)
            cubd::DeviceRadixSort::SortKeysDescending(tempStorage.getDevicePointer(), tempStorageSize,
                                                      keys, numElements, 0, sizeof(KeyType) * 8,
                                                      cuStream);
        else if constexpr (opType == RadixSortOpType::SortPairs)
            cubd::DeviceRadixSort::SortPairs(tempStorage.getDevicePointer(), tempStorageSize,
                                             keys, values, numElements, 0, sizeof(KeyType) * 8,
                                             cuStream);
        else if constexpr (opType == RadixSortOpType::SortPairsDescending)
            cubd::DeviceRadixSort::SortPairsDescending(tempStorage.getDevicePointer(), tempStorageSize,
                                                       keys, values, numElements, 0, sizeof(KeyType) * 8,
                                                       cuStream);

        CUDADRV_CHECK(cuStreamSynchronize(cuStream));

        cudau::TypedBuffer<KeyType> &sortedKeys = keys.selector ? keysB : keysA;
        cudau::TypedBuffer<ValueType> &sortedValues = values.selector ? valuesB : valuesA;
        KeyType* sortedKeysOnHost = sortedKeys.map();
        ValueType* sortedValuesOnHost;
        if constexpr (hasValues)
            sortedValuesOnHost = sortedValues.map();
        bool success = true;
        for (int i = 0; i < numElements; ++i) {
            const std::pair<KeyType, ValueType> &refPair = refKeyValuePairs[i];
            success &= sortedKeysOnHost[i] == refPair.first;
            if constexpr (hasValues)
                success &= sortedValuesOnHost[i] == refPair.second;
            if (!success)
                break;
        }
        if constexpr (hasValues)
            sortedValues.unmap();
        sortedKeys.unmap();
        printf("  N:%5u%s\n", numElements, success ? "" : " NG");

        allSuccess &= success;
    }
    printf("\n");

    tempStorage.finalize();
    if constexpr (hasValues) {
        valuesB.finalize();
        valuesA.finalize();
    }
    keysB.finalize();
    keysA.finalize();

    return allSuccess;
}
