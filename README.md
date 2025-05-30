# CUBd: 

[CUB](https://nvlabs.github.io/cub/)はCUDAにおいてGPGPUプログラミングする際に非常に便利なReductionやSortといった並列計算プリミティブ、構成要素を提供しています。
しかし欠点として、デバイス範囲のプリミティブを使用するためにCUBのヘッダーファイルをincludeすると、CUDA特有の予約語などがソースコードに含まれてしまうことがあります。
これはCUBのヘッダーをincludeしたファイルはNVCC経由でコンパイルする必要があることを意味します。
**CUBd**は、CUBのincludeを隠蔽することを目的にしたシンプルなライブラリ(の実装例)です。
使用方法は下記コード例にも示すように、namespaceがcubからcubdに変わったことを除き、オリジナルのCUBと全く同じにしています。\
\
[CUB](https://nvlabs.github.io/cub/) provides very useful parallel compute primitives like reduction and sort for CUDA GPGPU programming.
However, it has a drawback that including CUB's header files to use device-wide primitives brings CUDA-specific reserved words into the source code.
This means that those files including CUB's header needs to be compiled via NVCC.
**CUBd** is a simple library (example) to isolate the include of the CUB's headers.
As code example below shows, you can use it in the same way as CUB except for that namespace changes from cub to cubd.

- libcubd
- libcubd_static
- direct_use\
  cubdのソースコードを直接使用する例。\
  Example usage of direct use of the cubd source code.
- dynamic_link\
  libcubdによって生成される動的リンクライブラリの使用例。\
  Example usage of the dynamic link library generated from libcubd.
- static_link\
  libcubd_staticによって生成される静的リンクライブラリの使用例。\
  Example usage of the static link library generated from libcubd_static.

## コード例 / Code Example
CUBにおけるDevice-wide Radix Sort\
Device-wide Radix Sort in the CUB
```cpp
cub::DoubleBuffer<uint64_t> keysOnDevice(keysOnDeviceA, keysOnDeviceB);
cub::DoubleBuffer<uint32_t> valuesOnDevice(valuesOnDeviceA, valuesOnDeviceB);

cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageSize,
                                keysOnDevice, valuesOnDevice, numElements);
```

CUBd equivalent
```cpp
cubd::DoubleBuffer<uint64_t> keysOnDevice(keysOnDeviceA, keysOnDeviceB);
cubd::DoubleBuffer<uint32_t> valuesOnDevice(valuesOnDeviceA, valuesOnDeviceB);

cubd::DeviceRadixSort::SortPairs(tempStorage, tempStorageSize,
                                 keysOnDevice, valuesOnDevice, numElements);
```

## 注意 / Note
あくまでCUBdは実装「例」であり、CUBが実装している全ての(デバイス範囲)のプリミティブをサポートしていません。
しかしcubd.cuに必要なテンプレートの明示的インスタンス化を書き足すことで簡単にCUBdがサポートする処理を追加することができます。\
\
CUBd is just an implementation "example" and doesn't support all the (device-wide) primitives CUB provides.
However, it is easy to add a primitive which CUBd supports by writing explicit template instantiation to cubd.cu.

## 動作環境 / Confirmed Environment
現状以下の環境で動作を確認しています。\
I've confirmed that the program runs correctly on the following environment.

* Windows 11 (24H2) & Visual Studio 2022 (17.13.6)
* Ryzen 9 7950X, 64GB, RTX 4080 16GB
* NVIDIA Driver 576.02

動作させるにあたっては以下のライブラリが必要です。\
It requires the following libraries.

* CUDA 12.8.1 (cubd might also be built with bit older CUDA version 11.0-)\
  Note that CUDA has compilation issues with Visual Studio 2022 17.10.0.

## ライセンス / License
Released under the Apache License, Version 2.0 (see [LICENSE.md](LICENSE.md))

----
2025 [@Shocker_0x15](https://twitter.com/Shocker_0x15), [@bsky.rayspace.xyz](https://bsky.app/profile/bsky.rayspace.xyz)
