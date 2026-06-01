---
layout: post
title: "CUDA & Triton"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-05-26
tags:
  - Korean
  - C++
  - CUDA C++
  - CUDA
  - Python
  - Triton
  - GPU
language: ko
summary: "NVIDIA GPU computing stack으로서의 CUDA와 custom GPU kernel compiler로서의 Triton을 정리"
math: true
comments: true
comment_id: "study-system-optimization-cuda-triton"
permalink: /study/system-optimization/cuda-triton/
---

지금까지 CUDA에 대해서 어렴풋이 느낌만 가지고 "CUDA 설치해야 한다", "CUDA Kernel 짰다", "PyTorch CUDA 버전이 무엇이다"와 같은 말을 썼다. 또한 [VLA를 real-time speed로 inference하는 논문](https://arxiv.org/abs/2510.26742){:target="_blank" rel="noopener noreferrer"}을 읽으면서 OpenAI Triton이라는 것에 대해서 처음 접했고, 지금 읽고 있는 [AI Systems Performance Engineering](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/){:target="_blank" rel="noopener noreferrer"} 책에서도 여러 번 등장하는 것을 보았다. 이번에 이것들에 대해서 확실하게 정리하고 넘어가야겠다. CUDA C++와 Triton 문법에 대한 내용은 [다른 카테고리](/study/inference-systems/)에서 다룬다.

## **Introduction**
CUDA는 NVIDIA GPU를 CPU 옆의 범용 병렬 가속기로 사용하기 위한 **전체 플랫폼**이다. 단순히 cuda라는 라이브러리 하나가 아니라, 다음을 모두 포함하는 스택이다:

1. **CUDA programming model**: grid/block/thread/warp/stream 같은 실행 모델
2. **CUDA C++ language extensions**: `__global__`, `__device__`, `threadIdx`, `blockIdx` 등
3. **CUDA compiler/toolchain**: `nvcc`, `ptxas`, PTX, CUBIN, fatbinary, NVRTC
4. **CUDA Runtime API**: `cudaMalloc`, `cudaMemcpyAsync`, `cudaLaunchKernel`, stream/event/graph 관리
5. **CUDA Driver API**: `libcuda.so`, context/module/kernel launch 등 더 낮은 레벨 API
6. **GPU-accelerated libraries**: cuBLAS/cuBLASLt, cuDNN, [NCCL](/study/system-optimization/nccl-nixl/), cuFFT, cuRAND, cuSPARSE 등
7. **profiling/debugging tools**: Nsight Systems, Nsight Compute, CUPTI, Compute Sanitizer, cuda-gdb
8. **NVIDIA GPU hardware execution layer**: SM, warp scheduler, CUDA cores, Tensor Cores, shared memory, registers, L2, HBM, copy engine 등

딥러닝에서 `model.cuda()`나 `tensor.to("cuda")`를 쓴다는 것은 대부분 CUDA C++를 직접 작성한다는 뜻이 아니라, PyTorch/TensorRT/ONNX Runtime 같은 프레임워크가 내부적으로 CUDA Runtime/Driver + cuBLAS/cuDNN/[NCCL](/study/system-optimization/nccl-nixl/)/custom CUDA kernels/Triton kernels를 호출한다는 뜻이다.

OpenAI가 공개한 Triton, 현재는 보통 Triton language/compiler 또는 triton-lang으로 부르는 도구는 CUDA 전체를 대체하는 것이 아니다. 이 글의 NVIDIA GPU 맥락에서는 CUDA C++/NVCC로 custom GPU kernel을 직접 작성하던 부분을 Python 기반 tiled programming model과 JIT compiler로 대체하거나 보완하는 계층에 가깝다. Triton kernel은 PyTorch CUDA tensor pointer를 입력으로 받아 실행되는 경우가 많고, 최종적으로는 NVIDIA driver와 GPU execution stack 위에서 실행된다.

### **Terminologies**

| 용어                   | 정확한 의미                                                                       | 예시                                                           |
| -------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **CUDA Platform**    | NVIDIA GPU를 범용 병렬 컴퓨팅 장치로 쓰기 위한 전체 플랫폼                                       | programming model, APIs, compiler, libraries, tools          |
| **CUDA Toolkit**     | 개발자가 설치하는 SDK(Software Development Kit) 묶음                                                             | `nvcc`, headers, `libcudart`, cuBLAS, cuFFT, Nsight, samples |
| **CUDA Runtime API** | C/C++에서 가장 흔히 쓰는 고수준 CUDA API                                                | `cudaMalloc`, `cudaMemcpyAsync`, stream, event, graph        |
| **CUDA Driver API**  | NVIDIA driver가 노출하는 더 낮은 레벨 API                                              | `cuInit`, `cuCtxCreate`, `cuModuleLoad`, `cuLaunchKernel`    |
| **CUDA C++**         | CUDA programming model을 C++ extension으로 표현하는 언어                                     | `__global__ void kernel(...)`                                |
| **CUDA kernel**      | GPU device에서 병렬 실행되는 함수                                                      | matmul kernel, layernorm kernel, softmax kernel              |
| **CUDA library**     | NVIDIA가 최적화해서 제공하는 GPU 연산 라이브러리                                               | cuBLAS, cuDNN, [NCCL](/study/system-optimization/nccl-nixl/)                                          |
| **CUDA version**     | 문맥에 따라 driver compatibility, toolkit version, PyTorch build CUDA version을 의미 | `nvidia-smi`, `nvcc --version`, `torch.version.cuda`         |

중요한 점은 CUDA Runtime API는 CUDA Driver API 위에 구현되어 있다는 것이다. 실무적으로는 Runtime API를 더 자주 보게 된다. Runtime API는 primary context initialization, module management 등을 암묵적으로 처리해주므로 코드가 단순하다. 반대로 Driver API는 context와 module loading을 명시적으로 제어할 수 있어서 compiler/runtime/framework 구현에서 자주 등장한다.

## **CUDA Stack Overview**

딥러닝 모델을 PyTorch로 실행할 때의 CUDA stack은 대략 다음과 같다:

```text
[User Code]
  Python script / C++ app
  model(x), loss.backward(), optimizer.step()
  torch.compile(), TensorRT engine, custom Triton kernel

[Framework / Compiler Layer]
  PyTorch Dispatcher / ATen / Autograd
  TorchInductor / AOTAutograd / Triton JIT
  TensorRT Builder + Runtime
  ONNX Runtime CUDA EP, etc.

[CUDA-accelerated Libraries]
  cuBLAS / cuBLASLt    -> GEMM, batched GEMM, matmul
  cuDNN                -> DNN primitives: convolution, attention, normalization, softmax, pooling, pointwise/fused ops
  [NCCL](/study/system-optimization/nccl-nixl/)                 -> multi-GPU collective communication
  cuRAND/cuFFT/etc.    -> random, FFT, sparse, etc.
  Custom CUDA kernels  -> PyTorch native kernels, FlashAttention, fused ops
  Triton kernels       -> user-written or compiler-generated fused kernels

[CUDA API Layer]
  CUDA Runtime API: cudaMalloc, cudaMemcpyAsync, cudaLaunchKernel, streams, events
  CUDA Driver API:  contexts, modules, PTX/CUBIN loading, low-level launch

[NVIDIA Driver]
  user-mode driver + kernel-mode driver
  libcuda.so / nvidia.ko
  GPU memory management, command submission, context scheduling, JIT

[GPU Hardware]
  HBM/global memory, L2 cache, SMs, warp schedulers
  registers, shared memory, L1
  CUDA cores, Tensor Cores, special function units
  copy engines, NVLink/PCIe
```

`y = model(x)`를 실행한다는 것은 내부적으로 Python → PyTorch dispatcher → CUDA library/custom kernel selection → CUDA Runtime/Driver → GPU command queue → SM scheduling → warp execution → memory hierarchy access 순서로 내려가는 것을 의미한다.

## **CUDA Programming Model**
CUDA C++를 다룰 때 접하는 내용들인데, CUDA의 핵심 계층은 다음과 같다:

```text
Grid
 ├── Block 0
 │    ├── Warp 0: 32 threads
 │    ├── Warp 1: 32 threads
 │    └── ...
 ├── Block 1
 │    ├── Warp 0
 │    └── ...
 └── ...
```

| 개념              | 의미                                        | 성능 엔지니어링 관점                                      |
| --------------- | ----------------------------------------- | ------------------------------------------------ |
| **Thread**      | 가장 작은 CUDA programming model상의 실행 단위      | scalar element 하나 또는 작은 vector fragment 담당       |
| **Warp**        | 보통 32개 thread 묶음                          | SIMT 실행 단위. branch divergence가 성능에 중요            |
| **Block / CTA** | thread들의 묶음                               | 하나의 SM에 배정되어 shared memory와 `__syncthreads()` 공유 |
| **Grid**        | block들의 전체 집합                             | 하나의 kernel launch가 생성하는 전체 병렬 작업                 |
| **SM**          | Streaming Multiprocessor                  | block/warp가 실제로 schedule되는 GPU compute unit      |
| **Stream**      | GPU work queue                            | kernel/memcpy 순서와 overlap을 제어                    |
| **Event**       | stream상의 timestamp/synchronization marker | latency 측정과 dependency 제어                        |
| **CUDA Graph**  | 반복 launch sequence capture/replay         | CPU launch overhead 감소                           |

GPU hardware는 block을 SM에 배치하고, SM 내부 warp scheduler가 ready warp를 선택해서 instruction을 발행한다. [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/){:target="_blank" rel="noopener noreferrer"}는 latency hiding과 occupancy가 active warps 수, execution parameters, register/shared-memory resource constraints에 의해 결정된다고 설명한다.

## **CUDA memory hierarchy**
이 또한 CUDA C++를 다룰 때 자주 접하는 개념이다. CUDA 성능은 대부분 연산량보다 memory movement에 의해 결정된다. 딥러닝에서도 matmul은 Tensor Core를 잘 쓰면 compute-bound가 될 수 있지만, layernorm, softmax, activation, embedding, optimizer update, small-batch inference는 memory-bound가 되기 쉽다. CUDA memory hierarchy를 top-down으로 보면 다음과 같다:

```text
CPU DRAM
  │
  │ PCIe / NVLink
  ▼
GPU HBM / Global Memory
  │
  ▼
L2 Cache
  │
  ▼
SM-local memory system
  ├── L1 / Shared Memory
  ├── Registers
  └── Local Memory logical space, physically often global memory
```

| Memory                  | Scope                      | 속도/특징                                         | 예시                                  |
| ----------------------- | -------------------------- | --------------------------------------------- | ----------------------------------- |
| **Register**            | thread private             | 가장 빠름, 개수가 제한됨                              | accumulator, loop variable          |
| **Shared memory**       | block private              | SM-local SRAM, block 내부 협업용                   | tiled GEMM의 A/B tile                |
| **L1 cache**            | SM-local                   | load/store caching                            | global load locality                |
| **L2 cache**            | GPU 전체 공유                 | SM 간 공유되는 cache                               | weights reuse                       |
| **Global memory / HBM** | device global              | 크지만 느림                                        | tensor storage                      |
| **Local memory**        | thread-local logical space | 이름과 달리 physical location은 global memory일 수 있음 | register spill, dynamic local array |
| **Pinned host memory**  | CPU memory                 | async H2D/D2H copy에 유리                        | DataLoader pinned batch             |
| **Unified memory**      | managed memory             | CPU/GPU address space 편의성                     | prototyping, oversubscription       |

## **CUDA Compilation: CUDA C++ to GPU Instruction**

CUDA C++ kernel은 대략 다음 순서로 변환된다:

```text
CUDA C++ source (.cu)
  │
  ▼
nvcc
  ├── host code  -> system C++ compiler(gcc/clang/msvc)
  └── device code -> NVCC device compilation
                    ├── PTX intermediate code
                    └── CUBIN native binary
  │
  ▼
fatbinary
  │
  ▼
executable / shared library
  │
  ▼
runtime loading by CUDA Runtime/Driver
  │
  ▼
GPU native instructions executed on SMs
```

| 산출물           | 의미                                                    |
| ------------- | ----------------------------------------------------- |
| **PTX**       | NVIDIA GPU용 virtual ISA / intermediate representation |
| **CUBIN**     | 특정 GPU architecture용 native device binary             |
| **fatbinary** | 여러 architecture용 PTX/CUBIN을 담는 container              |

PTX는 NVIDIA GPU용 virtual ISA이자 intermediate representation이다. 고수준 CUDA/Triton/CUTLASS 계열 컴파일러가 PTX를 생성하면, target GPU architecture에 맞는 native instruction으로 변환된다. PTX만 포함된 경우에는 runtime에 NVIDIA driver가 PTX를 native binary로 JIT compile할 수 있다.

## **CUDA Installation**
CUDA를 설치하는 것은 실제로는 여러 층을 다루는 일이다.

### **NVIDIA Driver**
GPU를 compute device로 쓰기 위해 반드시 NVIDIA driver가 필요하다. NVIDIA Driver는 다음을 제공한다:

* OS kernel module
* user-mode driver library, 예: Linux의 `libcuda.so`
* GPU context/memory/command submission 관리
* CUDA Driver API entry point
* PTX JIT, module loading, scheduling support

`nvidia-smi`가 작동한다는 것은 보통 driver와 GPU device discovery가 된다는 뜻이다. 하지만 `nvidia-smi`에 보이는 CUDA version은 설치된 CUDA Toolkit 버전이 아니라 현재 설치된 NVIDIA driver가 지원하는 최대 CUDA API/Runtime 호환 버전(현재 NVIDIA driver가 지원할 수 있는 CUDA compatibility 수준)이다. 실제 개발 toolkit version은 `nvcc --version`, PyTorch build CUDA version(PyTorch binary가 포함/기대하는 CUDA 런타임 버전)은 `torch.version.cuda`로 따로 봐야 한다. 즉 `nvidia-smi`의 CUDA Version, `nvcc --version`의 CUDA Toolkit version, `torch.version.cuda`의 PyTorch build CUDA version은 서로 다른 값일 수 있다. 실행이 되려면 핵심적으로 NVIDIA driver가 해당 CUDA runtime/library가 요구하는 minimum driver version 이상이어야 한다. 보통 새 driver는 과거 CUDA runtime과의 backward compatibility를 제공하지만, 구체적인 호환성은 CUDA compatibility matrix와 framework build 조건을 확인해야 한다.

### **CUDA Toolkit**
CUDA C++를 직접 컴파일하거나 native extension을 빌드하려면 CUDA Toolkit이 필요하다. CUDA Toolkit은 CUDA application을 만들고, 빌드하고, 실행하기 위한 compiler, headers, runtime library, GPU-accelerated libraries, debugging/optimization tools 등을 포함한다. 다만 cuDNN, [NCCL](/study/system-optimization/nccl-nixl/), TensorRT 같은 딥러닝 관련 구성요소는 CUDA 위에서 동작하지만, 설치 방식상 CUDA Toolkit 자체와 별도로 배포되거나 PyTorch/TensorRT package에 함께 포함되는 경우가 많다.

CUDA Toolkit에는 보통 다음과 같은 것들이 들어있다:

```text
/usr/local/cuda-<version>/
  bin/
    nvcc
    cuda-gdb
    compute-sanitizer
    ...
  include/
    cuda_runtime.h
    cublas_v2.h
    ...
  lib64/
    libcudart.so
    libcublas.so
    libcufft.so
    ...
  extras/
    CUPTI/
  nsight systems / nsight compute tools

별도 또는 framework package를 통해 함께 설치/배포되는 경우가 많은 구성요소:
  cuDNN, NCCL, TensorRT 등
```

환경 변수는 보통 다음과 같이 설정한다:

```shell
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

PyTorch만 쓸 때는 Toolkit이 항상 필요하지는 않을 수도 있다. PyTorch를 pip/conda로 설치할 때 CUDA-enabled binary를 설치하면, 많은 경우 PyTorch wheel/conda package 안에 필요한 CUDA runtime 계열 라이브러리들이 함께 들어온다. 이 경우 system-wide CUDA Toolkit 없이도 PyTorch CUDA tensor 실행은 가능하다. 다만 다음 경우에는 Toolkit이 필요할 수 있다.

* CUDA C++ extension을 직접 빌드할 때
* custom CUDA kernel을 `.cu`로 컴파일할 때
* 특정 system CUDA library를 직접 link할 때
* `nvcc`, `cuda-gdb`, 일부 profiling/dev tool이 필요할 때
* source build를 할 때

### **CUDA Runtime API & Driver API**
Runtime API는 implicit initialization, context management, module management를 처리해주기 때문에 코드가 단순하다. 반대로 Driver API는 context와 module loading을 더 명시적으로 제어할 수 있다.

CUDA Runtime API를 쓰는 C++ 코드는 보통 다음과 같다:

```cpp
cudaSetDevice(0);

float* x_dev;
cudaMalloc(&x_dev, n * sizeof(float));

cudaMemcpyAsync(
    x_dev,
    x_host,
    n * sizeof(float),
    cudaMemcpyHostToDevice,
    stream
);

my_kernel<<<grid, block, shared_mem_bytes, stream>>>(x_dev, n);

cudaStreamSynchronize(stream);
```

위 코드에서 각 호출의 의미는 다음과 같다:

| 호출                      | CUDA stack에서 하는 일                                  |
| ----------------------- | -------------------------------------------------- |
| `cudaSetDevice(0)`      | 현재 host thread의 target CUDA device 설정              |
| `cudaMalloc`            | GPU device memory allocation 요청                    |
| `cudaMemcpyAsync`       | host-device 또는 device-device copy를 stream에 enqueue |
| `kernel<<<...>>>`       | kernel launch command를 stream에 enqueue             |
| `cudaStreamSynchronize` | 해당 stream의 모든 이전 작업 완료까지 CPU thread 대기             |

Driver API로 같은 일을 하는 코드는 다음과 같다:

```cpp
cuInit(0);
cuDeviceGet(&dev, 0);
cuCtxCreate(&ctx, 0, dev);
cuMemAlloc(&dptr, bytes);
cuModuleLoad(&module, "kernel.cubin");
cuModuleGetFunction(&func, module, "my_kernel");
cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem, stream, args, nullptr);
```

PyTorch에서 CUDA가 어떻게 쓰이는지에 대한 예시는 [여기서](/study/system-optimization/torch-cuda-example/) 다룬다.

## **Triton은 CUDA의 어느 부분에 대응되는가?**

```text
기존 방식:
  CUDA C++ kernel source
    -> nvcc
    -> PTX/CUBIN
    -> CUDA Runtime/Driver launch
    -> GPU

Triton 방식:
  Python @triton.jit kernel
    -> Triton compiler
    -> GPU code, e.g. PTX/CUBIN path
    -> CUDA Runtime/Driver launch
    -> GPU
```

즉, Triton은 다음을 대체하거나 보완한다:

* CUDA C++ `__global__` kernel 작성
* manual thread indexing
* 일부 shared memory/register tiling boilerplate
* NVCC-based custom extension workflow
* 일부 kernel fusion 작업

하지만 다음은 여전히 필요하다:

* NVIDIA driver
* GPU hardware
* CUDA-compatible execution environment
* PyTorch tensor/device memory
* CUDA stream semantics
* 경우에 따라 cuBLAS/cuDNN/[NCCL](/study/system-optimization/nccl-nixl/) 같은 libraries

CUDA C++과 Triton의 구성 요소를 대응시키면 다음과 같다:

| CUDA C++              | Triton                                                    | 설명                                    |
| --------------------- | --------------------------------------------------------- | ------------------------------------- |
| `.cu` file            | Python function with `@triton.jit`                        | kernel source 위치                      |
| `__global__`          | `@triton.jit`                                             | GPU kernel 표시                         |
| `blockIdx.x`          | `tl.program_id(0)`                                        | tile/program id                       |
| `threadIdx.x`         | `tl.arange(...)` + compiler mapping                       | 명시 thread보다 vectorized tile 중심        |
| `cudaMalloc`          | PyTorch tensor allocation 사용 가능                           | Triton은 보통 PyTorch tensor pointer를 받음 |
| `cudaMemcpyAsync`     | PyTorch `.to("cuda")` 등                                   | data movement는 framework/CUDA가 담당     |
| `__shared__`          | compiler-managed SRAM usage, block pointers, dot patterns | low-level control은 CUDA C++가 더 직접적    |
| `__syncthreads()`     | Triton compiler/program semantics                         | 명시적 barrier control은 제한적/추상화          |
| `wmma/mma.sync`       | `tl.dot`                                                  | matmul/tensor core path로 lowering 가능  |
| `nvcc`                | Triton compiler/JIT                                       | compile path                          |
| C++ extension binding | Python direct call                                        | 개발 편의성 차이                             |

딥러닝 training/inference에서 Triton이 유리한 경우는 다음과 같다:

| 병목 유형                 | 예시                               | Triton이 유리한 이유                        |
| --------------------- | -------------------------------- | ------------------------------------- |
| Pointwise chain       | bias + GELU + dropout + residual | fusion으로 HBM 왕복 감소                    |
| Reduction             | layernorm, RMSNorm, softmax      | row/block 단위 reduction을 tile로 표현하기 쉬움 |
| Non-standard layout   | MoE routing, ragged tensor       | cuBLAS/cuDNN으로 안 맞는 shape 처리          |
| Attention-like op     | FlashAttention variants          | SRAM tiling, custom memory pattern    |
| Small kernel overhead | many tiny ops                    | fusion으로 launch 수 감소                  |
| Research kernel       | 새로운 논문 kernel                    | CUDA C++보다 iteration 속도 빠름            |

반대로 CUDA C++이 더 적합한 경우도 존재한다:

| 상황                                 | CUDA C++가 더 유리한 이유                                 |
| ---------------------------------- | -------------------------------------------------- |
| 매우 low-level warp scheduling 제어 필요 | warp primitive, inline PTX, 특수 barrier를 직접 제어      |
| 복잡한 shared memory layout 최적화       | bank conflict, cp.async, TMA 등을 세밀하게 제어            |
| production-grade library 수준 GEMM   | CUTLASS/cuBLASLt/CUDA C++ template metaprogramming |
| non-DNN GPU programming            | graph algorithm, simulation, custom data structure |
| debugging/profiling integration    | C++ toolchain과 Nsight workflow가 더 직접적              |

둘 중 하나를 선택하는 기준은 다음과 같이 잡을 수 있을 것 같다:

```text
PyTorch eager/compile를 이해한다
  -> 어떤 op가 cuBLAS/cuDNN/NCCL/custom kernel로 가는지 본다
  -> Nsight Systems로 timeline을 본다
  -> Nsight Compute로 kernel bottleneck을 본다
  -> memory-bound/fusion 문제면 Triton으로 빠르게 custom kernel 작성
  -> Tensor Core/GEMM 극한 최적화면 cuBLASLt/CUTLASS/CUDA C++까지 내려감
```


{% include comments.html %}
