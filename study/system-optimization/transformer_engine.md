---
layout: default
title: "NVIDIA Transformer Engine"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-05-28
tags:
  - Korean
  - CUDA
  - GPU
language: ko
summary: "Transformer 기반 LLM의 GEMM/attention 연산을 FP8/NVFP4 Tensor Core 경로로 가속하는 NVIDIA Transformer Engine hardware-software stack 정리"
math: true
comments: true
comment_id: "study-system-optimization-transformer-engine"
permalink: /study/system-optimization/transformer-engine/
---

# **NVIDIA Transformer Engine**

## **Hardware & Software**

NVIDIA에서 Transformer Engine(TE)이라는 말은 두가지로 사용된다:

| 구분                      | 의미                                                                                       | 핵심 역할                                        |
| ----------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------- |
| **TE hardware**         | GPU 안의 Tensor Core + 저정밀도 데이터 타입 지원 + scaling/casting datapath + TMEM/TMA 같은 데이터 공급 구조   | FP8/FP4/NVFP4 같은 낮은 precision으로 GEMM을 빠르게 수행 |
| **TE software library** | `transformer_engine` 라이브러리, PyTorch/JAX/C++ API, fused Transformer layer, FP8/FP4 recipe | 모델 코드에서 TE hardware를 쉽게 쓰도록 abstraction 제공   |

즉 TE hardware는 TE software가 target으로 삼는 GPU 하드웨어 기능들이다. 세대별로 보면 Hopper에서는 FP8 Transformer Engine이 본격적으로 중요해졌고, Blackwell에서는 MXFP8과 NVFP4 같은 더 공격적인 저정밀도 format과 TMEM이 추가되면서 TE hardware path가 더 강화되었다. 따라서 FP8은 Hopper/Ada/Blackwell 계열에서 생각할 수 있지만, NVFP4/FP4 중심 설명은 Blackwell 계열을 기준으로 이해하는 것이 안전하다.

## **Transformer/LLM**

Transformer 기반 LLM의 대부분의 연산은 결국 $Y = XW$와 같은 큰 행렬곱이다. 예를 들어 LLM 하나의 layer 안에는 다음과 같은 matrix-heavy 연산이 반복된다:

| Transformer 구성 요소 | 실제 연산                          |
| ----------------- | ------------------------------ |
| Q/K/V projection  | `X @ Wq`, `X @ Wk`, `X @ Wv`   |
| attention score   | `Q @ K^T`                      |
| attention output  | `softmax(QK^T / sqrt(d_k)) @ V` |
| MLP / FFN         | `X @ W1`, activation, `H @ W2` |
| MoE expert        | 선택된 expert별 대형 GEMM            |

즉 LLM의 성능은 상당 부분 GEMM throughput, memory bandwidth, activation/weight movement, precision format에 의해 결정된다. Tensor Core는 이런 MMA(matrix multiply-accumulate) 연산을 일반 CUDA core보다 훨씬 빠르게 처리하는 전용 유닛이다. 따라서 TE hardware를 활용한다는 말은 "`nn.Linear`, attention GEMM, FFN/MLP GEMM과 같은 Transformer 핵심 행렬 연산을 framework의 기본 dense matmul 경로에만 의존하지 않고, **Tensor Core + 저정밀도 format(FP8, Blackwell의 경우 MXFP8/NVFP4) + scaling/casting-aware fused kernel** 경로로 실행하겠다"라는 의미이다.

## **Key Idea**

핵심은 **낮은 precision으로 더 많은 연산을 더 적은 memory traffic으로 처리하겠다**는 것이다. 각 data format별로 값 하나가 차지하는 크기는 다음과 같다:

| Format      | 값 하나의 크기 | FP32 대비 weight/activation traffic |
| ----------- | ------- | -------------------------------- |
| FP32        |  4 bytes |                                1× |
| BF16/FP16   |  2 bytes |                               1/2 |
| FP8         |   1 byte |                               1/4 |
| FP4 / NVFP4 | 0.5 byte |                               1/8 |

위 표는 raw value payload 기준이다. 실제 FP8/FP4/NVFP4 경로에서는 scaling factor, amax history, block scale 같은 metadata가 추가될 수 있으므로 실제 memory traffic 감소율은 kernel 구현과 scaling recipe에 따라 달라진다.

LLM inference/training에서는 GPU가 계산을 못 해서 느린 경우도 있지만, 실제로는 HBM에서 weight/activation을 가져오느라 느린 경우가 많다. 값 하나의 bit-width를 줄이면 같은 HBM bandwidth로 더 많은 operand를 공급할 수 있고, 해당 precision을 hardware Tensor Core가 native로 지원하는 경우 더 높은 MMA throughput을 얻을 수 있다.

하지만 전부 FP8/FP4로 바꾸는 것이 아니다. 낮은 precision을 쓰되, 정확도가 중요한 부분은 더 높은 precision을 유지해야 한다. 예를 들면 다음과 같다:

| 부분                            | 흔한 선택                              |
| ----------------------------- | ---------------------------------- |
| 대형 GEMM input                 | FP8, NVFP4, BF16                   |
| accumulation                  | FP16/BF16/FP32                     |
| LayerNorm, softmax, reduction | 보통 FP16/BF16/FP32 유지               |
| master weight                 | training에서는 보통 higher precision 유지 |
| inference weight              | FP8/FP4 quantized weight 가능        |

## **TMEM/TMA**
Blackwell에서는 각 SM에 TMEM(Tensor Memory)이라는 Tensor Core 인접 on-chip storage가 추가되었다. TMEM은 Tensor Core 연산의 intermediate result와 accumulator reuse를 돕는 공간으로 이해하면 된다. 반면 TMA(Tensor Memory Accelerator)는 global memory의 multi-dimensional tile을 shared memory로 효율적으로 옮기기 위한 비동기 data movement engine이다. 즉 TMEM은 Tensor Core 근처의 저장/재사용 공간이고, TMA는 HBM/global memory에서 shared memory로 tile을 공급하는 이동 경로다. Tensor Core 연산의 intermediate result와 accumulator reuse를 돕기 위한 공간이다. [AI Systems Performance Engineering](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/){:target="_blank" rel="noopener noreferrer"} 기준으로 TMEM은 256 KB per-SM SRAM buffer이다.

일반적인 PyTorch application code에서 TMEM을 `cudaMalloc`처럼 직접 allocate해서 쓰는 것은 아니다. 보통 cuBLASLt, CUTLASS, Transformer Engine, TensorRT-LLM 같은 library/kernel stack이 TMEM을 활용한다. 다만 CUTLASS/CuTe 같은 low-level kernel programming layer에서는 Blackwell TMEM이 명시적인 data locale로 노출될 수 있다.

| 구성 요소                   | 역할                                                       |
| ----------------------- | -------------------------------------------------------- |
| **HBM**                 | weight/activation 원본 저장                                  |
| **Asynchronous copy path** | `cp.async` 또는 TMA를 이용해 global memory의 tile을 shared memory로 미리 가져온다. |
| **Shared Memory**       | Tensor Core operand staging                              |
| **TMEM**                | Tensor Core accumulator/intermediate result 저장           |
| **Tensor Core MMA**     | FP8/FP4/BF16/TF32 matrix multiply-accumulate             |
| **TE software/runtime** | scaling, casting, fused kernel, Transformer layer API 제공 |

`cp.async`와 TMA(Tensor Memory Accelerator)는 아래와 같이 구분할 수 있다:
* `cp.async`: Ampere 이후에서 많이 쓰는 thread-level asynchronous copy 계열.
* TMA: Hopper 이후에서 본격적으로 쓰이는 bulk tensor tile movement engine. multidimensional tile을 global memory에서 shared memory로 옮기는 데 특화되어 있다.

Performance 관점에서 핵심은 data movement와 compute를 overlap시키는 것이다:

1. 다음 tile을 HBM에서 shared memory로 미리 가져온다.
2. 현재 tile은 Tensor Core에서 MMA한다.
3. Blackwell TMEM을 활용하는 kernel에서는 partial sum/intermediate state를 TMEM 쪽에 유지해 reuse를 높인다.
4. 가능한 한 HBM 왕복을 줄인다.
5. Tensor Core가 놀지 않게 계속 tile을 공급한다.


## **주의할 점: TE를 사용한다고 무조건 속도가 빨라지지는 않는다**

### **TE의 gain이 높은 경우**

| 조건                        | 이유                                       |
| ------------------------- | ---------------------------------------- |
| GEMM-heavy LLM layer      | Tensor Core 사용률이 높음                      |
| batch/sequence가 충분히 큼     | quantization/scaling overhead가 amortize됨 |
| GPU가 해당 저정밀도 format을 hardware 지원 | FP8: Hopper/Ada/Blackwell, MXFP8/NVFP4: Blackwell |
| accuracy가 FP8/FP4를 견딤     | scaling/calibration 후 품질 유지              |
| fused kernel 사용 가능        | HBM 왕복 감소                                |

### **TE의 gain이 낮은 경우**

| 조건                         | 문제                                         |
| -------------------------- | ------------------------------------------ |
| small batch, tiny GEMM     | Tensor Core launch/scale overhead가 상대적으로 큼 |
| memory irregular workload  | GEMM이 아니라 random access가 bottleneck        |
| softmax/layernorm 중심 병목    | 단순 Tensor Core만으로 해결 안 됨                   |
| CPU/data loader bottleneck | GPU TE를 써도 GPU가 기다림                        |
| unsupported GPU            | FP4/NVFP4 경로 사용 불가                         |
| calibration 실패             | accuracy loss 발생                           |

## **Profiler에서 확인할 것**

TE를 사용한다고 주장하려면 실제 profiler로 확인해야 한다. 단순히 `transformer_engine` module을 import했다고 해서 bottleneck이 사라지는 것은 아니다.

| Tool | 확인할 것 |
| ---- | -------- |
| **PyTorch Profiler** | `Linear`, `matmul`, attention block이 어떤 CUDA kernel로 내려가는지 확인 |
| **Nsight Systems** | kernel launch gap, CPU-GPU synchronization, TE/cuBLASLt/cuDNN kernel timeline 확인 |
| **Nsight Compute** | Tensor Core utilization, memory throughput, roofline, stall reason 확인 |
| **DCGM / nvidia-smi** | GPU utilization, HBM 사용량, power, clock throttling 확인 |

성능 개선이 제대로 일어났다면 보통 다음과 같은 변화가 보여야 한다.

```text
Before:
  high HBM traffic
  low Tensor Core utilization
  memory stall dominant

After:
  lower bytes moved per operation
  higher Tensor Core utilization
  higher arithmetic intensity
  lower memory-related stall
```

즉 TE의 목적은 단순히 dtype을 낮추는 것이 아니라, Transformer의 핵심 GEMM/attention/MLP 경로를 더 높은 arithmetic intensity와 더 낮은 memory traffic으로 실행하는 것이다.











{% include comments.html %}