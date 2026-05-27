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
  - Writing
language: ko
summary: "NVIDIA Transformer Engine 하드웨어에 대한 정리"
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

즉 TE hardware는 TE software가 target으로 삼는 GPU 하드웨어 기능들이다.

## **Transformer/LLM**

Transformer 기반 LLM의 대부분의 연산은 결국 $Y = XW$와 같은 큰 행렬곱이다. 예를 들어 LLM 하나의 layer 안에는 다음과 같은 matrix-heavy 연산이 반복된다:

| Transformer 구성 요소 | 실제 연산                          |
| ----------------- | ------------------------------ |
| Q/K/V projection  | `X @ Wq`, `X @ Wk`, `X @ Wv`   |
| attention score   | `Q @ K^T`                      |
| attention output  | `softmax(QK^T) @ V`            |
| MLP / FFN         | `X @ W1`, activation, `H @ W2` |
| MoE expert        | 선택된 expert별 대형 GEMM            |

즉 LLM의 성능은 상당 부분 GEMM throughput, memory bandwidth, activation/weight movement, precision format에 의해 결정된다. Tensor Core는 이런 MMA(matrix multiply-accumulate) 연산을 일반 CUDA core보다 훨씬 빠르게 처리하는 전용 유닛이다. 따라서 TE hardware를 활용한다는 말은 "`nn.Linear`, attention GEMM, GGN GEMM과 같은 Transformer 핵심 행렬 연산을 일반 FP32 CUDA core 경로가 아니라 **Tensor Core + FP8/FP4/NVFP4 + scaling/casting-aware fused kernel** 경로로 실행하겠다"라는 의미이다.

## **Key Idea**

핵심은 **낮은 precision으로 더 많은 연산을 더 적은 memory traffic으로 처리하겠다**는 것이다. 각 data format별로 값 하나가 차지하는 크기는 다음과 같다:

| Format      | 값 하나의 크기 | FP32 대비 weight/activation traffic |
| ----------- | ------- | -------------------------------- |
| FP32        |  4 bytes |                                1× |
| BF16/FP16   |  2 bytes |                               1/2 |
| FP8         |   1 byte |                               1/4 |
| FP4 / NVFP4 | 0.5 byte |                               1/8 |


LLM inference/training에서는 GPU가 계산을 못 해서 느린 경우도 있지만, 실제로는 HBM에서 weight/activation을 가져오느라 느린 경우가 많다. 값 하나의 bit-width를 줄이면 같은 HBM transaction으로 더 많은 값을 가져올 수 있고, Tensor Core도 더 많은 MMA를 한 cycle에 처리할 수 있다.

하지만 전부 FP8/FP4로 바꾸는 것이 아니다. 낮은 precision을 쓰되, 정확도가 중요한 부분은 더 높은 precision을 유지해야 한다. 예를 들면 다음과 같다:

| 부분                            | 흔한 선택                              |
| ----------------------------- | ---------------------------------- |
| 대형 GEMM input                 | FP8, NVFP4, BF16                   |
| accumulation                  | FP16/BF16/FP32                     |
| LayerNorm, softmax, reduction | 보통 FP16/BF16/FP32 유지               |
| master weight                 | training에서는 보통 higher precision 유지 |
| inference weight              | FP8/FP4 quantized weight 가능        |














{% include comments.html %}