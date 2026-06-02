---
layout: post
title: "Data, Tensor, Pipeline, and Expert Parallelism"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-06-02
tags:
  - Korean
  - Writing
language: ko
summary: "multi-GPU AI system에서 모델이 커지거나 요청량이 늘어날 때, 데이터를 복제할지, layer 안의 tensor 연산을 쪼갤지, layer stack을 나눌지, MoE expert를 분산할지를 결정"
math: true
comments: true
comment_id: "study-system-optimization-parallelism"
permalink: /study/system-optimization/parallelism/
---

## **Introduction**

Data / Tensor / Pipeline / Expert Parallelism의 핵심은 **model weights나 data를 GPU들에 어떻게 split할 것인가**이다. 데이터 batch/request를 나눌 수도 있고, transformer layer 안의 matrix multiplication을 나눌 수도 있고, 전체 layer sequence를 여러 GPU에 배치할 수도 있고, MoE 모델이라면 expert 자체를 여러 GPU에 분산할 수도 있다. 가장 먼저 확인해야 하는 것은 아래와 같다:

1. **모델이 GPU 하나에 들어가는가?**
    * 들어간다면 data parallelism으로 throughput을 늘리는 것이 가장 단순하다.
2. **모델이 GPU 하나에 들어가지 않는가?**
    * 그렇다면 모델 자체를 쪼개야 한다. layer 안의 큰 matrix를 쪼개면 tensor parallelism이고, layer들을 순서대로 나누면 pipeline parallelism이다. 모델이 MoE라면 expert를 나누는 expert parallelism이 추가된다.

이 네 가지 parallelism을 하나로 요약하면 아래와 같다:

| 전략                       | 무엇을 나누나?                           | 주 목적                  | 대표 통신 패턴                                         | 핵심 병목                            |
| ------------------------ | ---------------------------------- | --------------------- | ------------------------------------------------ | -------------------------------- |
| **Data Parallelism**     | 데이터 또는 request                     | 전체 처리량 증가             | training: gradient [all-reduce](/study/system-optimization/nccl-nixl/) / inference: 거의 없음 | 모델 replica마다 GPU memory 필요       |
| **Tensor Parallelism**   | layer 내부 weight matrix, activation | 큰 layer를 여러 GPU로 계산   | [all-reduce](/study/system-optimization/nccl-nixl/), [all-gather](/study/system-optimization/nccl-nixl/), [reduce-scatter](/study/system-optimization/nccl-nixl/)           | 매 layer 통신 비용                    |
| **Pipeline Parallelism** | layer sequence                     | 모델을 깊이 방향으로 분할        | stage 간 activation send/recv                     | pipeline bubble, stage imbalance |
| **Expert Parallelism**   | MoE expert                         | 많은 expert를 여러 GPU에 분산 | [all-to-all](/study/system-optimization/nccl-nixl/) token dispatch/combine                | routing overhead, load imbalance |

위 parallelism들은 서로 조합이 가능하다. 실제로 대규모 LLM/MoE serving에서는 TP로 layer의 "폭"을 나누고, PP로 model의 "깊이"를 나누고, EP로 expert를 나누고, 마지막으로 DP replica를 늘려 전체 user traffic을 처리한다. 이를 hybrid parallelism이라고 한다.
























{% include comments.html %}