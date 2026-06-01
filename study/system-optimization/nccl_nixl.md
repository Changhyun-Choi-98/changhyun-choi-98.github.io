---
layout: post
title: "NCCL, NIXL, and Distributed GPU Communication"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-06-01
tags:
  - Korean
  - CUDA
language: ko
summary: "GPU 수가 늘어날수록 병목이 되는 inter-GPU data movement를 NCCL collective communication과 NIXL inference transfer 관점에서 정리한다"
math: true
comments: true
comment_id: "study-system-optimization-nccl-nixl"
permalink: /study/system-optimization/nccl-nixl/
---

대규모 AI 시스템에서 성능을 결정하는 것은 GPU 하나의 연산 속도만이 아니다. 모델이 하나의 GPU에 들어가지 않거나, training/inference workload가 여러 GPU와 여러 node(GPU 서버)로 분산되면, GPU들은 계속해서 서로 데이터를 주고받아야 한다. 이때 **communication overhead**가 커지면 GPU는 계산을 하지 못하고 기다리게 되고, 전체 시스템의 goodput(실제로 유용한 일을 하는 비율)이 떨어진다. 즉 예를 들어 연구 단계의 8-GPU workload를 production 단계의 80,000-GPU workload로 키우려면, 단순히 GPU 수만 늘리는 것이 아니라 GPU 사이의 communication을 최적화해야 한다. 이런 맥락에서 **NCCL**은 distributed collective communication용 라이브러리이고, **NIXL**은 distributed inference에서 GPU↔GPU 및 GPU/CPU/storage tier 사이의 point-to-point data movement용 라이브러리로 등장한다.

## **communication 구분**

대규모 GPU 시스템에서 communication은 크게 두 부류로 나눌 수 있다:

| 구분                               | 의미                                        | 대표 예시                                                      | 주로 쓰이는 라이브러리                    |
| -------------------------------- | ----------------------------------------- | ---------------------------------------------------------- | ------------------------------- |
| **Collective communication**     | 여러 GPU가 같은 communication operation에 함께 참여 | `all-reduce`, `all-gather`, `reduce-scatter`, `all-to-all` | **NCCL**                        |
| **Point-to-point communication** | 특정 GPU/worker에서 다른 GPU/worker 또는 storage tier로 데이터를 직접 이동 | prefill worker → decode worker로 KV cache 전송 | **NIXL**, UCX/libfabric, GPUDirect RDMA 기반 transport 등 |

Collective communication은 “cluster 전체의 모든 GPU”가 아니라, 해당 `communicator` 또는 `process group`에 속한 여러 `rank`가 같은 통신 연산에 함께 참여하는 패턴이다. 여기서 `rank`는 보통 distributed job 안의 한 process이며, 일반적인 GPU training에서는 rank 하나가 GPU 하나를 담당한다. 예를 들어 8-GPU DDP job에서 하나의 process group이 8개 rank로 구성되어 있다면, gradient 평균을 위해 8개 rank가 모두 같은 순서로 `all-reduce`를 호출해야 한다. 각 rank가 다른 count, datatype, operation 순서로 collective를 호출하면 hang, crash, data corruption 같은 문제가 생길 수 있다.

반면 point-to-point communication은 반드시 모든 GPU가 동시에 참여할 필요가 없다. 예를 들어 LLM inference에서 prefill GPU가 만든 KV cache를 특정 decode GPU로 넘기는 상황은 “모든 GPU가 함께 평균을 내는 작업”이 아니라 “한 worker가 다른 worker에게 큰 tensor/cache를 빠르게 넘기는 작업”에 가깝다. 이 영역에서 NIXL이 중요해진다.

## **NCCL**

NCCL(NVIDIA Collective Communications Library)은 NVIDIA GPU 사이의 high-performance communication을 위한 라이브러리이다. 즉 GPU들이 함께 수행하는 collective communication primitive를 빠르게 실행하도록 설계되어 있다. 이를 위해 topology-aware inter-GPU communication primitive를 제공하며, `AllReduce`, `AllGather`, `ReduceScatter`, `AlltoAll` 같은 collective를 지원한다. 다만 “NCCL = collective only”라고 이해하면 안 된다. 현재 NCCL은 `ncclSend`, `ncclRecv` 같은 point-to-point API도 제공한다. 여러 `send`/`recv`를 group call로 묶으면 one-to-all, all-to-one, all-to-all 같은 더 복잡한 communication pattern도 표현할 수 있다. 하지만 NCCL의 주된 강점은 여러 GPU rank가 함께 참여하는 synchronized collective를 topology-aware하게 빠르게 처리하는 데 있다. 반대로 NIXL은 inference pipeline 안에서 prefill worker, decode worker, cache/storage tier 같은 서로 다른 component 사이의 asynchronous large-object transfer를 더 직접적으로 겨냥한다.

여러 GPU가 하나의 모델을 함께 학습하려면, 각 GPU가 계산한 결과를 계속 맞춰줘야 한다. 예를 들어 DDP(Distributed Data Parallel) training을 생각해보자. 각 GPU는 서로 다른 mini-batch를 처리하고 자기 batch에 대한 gradient를 계산한다. model weight update는 모든 GPU가 같은 방향으로 해야 하기 때문에 모든 GPU의 gradient를 합치고 평균을 내야 한다. 이때 쓰이는 대표 collective가 `all-reduce`이다.

NCCL은 단순한 "데이터 복사 함수 모음집"이 아니다. NCCL은 GPU들이 어떤 방식으로 연결되어 있는지, 즉 topology를 보고 최적의 communication path를 선택한다. 이때 PCIe, NVLink, NVSwitch, InfiniBand, TCP sockets 같은 다양한 interconnect를 지원한다. 예를 들어 같은 node 안에서 GPU들이 NVLink/NVSwitch로 연결되어 있으면 그 경로를 우선 사용하고, node를 넘어가야 하면 InfiniBand, RoCE, TCP socket 같은 네트워크 경로를 사용한다.

## **Collective operation**

Collective operation은 여러 GPU, 더 정확하게는 여러 rank가 함께 참여하는 통신 연산이다. 대표적인 예시는 아래 4가지가 있다.

### **All-reduce**

All-reduce는 distributed training에서 가장 대표적인 collective로, 모든 GPU의 값을 합쳐서 모든 GPU가 같은 결과를 갖게 한다. 여러 device의 데이터를 sum/min/max 같은 연산으로 reduce한 뒤, 그 결과를 모든 rank의 receive buffer에 저장한다. 예를 들어서 4개의 GPU가 있고, 각 GPU가 gradient를 하나씩 가지고 있다고 하자:

```text
GPU 0: g0
GPU 1: g1
GPU 2: g2
GPU 3: g3
```

`all-reduce(sum)`을 하면 모든 GPU가 같은 결과를 받는다:

```text
GPU 0: g0 + g1 + g2 + g3
GPU 1: g0 + g1 + g2 + g3
GPU 2: g0 + g1 + g2 + g3
GPU 3: g0 + g1 + g2 + g3
```

DDP에서는 보통 이 값을 `world_size`로 나눠 평균 gradient를 만든다.

All-reduce는 “각 GPU가 따로 계산한 gradient를 하나의 공통 gradient로 맞추는 작업”이다. GPU 수가 늘어날수록 계산량은 분산되지만, gradient를 맞추는 communication 비용도 커진다. 그래서 대규모 training에서는 all-reduce가 병목이 되는 경우가 많다. 성능 관점에서 중요한 것은 all-reduce를 computation 뒤에 몰아서 실행하지 않는 것이다. 

PyTorch DDP는 보통 GPU당 하나의 process를 두기 때문에 `nn.DataParallel`처럼 single process 안에서 여러 GPU를 Python thread로 제어할 때 생기는 GIL(Global Interpreter Lock, 한 번에 하나의 스레드만 파이썬 코드를 실행할 수 있도록 하는 잠금장치) contention을 피한다. 또한 backward pass 중 gradient bucket이 준비되는 즉시 NCCL `all-reduce`를 별도 communication CUDA stream에 enqueue하여, 뒤쪽 layer의 gradient communication과 앞쪽 layer의 backward computation을 overlap한다. 즉 DDP의 성능 이점은 크게 두 가지다. 첫째, process-per-GPU 구조로 Python-side scheduling 병목을 줄인다. 둘째, NCCL all-reduce를 backward computation과 겹쳐 exposed communication time을 줄인다.

```text
나쁜 경우:
backward compute 전체 완료 → all-reduce 시작 → GPU 대기

좋은 경우:
layer별 gradient 계산 → 준비된 bucket부터 all-reduce 시작
                      → 다음 layer gradient 계산과 communication overlap
```

### **Reduce-scatter**

Reduce-scatter는 `all-reduce`를 두 단계로 쪼개서 볼 때 첫 번째에 해당하는 operation이다. 먼저 여러 rank의 값을 reduce한다. 하지만 reduce된 전체 결과를 모든 rank에게 복제하지 않고, rank별로 shard를 나눠서 받는다.

예를 들어 4개 GPU가 각각 같은 크기의 gradient vector를 가지고 있다고 하자. `reduce-scatter(sum)`을 하면 전체 gradient가 먼저 sum으로 reduce되고, 그 결과가 4개 shard로 나뉘어 각 GPU에 분배된다.

```text
full_result = g0 + g1 + g2 + g3

GPU 0: full_result의 shard 0
GPU 1: full_result의 shard 1
GPU 2: full_result의 shard 2
GPU 3: full_result의 shard 3
```

이 operation은 FSDP(PyTorch Fully Sharded Data Parallel)나 ZeRO(Zero Redundancy Optimizer)-style sharded training에서 중요하다. DDP에서는 모든 GPU가 full gradient를 갖는 방향으로 동작하지만, sharded training에서는 gradient, parameter, optimizer state를 여러 GPU에 나눠 저장한다. 따라서 모든 rank가 전체 결과를 복제하는 `all-reduce`보다, reduce된 결과의 shard만 각 rank가 받는 `reduce-scatter`가 메모리 효율적이다. 또한 all-reduce는 개념적으로 다음처럼 볼 수 있다:

```text
all-reduce = reduce-scatter + all-gather
```

즉 먼저 reduce된 결과를 rank별 shard로 나눠 받은 뒤, 필요하면 다시 `all-gather`로 모든 shard를 모아 모든 rank가 full result를 갖게 만들 수 있다.

### **All-gather**

All-gather는 각 rank가 가진 조각을 모아 모든 rank에게 동일하게 배포하는 operation이다. k개 rank에서 각각 N개 값을 모아 k×N 크기의 output buffer를 만들고, 그 결과를 모든 rank에 배포한다. 예를 들어 4개 GPU가 각각 parameter shard 또는 activation shard를 가지고 있다고 하자:

```text
GPU 0: x0
GPU 1: x1
GPU 2: x2
GPU 3: x3
```

`all-gather`를 하면 모든 GPU가 `[x0, x1, x2, x3]`를 갖게 된다:

```text
GPU 0: [x0, x1, x2, x3]
GPU 1: [x0, x1, x2, x3]
GPU 2: [x0, x1, x2, x3]
GPU 3: [x0, x1, x2, x3]
```

All-gather는 sharded training에서 자주 나온다. 예를 들어 FSDP에서는 parameter, gradient, optimizer state를 여러 GPU에 나눠 저장하지만 특정 transformer block을 계산하려면 해당 block의 weight 전체가 필요할 수 있다. 이때 각 GPU가 가진 weight shard를 모아 일시적으로 full weight를 만들기 위해 all-gather가 쓰인다.

all-gather는 엄밀히 말하면 arithmetic aggregation은 아니다. `all-reduce`는 sum/max/min 같은 reduction을 하지만, `all-gather`는 각 rank의 데이터를 “모아서 복제”한다.

### **All-to-all**

All-to-all은 모든 GPU가 모든 GPU에게 서로 다른 데이터를 보내는 방식이다. 각 GPU가 모든 GPU에게 보낼 데이터를 따로 가지고 있고, 서로 교환하는 방식이기 때문에 communication pattern이 더 복잡하다(각 rank가 k×N 크기의 input buffer를 제공하고, j번째 chunk를 destination rank j로 보내며, 각 rank는 모든 source rank에서 온 chunk를 받음). 예를 들어 GPU 0이 아래와 같이 데이터를 가지고 있다고 하자:

```text
GPU 0 input: [a00, a01, a02, a03]
```

여기서 `a00`은 GPU 0에게 남길 데이터, `a01`은 GPU 1에게 보낼 데이터, `a02`는 GPU 2에게 보낼 데이터, `a03`은 GPU 3에게 보낼 데이터다. 4개 GPU 전체를 표현하면 아래와 같다:

```text
GPU 0 input: [a00, a01, a02, a03]
GPU 1 input: [a10, a11, a12, a13]
GPU 2 input: [a20, a21, a22, a23]
GPU 3 input: [a30, a31, a32, a33]
```

`all-to-all` 후에 GPU 2는 각 GPU가 GPU 2에 보낸 조각을 받는다:

```text
GPU 2 output: [a02, a12, a22, a32]
```

All-to-all은 특히 MoE의 expert parallelism에서 중요하다. MoE에서는 token마다 선택된 expert가 다를 수 있다. 어떤 token은 GPU 0의 expert로, 어떤 token은 GPU 3의 expert로 가야 한다. 따라서 각 GPU가 가진 token activation을 expert가 있는 GPU로 보내야 하고, expert computation이 끝난 뒤 다시 원래 순서에 맞게 모아야 한다. 이때 all-to-all pattern이 핵심 병목이 될 수 있다. all-to-all exchange는 구현이나 runtime scheduling에 따라 pairwise exchange를 여러 round로 나누고, 통신 상대를 회전시키는 butterfly/shuffle-style pattern으로 스케줄링할 수 있다. 이렇게 하면 특정 NVSwitch port나 link에 traffic이 한꺼번에 몰리는 것을 줄일 수 있다.


### **어떤 workload에서 어떤 communication이 나오나?**

정리하면 collective operation은 추상적인 통신 연산이 아니라, 실제 distributed training/inference parallelism 전략과 직접 연결된다.

| Workload / parallelism | 자주 나오는 communication | 의미 |
| --------------------- | ------------------------- | ---- |
| **DDP** | `all-reduce` | 각 GPU가 계산한 gradient를 평균내서 모든 rank가 같은 model update를 수행 |
| **FSDP / ZeRO-style sharding** | `all-gather`, `reduce-scatter` | 필요한 parameter shard를 모으고, gradient/optimizer state는 다시 shard로 나눠 저장 |
| **Tensor Parallelism** | `all-reduce`, `all-gather`, `reduce-scatter` | 하나의 layer 계산을 여러 GPU에 나눠 수행한 뒤 partial result를 합치거나 모음 |
| **Pipeline Parallelism** | point-to-point send/recv | 한 stage의 activation을 다음 stage로 전달 |
| **MoE / Expert Parallelism** | `all-to-all` | token activation을 선택된 expert가 있는 GPU로 보내고, 계산 후 다시 원래 위치로 모음 |
| **Disaggregated inference** | NIXL 기반 point-to-point transfer | prefill worker가 만든 KV cache를 decode worker 또는 memory/storage tier로 이동 |

즉 NCCL은 DDP/FSDP/TP/MoE처럼 여러 GPU가 하나의 parallel computation에 함께 참여하는 상황에서 핵심이고, NIXL은 disaggregated inference처럼 stage/component 사이에서 큰 tensor나 KV cache를 빠르게 옮겨야 하는 상황에서 중요하다.


### **NCCL이 collective를 빠르게 만드는 3가지 방법**

#### **1. Topology awareness**
NCCL은 GPU들이 어떤 경로로 연결되어 있는지 확인하고 빠른 경로를 우선 사용한다. 예를 들어 GPU 0–1은 NVLink로 빠르게 연결되어 있고 GPU 2–3도 NVLink로 연결되어 있지만, 두 그룹 사이가 PCIe로만 연결되어 있다면, 모든 데이터를 무작정 PCIe로 흘리지 않고 먼저 NVLink-connected pair 안에서 reduce한 뒤 cross-group exchange를 최소화하는 식의 hierarchical pattern을 선택할 수 있다. 

#### **2. Ring, tree, PAT 같은 algorithm**

NCCL은 message size와 interconnect topology, GPU generation을 보고 collective별로 ring, tree, PAT 같은 algorithm을 선택한다.

#### **3. SHARP/NVLS 같은 in-network aggregation 활용**

일부 고성능 fabric에서는 reduction 일부를 GPU가 아니라 network switch가 처리할 수 있다. all-reduce 같은 collective의 일부 aggregation을 network fabric에 offload하면 GPU가 처리해야 하는 데이터 이동량과 latency를 줄일 수 있다. 이것은 대규모 cluster에서 매우 중요하다. GPU 수가 커질수록 “계산하는 GPU”만 중요한 것이 아니라, “데이터를 합치는 network fabric” 자체가 일종의 accelerator가 된다.


## **NIXL**

NIXL(NVIDIA Inference Xfer Library)은 distributed inference에서 large tensor, KV cache, model shard 같은 데이터를 빠르게 옮기기 위한 open-source data movement library다. NVIDIA가 만들었지만 특정 inference engine 하나에만 묶인 라이브러리는 아니며, GPU memory, CPU memory, file/block/object storage를 backend plugin 구조로 추상화한다. 따라서 NIXL의 핵심은 “KV cache만 전송하는 라이브러리”라기보다, disaggregated inference에서 component 사이의 large-object transfer를 non-blocking 방식으로 수행하는 data movement layer라고 보는 것이 정확하다.

LLM inference는 크게 두 단계로 나눌 수 있다:

| 단계          | 역할                            | 병목 성격                     |
| ----------- | ----------------------------- | ------------------------- |
| **Prefill** | prompt 전체를 처리해서 KV cache 생성   | compute-bound 경향          |
| **Decode**  | KV cache를 참조하면서 token을 하나씩 생성 | memory-bandwidth-bound 경향 |

대규모 serving에서는 prefill과 decode의 병목 특성이 다르기 때문에, 두 단계를 서로 다른 GPU pool로 분리하는 disaggregated prefill/decode architecture가 점점 중요해지고 있다. 이 구조에서는 prefill worker가 만든 KV cache를 decode worker에게 매우 빠르게(low latency/high throughput) 넘겨야 한다. 

하지만 KV cache는 작지 않다. 긴 prompt, long-context, multi-turn conversation에서는 KV cache가 수백 MB에서 수 GB 이상까지 커질 수 있다. 이것을 CPU memory로 복사한 뒤 TCP로 보내고 다시 GPU memory로 복사하면 latency가 너무 커지게 된다. NIXL은 가능한 경우 GPUDirect RDMA, NVLink/NVSwitch, GPUDirect Storage(GDS), object-store backend 같은 빠른 data path/backend를 사용해 GPU memory, CPU memory, SSD, distributed filesystem, object storage 사이의 transfer를 수행한다. 중요한 점은 “항상 RDMA만 쓴다”가 아니라, source와 destination의 위치에 따라 가능한 backend 중 적절한 path를 선택하도록 설계되어 있다는 것이다. GPU↔GPU transfer에서는 host memory staging을 피하는 것이 핵심이고, GPU↔file/block storage transfer에서는 GDS 같은 path를 통해 CPU bounce buffer를 줄이는 것이 핵심이다. object storage는 별도의 object-store backend/plugin을 통해 다뤄질 수 있다.

```text
느린 GPU↔GPU path:
GPU memory → CPU memory → network stack → CPU memory → GPU memory

빠른 GPU↔GPU path:
GPU memory → NVLink / NVSwitch / RDMA → GPU memory

빠른 GPU↔file/block storage path:
GPU memory ↔ GDS / NVMe / NVMe-oF / distributed filesystem ↔ storage tier

object storage path:
GPU/CPU memory ↔ object-store backend ↔ object storage
```

### **NCCL과 NIXL**

NCCL과 NIXL은 대체 관계가 아니라 보완 관계다. NCCL은 여러 GPU rank가 같은 computation에 함께 참여할 때 필요한 synchronized collective communication에 강하다. 예를 들어 DDP의 gradient `all-reduce`, FSDP의 `all-gather`/`reduce-scatter`, MoE의 `all-to-all` 같은 패턴이 여기에 속한다.

반면 NIXL은 disaggregated inference처럼 서로 다른 stage나 component 사이에서 large tensor/cache를 비동기적으로 이동해야 하는 상황에 초점을 둔다. 대표적인 예시는 prefill worker가 만든 KV cache를 decode worker로 보내거나, GPU HBM에 다 담기 어려운 KV cache를 CPU memory, SSD, shared storage tier로 offload/load하는 경우다.

주의할 점은 NCCL도 `send`/`recv` 같은 point-to-point API를 지원한다는 것이다. 하지만 large-scale inference에서는 KV cache 이동, storage tiering, non-blocking metadata exchange, heterogeneous backend 선택 같은 요구가 중요해지므로, 이런 경우에는 NIXL이 더 직접적인 abstraction을 제공한다.

| 항목    | NCCL                                                        | NIXL                                                                  |
| ----- | ----------------------------------------------------------- | --------------------------------------------------------------------- |
| 핵심 목적 | GPU group의 collective communication                         | inference pipeline의 point-to-point data movement                      |
| 대표 패턴 | all-reduce, all-gather, reduce-scatter, all-to-all          | prefill → decode KV cache transfer, cache offload/load                |
| 참여 방식 | 같은 communicator/process group에 속한 rank들이 같은 collective에 참여 | source/destination 중심의 비동기 transfer |
| 주 사용처 | distributed training, tensor parallel, FSDP, MoE collective | disaggregated inference, KV cache tiering, model shard/cache movement |
| 성능 핵심 | topology-aware collective algorithm                         | nonblocking transfer, memory/storage hierarchy abstraction            |
| 예시    | 8 GPU의 gradient all-reduce                                  | GPU 0의 1 GB KV cache를 GPU 1 또는 SSD tier로 이동                           |

최신 NVIDIA 자료 기준으로도 이 방향성은 유지된다. 다만 NIXL은 단순히 “KV cache 전송 라이브러리”에만 머물지 않고, disaggregated KV cache movement, long-context storage, model weight transfer, elastic expert parallelism 같은 inference framework 내부 data movement 전반을 위한 abstraction으로 확장되고 있다. [NVIDIA 2026년 글](https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/){:target="_blank" rel="noopener noreferrer"}도 NIXL이 non-blocking API, dynamic metadata exchange, pluggable backend plugin을 제공하며 [Dynamo](https://www.nvidia.com/en-us/ai/dynamo/){:target="_blank" rel="noopener noreferrer"}, TensorRT-LLM, vLLM, LMCache 등 주요 inference framework와 통합되고 있다고 설명한다.



## **Performance engineer checklist**

### **Communication을 computation과 overlap해야 한다**

분산 training/inference에서 이상적인 성능은 communication 시간이 완전히 사라지는 것이 아니라, 계산 뒤에 드러나는 exposed communication time을 줄이는 것이다:

```text
No overlap:
total time = compute time + communication time

With overlap:
total time ≈ max(compute time, communication time) + small tail
```

### **빠른 interconnect 안에 traffic을 최대한 가둬야 한다**

가능하면 데이터 이동을 아래처럼 더 가까운 memory/fabric 안에서 끝내는 것이 좋다. 정확한 우선순위는 hardware generation, link count, topology, message size에 따라 달라질 수 있지만, performance engineering 관점의 기본 원칙은 “가까운 곳에서 끝내고, 느린 fabric으로 나가는 traffic을 줄인다”이다.

```text
가장 가까움:
GPU local HBM

같은 node/rack의 빠른 GPU fabric:
NVLink / NVSwitch

host 또는 off-node 경로:
PCIe → InfiniBand / RoCE with GPUDirect RDMA

fallback 또는 피하고 싶은 경로:
TCP/IP → CPU memory staging → storage staging
```

NVL72 같은 rack-scale system에서는 가능한 한 NVLink/NVSwitch domain 안에서 collective를 끝내고, rack 밖으로 나가는 communication을 최소화하는 것이 중요하다.

### **PyTorch에서는 `DataParallel`보다 `DistributedDataParallel`을 써야 한다**

성능을 생각한다면 PyTorch `nn.DataParallel`은 피하는 것이 좋다. `DataParallel`은 single process가 여러 GPU를 제어하기 때문에 Python GIL, GPU 0 중심 gather/scatter, communication overlap 부족 문제가 생긴다. 반면 `DistributedDataParallel`은 GPU당 process를 두고 NCCL all-reduce를 사용해 gradient communication을 overlap한다.

### **NCCL communicator는 반복문 안에서 만들면 안 된다**

NCCL communicator 생성은 비싸다. communicator는 rank들이 서로 통신할 수 있도록 group, route, buffer 등을 설정하는 객체다. 이것을 매 iteration마다 만들고 지우면 collective 자체보다 초기화 비용이 더 커질 수 있다. 따라서 `init_process_group` 같은 communicator setup은 한 번만 하고 반복해서 재사용해야 한다.

### **Inference KV cache 이동은 CPU staging을 피해야 한다**

Disaggregated inference에서 KV cache를 CPU memory로 불필요하게 꺼냈다가 다시 GPU로 넣는 구조는 latency와 bandwidth 양쪽에서 손해가 크다. GPU↔GPU 또는 node↔node transfer에서는 NVLink/NVSwitch, RDMA, GPUDirect RDMA 기반 path를 우선 활용하고, GPU↔storage transfer에서는 GDS나 NIXL의 storage backend를 활용해 CPU bounce buffer와 host staging을 줄여야 한다.


{% include comments.html %}
