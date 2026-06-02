---
layout: post
title: "Data, Tensor, Pipeline, and Expert Parallelism"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-06-02
tags:
  - Korean
  - PyTorch
language: ko
summary: "multi-GPU AI system에서 모델이 커지거나 요청량이 늘어날 때, 데이터를 복제할지, layer 안의 tensor 연산을 쪼갤지, layer stack을 나눌지, MoE expert를 분산할지를 결정하는 병렬화 전략"
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
| **Data Parallelism**     | 데이터 또는 request                     | 전체 throughput 증가             | training: gradient [all-reduce](/study/system-optimization/nccl-nixl/) / inference: 거의 없음 | 모델 replica마다 GPU memory 필요       |
| **Tensor Parallelism**   | layer 내부 weight matrix, activation | 큰 layer를 여러 GPU로 계산   | [all-reduce](/study/system-optimization/nccl-nixl/), [all-gather](/study/system-optimization/nccl-nixl/), [reduce-scatter](/study/system-optimization/nccl-nixl/)           | 매 layer 통신 비용                    |
| **Pipeline Parallelism** | layer sequence                     | 모델을 깊이 방향으로 분할        | stage 간 activation send/recv                     | pipeline bubble, stage imbalance |
| **Expert Parallelism**   | MoE expert                         | 많은 expert를 여러 GPU에 분산 | [all-to-all](/study/system-optimization/nccl-nixl/) token dispatch/combine                | routing overhead, load imbalance |

위 parallelism들은 서로 조합이 가능하다. 실제로 대규모 LLM/MoE serving에서는 TP로 layer의 "폭"을 나누고, PP로 model의 "깊이"를 나누고, EP로 expert를 나누고, 마지막으로 DP replica를 늘려 전체 user traffic을 처리한다. 이를 hybrid parallelism이라고 한다.

## **Data Parallelism**

가장 직관적인 분산 방식이다. 모델 전체를 각 GPU에 복제하고, 각 GPU가 서로 다른 데이터 또는 서로 다른 inference request를 처리한다. Training에서는 각 GPU가 같은 모델 replica를 가지고 서로 다른 mini-batch shard를 forward/backward한다. 이후 각 GPU에서 계산한 gradient를 평균내기 위해 [all-reduce](/study/system-optimization/nccl-nixl/)를 수행한다.

```text
GPU 0: model replica + batch shard 0 → gradient g0
GPU 1: model replica + batch shard 1 → gradient g1
GPU 2: model replica + batch shard 2 → gradient g2
GPU 3: model replica + batch shard 3 → gradient g3

all-reduce:
g = average(g0, g1, g2, g3)

모든 GPU가 같은 update를 적용
```

즉, data parallel training에서는 GPU마다 다른 데이터를 보지만, gradient synchronization 이후에는 동일한 모델 parameter update를 수행한다. 그래서 GPU별로 따로 학습하는 것이 아니라, 하나의 global batch를 여러 GPU가 나눠서 처리하는 것에 가깝다.

Inference에서는 더 단순하다. 모델이 GPU 하나에 들어간다면 각 GPU에 모델 replica를 올리고, incoming request를 여러 GPU에 load balancing하면 된다. 이 경우 개별 request는 GPU 하나에서 끝나므로 inter-GPU synchronization이 거의 필요 없다. 그래서 data parallel inference는 매우 단순하고, request 수가 많을 때 throughput을 거의 linear하게 scaling할 수 있다. 단점은 individual query latency를 줄이지 못하며 replica마다 full model memory가 필요하다는 점이다.

### **Data Parallelism이 좋은 경우**

Data Parallelism은 모델이 GPU 하나에 들어가고, request 또는 training batch가 충분히 많을 때 가장 좋다.

예를 들어 quantization 또는 충분한 GPU memory 덕분에 13B급 모델이 한 GPU의 memory budget 안에 들어가고, 동시에 수백 개 request가 들어오는 serving 시스템이라면 tensor/pipeline parallelism으로 한 request를 여러 GPU에 찢기보다, 모델 replica를 여러 개 띄워 request를 분산하는 편이 단순하고 빠를 수 있다.

### **Data Parallelism의 한계**

가장 큰 한계는 memory 중복이다. GPU 8장에 모델을 각각 복제하면 모델 weight가 8번 올라간다는 뜻이다. 모델이 GPU 하나에 안 들어가는 순간 순수 data parallelism은 불가능해진다.

또한 training에서는 gradient [all-reduce](/study/system-optimization/nccl-nixl/)가 필요하다. GPU가 늘수록 communication cost가 커지고, 이 communication이 backward computation과 잘 overlap되지 않으면 GPU가 통신을 기다리며 idle 상태가 된다.

PyTorch 관점에서, PyTorch의 `DataParallel`은 input을 batch dimension으로 쪼개고 forward에서 module을 각 device에 replicate하고 backward에서 gradient를 original module로 합친다. 하지만 multi-GPU training에서는 `DistributedDataParallel` 사용을 권장한다. 각 model replica의 gradient를 동기화하는 방식이고, single-node multi-GPU에서도 이것이 더 유의미하게 빠르다고 [PyTorch(v2.12) 공식 문서](https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html){:target="_blank" rel="noopener noreferrer"}에 명시되어 있다. [지난 게시물](/study/system-optimization/nccl-nixl/)에서 아래와 같이 설명했다:

> 성능을 생각한다면 PyTorch `nn.DataParallel`은 피하는 것이 좋다. `DataParallel`은 single process가 여러 GPU를 제어하기 때문에 Python GIL, GPU 0 중심 gather/scatter, communication overlap 부족 문제가 생긴다. 반면 `DistributedDataParallel`은 GPU당 process를 두고 NCCL all-reduce를 사용해 gradient communication을 overlap한다.

## **Tensor Parallelism**

Tensor parallelism은 모델의 layer 자체, 특히 transformer의 큰 matrix multiplication을 여러 GPU에 나누는 방식이다. 즉, tensor parallelism은 layer 안의 폭(width)을 나누는 방식이다.

Transformer layer의 핵심 연산은 대부분 $Y = XW$와 같이 큰 matrix multiplication이다. 여기서 $W$가 너무 크면 GPU 하나에 올리기 어렵거나, GPU 하나가 계산하기에 시간이 너무 오래 걸릴 수 있다. Tensor parallelism은 이 $W$를 여러 GPU에 나눈다. 예를 들어 column-wise로 나눈다면 아래와 같다:

```text
W = [W0, W1]

GPU 0: Y0 = XW0
GPU 1: Y1 = XW1

Y = concat(Y0, Y1)
```

row-wise로 나누는 경우에는 아래와 같이 각 GPU가 partial output을 만든 뒤 [all-reduce](/study/system-optimization/nccl-nixl/)로 결과를 합쳐야 한다:

```text
GPU 0: partial Y0
GPU 1: partial Y1

Y = all_reduce_sum(Y0, Y1)
```

Tensor parallelism은 큰 layer를 GPU 여러 장에 나눠 compute-bound layer에서 near-linear speedup을 얻을 수 있지만 매 layer마다 all-reduce 같은 frequent communication이 발생한다.

### **Tensor Parallelism이 좋은 경우**

Tensor parallelism은 아래와 같은 상황들에서 유용하다:

1. 하나의 layer가 너무 커서 GPU 하나에 안 들어갈 때
2. single request latency를 줄이고 싶을 때
    * 단, layer compute 감소분이 [all-reduce](/study/system-optimization/nccl-nixl/)/[all-gather](/study/system-optimization/nccl-nixl/) 같은 collective communication 증가분보다 커야 한다.
3. GEMM이 compute-bound라 여러 GPU에 나누면 계산 시간이 줄어들 때
4. NVLink/NVSwitch처럼 GPU 간 bandwidth가 매우 높은 환경일 때

예를 들어 transformer의 MLP projection이 너무 커서 한 GPU의 memory나 compute capacity를 초과한다면, 그 matrix를 여러 GPU에 나누는 것이 자연스럽다.

### **Tensor Parallelism의 한계**

Tensor parallelism은 GPU를 많이 쓸수록 무조건 좋아지는 방식이 아니다. TP의 가장 큰 문제는 매 layer마다 통신이 생긴다는 점이다. Transformer는 layer가 수십 개에서 수백 개까지 이어지기 때문에, layer마다 [all-reduce](/study/system-optimization/nccl-nixl/)/[all-gather](/study/system-optimization/nccl-nixl/)가 발생하면 interconnect가 병목이 된다. 따라서 TP는 “GPU 수를 늘리는 문제”라기보다 “compute 감소분이 communication 증가분보다 큰 지점까지 TP degree를 선택하는 문제”다.

TP는 가능하면 같은 node 내부, 특히 NVLink/NVSwitch로 묶인 GPU들 안에서 쓰는 것이 좋다. [AI Systems Performance Engineering](https://www.oreilly.com/library/view/ai-systems-performance/9798341627772/){:target="_blank" rel="noopener noreferrer"}에서는 intranode TP는 Blackwell NVL72 같은 NVSwitch domain 안에 유지하고, inter-rack으로 확장하는 것은 topology가 허용할 때만 하라고 설명한다. 따라서 TP degree는 “크면 클수록 좋은 값”이 아니라, profiling을 통해 compute 감소와 communication 증가가 균형을 이루는 지점에서 정해야 한다.

PyTorch 관점에서, `torch.distributed.tensor.parallel`이 DTensor 위에서 Colwise, Rowwise, Sequence Parallelism 같은 style을 제공하지만, [현재(v2.12) 문서](https://docs.pytorch.org/docs/2.12/distributed.tensor.parallel.html){:target="_blank" rel="noopener noreferrer"} 기준 API가 experimental이며 변경 가능성이 있다고 명시되어 있다.

## **Pipeline Parallelism**

Pipeline parallelism은 모델을 layer sequence 방향으로 나눈다. layer 1부터 layer 60까지 있는 transformer가 있고 이것을 3개 GPU에 나눈다면 아래와 같다:

```text
input
  → GPU 0: layers 1–20
  → GPU 1: layers 21–40
  → GPU 2: layers 41–60
  → output
```

즉 pipeline parallelism은 모델의 깊이(depth)를 나누는 방식이다. 매우 깊은 모델이 GPU 하나에 들어가지 않을 때 model state를 layer 단위로 분산할 수 있다.

### **Pipeline parallelism과 microbatching**

Pipeline parallelism을 단순히 하나의 batch에 대해 순차 실행하면 GPU utilization이 낮다. GPU 0이 일하는 동안 GPU 1, GPU 2는 기다리고, GPU 2가 일할 때 GPU 0은 다시 기다린다. 그래서 microbatching을 사용한다:

```text
Microbatch 0: GPU 0 → GPU 1 → GPU 2
Microbatch 1:         GPU 0 → GPU 1 → GPU 2
Microbatch 2:                 GPU 0 → GPU 1 → GPU 2
```
이렇게 하면 assembly line처럼 서로 다른 microbatch가 서로 다른 pipeline stage에서 동시에 처리된다.

### **Pipeline bubble**

PP는 prefill처럼 긴 sequence를 처리하거나 large batch를 처리할 때 throughput을 높일 수 있지만, decode phase처럼 token을 하나씩 생성하는 경우에는 pipeline bubble 때문에 pure PP의 이점이 줄어든다. Pipeline bubble은 pipeline이 처음 채워지는 구간과 마지막에 비워지는 구간에서는 일부 GPU가 idle 상태가 되는 상황을 의미하며 이것이 PP의 핵심 단점이다. 예를 들어 4-stage pipeline에서 microbatch 수가 적으면 다음과 같은 현상이 발생한다:

```text
초기 구간: GPU 0만 일하고 GPU 1–3은 대기
중간 구간: GPU 0–3이 모두 일함
마지막 구간: GPU 3만 일하고 GPU 0–2는 대기
```

이 idle 구간이 pipeline bubble이다. 그래서 PP는 microbatch 수가 충분하거나, long sequence/prefill처럼 pipeline을 길게 유지할 수 있는 workload에서 더 유리하다.

### **Pipeline Parallelism이 좋은 경우**

Pipeline parallelism은 주로 모델이 너무 깊거나 전체 layer stack이 GPU 하나에 들어가지 않을 때 사용한다. 또한 TP만으로 model memory 문제가 해결되지 않을 때 PP를 추가한다. Tensor parallelism이 layer 안을 쪼개어 모델의 폭을 나눈다면 pipeline parallelism은 layer들을 stage로 나눠서 모델의 깊이를 나눈다. TP는 매 layer의 matrix 연산을 여러 GPU가 같이 계산하게 만들고, PP는 layer stack을 여러 GPU에 순서대로 배치한다.

다만 PP는 single request latency를 줄이기 위한 만능 전략은 아니다. Stage 간 activation transfer와 pipeline fill/flush overhead가 있기 때문에, 보통은 “한 GPU에 안 들어가는 모델을 나눠 담기 위한 capacity scaling 전략”으로 먼저 이해하는 것이 좋다.

PyTorch 관점에서, `torch.distributed.pipelining`이 model execution을 partition해서 multiple micro-batches가 서로 다른 model part를 동시에 실행할 수 있게 해주지만, [현재(v2.12) 문서](https://docs.pytorch.org/docs/2.12/distributed.pipelining.html){:target="_blank" rel="noopener noreferrer"} 기준 alpha 상태이며 API 변경 가능성이 있다고 명시되어 있다.

## **Expert Parallelism**

Expert parallelism은 MoE (mixture-of-experts) 모델에서 사용되는 병렬화 방식이다. MoE layer에는 여러 개의 expert network가 있다. 일반적인 dense model은 모든 token이 같은 MLP block을 통과하지만, MoE에서는 router 또는 gating network가 token마다 사용할 expert를 선택한다. 즉 각 GPU마다 담당하는 expert를 따로 배치할 수 있다. 각 token은 모든 expert를 거치지 않고 router가 top-1 또는 top-2 expert를 선택한다. 이때 token activation은 해당 expert가 있는 GPU로 이동해야 한다. 그래서 expert parallelism은 보통 [all-to-all](/study/system-optimization/nccl-nixl/) communication을 필요로 한다. MoE layer에서는 token activation vector를 expert가 있는 GPU로 보내기 위해 [all-to-all](/study/system-optimization/nccl-nixl/) communication을 수행하고, expert computation 이후 다시 원래 순서로 되돌리기 위해 또 다른 [all-to-all](/study/system-optimization/nccl-nixl/)이 필요하다.

다만 MoE 모델이라고 해서 모든 연산이 expert로만 구성되는 것은 아니다. Attention, embedding, router, shared dense layer, output head 같은 dense path는 여전히 남아 있을 수 있다. 따라서 큰 MoE 모델에서는 expert는 EP로 나누고, dense layer나 매우 큰 expert 내부 연산은 TP로 나누고, 전체 layer stack은 PP로 나누는 식의 hybrid parallelism이 자주 필요하다.


### **Expert Parallelism의 장점**

Expert parallelism의 가장 큰 장점은 전체 parameter 수는 크게 늘리면서도 token당 compute는 제한할 수 있다는 점이다. Dense model에서는 parameter가 10배 늘면 보통 token당 compute도 크게 늘어난다. 반면 MoE에서는 expert가 100개 있어도 token 하나가 top-2 expert만 사용한다면, token당 active parameter는 일부에 불과하다. 이 구조 덕분에 massive MoE model은 dense model보다 훨씬 큰 parameter capacity를 가질 수 있다.


### **Expert Parallelism의 bottleneck**

1. all-to-all communication overhead가 크다.
    * Token activation을 expert 위치로 보내고 다시 모아야 한다.
2. load imbalance가 생길 수 있다
    * Router가 특정 expert에 token을 많이 보내면 해당 expert가 있는 GPU만 바빠지고, 다른 GPU는 idle 상태가 된다.
    * 이때 가장 늦게 끝나는 expert가 전체 MoE layer의 latency를 결정하는 straggler effect가 생긴다.
3. expert별 token 수가 충분해야 communication overhead가 amortize된다.
    * Expert 하나당 처리할 token 수가 너무 작으면 계산보다 통신 비용이 더 커진다.

MoE serving에서는 expert placement와 routing이 성능을 크게 좌우한다. 즉 “expert를 몇 개 만들었는가”보다 “token이 expert 사이에 얼마나 균형 있게 분산되는가”가 더 중요하다. MoE에서는 router가 곧 scheduler이고, expert placement가 곧 communication optimization이다. 자주 같이 선택되는 expert를 같은 GPU 또는 같은 node에 배치하면 communication을 줄일 수 있다. 특정 expert가 과부하되면 expert replica를 만들거나, capacity factor(expert별로 받을 수 있는 token 수의 상한을 두어 특정 expert가 과도하게 붐비는 것을 막는 장치)를 조정하거나, routing bias를 조정할 수 있다.



## **Parallelism 전략 조합 방법**

### **1단계: 모델이 GPU 하나에 들어가고 동시 request가 많다**
이 경우에는 **data parallelism**이 가장 단순하다.

```text
GPU마다 full model replica
각 GPU가 독립 request 처리
→ throughput 증가
```
개별 request latency는 줄지 않지만, 동시에 처리할 수 있는 request 수가 늘어난다.

### **2단계: 모델이 GPU 하나에 안 들어가는데 layer가 너무 크다**
이 경우에는 **tensor parallelism**을 고려한다.

```text
하나의 transformer layer 안의 matrix를 여러 GPU에 분할
→ layer compute와 memory를 여러 GPU에 분산
```
단, TP는 매 layer마다 통신이 생기므로 NVLink/NVSwitch 같은 빠른 interconnect 안에서 구성하는 것이 좋다.

### **3단계: 모델이 너무 깊어서 전체 layer stack이 안 들어간다**
이 경우에는 **pipeline parallelism**을 고려한다.

```text
GPU 0: 앞쪽 layers
GPU 1: 중간 layers
GPU 2: 뒤쪽 layers
→ model depth를 여러 GPU에 분산
```
단, PP는 pipeline bubble이 생기므로 microbatching과 stage balancing이 중요하다.

### **4단계: 모델이 MoE다**
이 경우에는 **expert parallelism**을 고려한다.

```text
expert들을 여러 GPU에 분산
router가 token별 expert 선택
token activation을 expert GPU로 all-to-all dispatch
```
단, EP는 routing overhead와 load imbalance가 핵심 병목이다.

### **5단계: 더 큰 시스템에서는 hybrid로 간다**
예를 들어 64 GPU cluster에서 massive MoE model을 serving한다고 하자.

```text
Pipeline Parallelism:
  60 layers를 4개 stage로 나눔
  → stage당 15 layers

Tensor Parallelism:
  각 stage 내부의 큰 matrix 연산을 2-way TP로 분할

Expert Parallelism:
  MoE expert들을 GPU group에 분산
  top-2 gating으로 token을 expert로 routing

Data Parallelism:
  이런 64-GPU replica를 여러 개 만들어 request throughput 증가
```

단, 이런 조합은 정답이 고정되어 있는 것이 아니라 model size, layer shape, sequence length, batch size, GPU topology, interconnect bandwidth에 따라 달라진다. 따라서 실제 시스템에서는 Nsight Systems로 end-to-end timeline과 communication overlap을 보고, Nsight Compute로 kernel-level Tensor Core utilization과 memory bottleneck을 확인하면서 TP/PP/EP/DP degree를 조정해야 한다.

## **요약**

<aside class="content-summary" markdown="1">

* Data parallelism은 **모델을 복제하고 데이터를 나누는 방식**이다. 모델이 GPU 하나에 들어갈 때 throughput을 늘리는 가장 단순한 전략이다.

* Tensor parallelism은 **layer 내부의 tensor 연산을 나누는 방식**이다. Transformer의 큰 matrix multiplication을 여러 GPU가 함께 계산하지만, 매 layer마다 communication이 필요하다.

* Pipeline parallelism은 **layer들을 stage로 나누는 방식**이다. 모델이 너무 깊어서 GPU 하나에 들어가지 않을 때 유용하지만, pipeline bubble과 stage imbalance를 관리해야 한다.

* Expert parallelism은 **MoE expert를 여러 GPU에 분산하는 방식**이다. Token마다 일부 expert만 활성화하므로 전체 parameter capacity를 크게 늘릴 수 있지만, all-to-all routing과 expert load balancing이 병목이 된다.

* 실전 대규모 AI system에서는 하나의 parallelism만 고르는 것이 아니라, TP로 layer의 폭을 나누고, PP로 layer의 깊이를 나누고, EP로 expert를 나누고, DP로 request throughput을 늘리는 식으로 조합한다.

</aside>


{% include comments.html %}
