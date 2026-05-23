---
layout: default
title: "Overview of NVIDIA GPU Models"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-05-23
tags:
  - Korean
language: ko
summary: "NVIDIA GPU의 세대와 모델들의 차이와 용도를 정리"
math: true
comments: true
comment_id: "study-system-optimization-nvidia-gpu-overview"
permalink: /study/system-optimization/nvidia-gpu-overview/
---

# **Overview of NVIDIA GPU Models**

## **Generation Overview**

NVIDIA GPU 세대는 대략 아래와 같이 분류된다. 세대별로 아키텍처가 변화했다.

|    시대 | 아키텍처            | 대표 제품                          |
| ----: | --------------- | ------------------------------ |
| 2006~ | Tesla           | GeForce 8800, 초기 CUDA GPU      |
| 2010~ | Fermi           | GTX 400/500, Tesla M 계열        |
| 2012~ | Kepler          | GTX 600/700, Tesla K80         |
| 2014~ | Maxwell         | GTX 900, Jetson Nano 계열        |
| 2016~ | Pascal          | GTX 10, Tesla P100             |
| 2017~ | Volta           | Tesla V100                     |
| 2018~ | Turing          | RTX 20, GTX 16, T4             |
| 2020~ | Ampere          | RTX 30, A100, A10              |
| 2022~ | Ada Lovelace    | RTX 40, L4, L40S, RTX 6000 Ada |
| 2022~ | Hopper          | H100, H200, H800               |
| 2024~ | Blackwell Data Center | B200, GB200 |
| 2025~ | Blackwell Consumer/Pro | GeForce RTX 50, RTX PRO Blackwell |
| 2025~ | Blackwell Ultra | B300, GB300 |

CUDA 관점에서는 GPU마다 compute capability가 정해지며, 이는 해당 GPU가 지원하는 CUDA hardware feature와 instruction capability를 나타낸다. 예를 들어 Turing 계열 T4/RTX 20은 7.5, A100은 8.0, A10/RTX 30은 8.6, Ada 계열 L4/L40S/RTX 40/RTX 6000 Ada는 8.9, Hopper 계열 H100/H200/GH200은 9.0, Blackwell 계열 B200/GB200은 10.0, Blackwell Ultra 계열 B300/GB300은 10.3, GeForce RTX 50 및 RTX PRO Blackwell 계열은 12.0으로 분류된다.

같은 아키텍처 안에서도 모델이 많이 나뉘는 이유는 GPU가 들어가는 시장이 다르기 때문이다. 크게 다음과 같이 정리할 수 있다:

| 구분                         | 대표 제품군                               | 핵심 차이                                |
| -------------------------- | ------------------------------------ | ------------------------------------ |
| **Data Center / AI / HPC** | A100, H100, H200, B200, B300         | HBM, NVLink, MIG, FP8/FP4, 대규모 학습/추론 |
| **Inference Server**       | T4, L4, A10, L40S                    | 전력 효율, 비디오 엔진, PCIe 서버 장착성           |
| **Workstation / Pro**      | RTX 6000 Ada, RTX PRO 6000 Blackwell | ECC 메모리, 드라이버 안정성, 전문 SW 인증          |
| **Consumer GeForce**       | RTX 3090, 4090, 5090                 | 게임/개인 연구/렌더링, 가격 대비 성능               |
| **Embedded / Edge**        | Jetson Orin, Thor                    | 로봇/엣지 AI, 전력 제한, SoC 통합              |

다만 이 분류는 서로 완전히 배타적인 제품군 구분이라기보다는 주 용도 관점의 분류이다. 예를 들어 L40S는 Data Center GPU이면서 inference server, rendering server, video workload, virtual workstation 용도로도 사용될 수 있다.

## **Data Center / AI GPU 계열**

### **Ampere 세대: A100, A800, A10**

[A100](https://www.nvidia.com/en-us/data-center/a100/){:target="_blank" rel="noopener noreferrer"}은 Ampere 세대의 대표 AI/HPC(High-Performance Computing, 대규모 과학·공학·AI 계산을 고성능 병렬 컴퓨팅 자원으로 수행하는 분야) GPU이다.

| 항목     | A100                             |
| ------ | -------------------------------- |
| 아키텍처   | Ampere                           |
| 대표 메모리 | 40GB / 80GB HBM2e                |
| 메모리 대역폭 | 약 2TB/s                 |
| NVLink  | 600GB/s                    |
| 주요 특징  | Tensor Core, MIG, NVLink, 대규모 학습 |
| 포지션    | H100 이전 세대의 데이터센터 표준 GPU         |

A800은 A100의 중국 수출규제 대응 variant이다. 핵심 연산 아키텍처는 Ampere 계열이지만, interconnect나 특정 성능 지표가 제한되었다. A10은 Ampere 기반의 PCIe 데이터센터 GPU로, A100처럼 HBM/NVLink 중심의 대규모 학습용 GPU라기보다는 cloud graphics, VDI, inference workload를 겨냥한 24GB GDDR6 기반 서버 GPU이다.

### **Hopper 세대: H100, H200, H800**

[H100](https://www.nvidia.com/en-us/data-center/h100/){:target="_blank" rel="noopener noreferrer"}은 Hopper 세대의 대표 AI GPU이다.

| 항목      | H100                                   |
| ------- | -------------------------------------- |
| 아키텍처    | Hopper                                 |
| 대표 메모리  | 80GB HBM3                              |
| 메모리 대역폭 | SXM 기준 약 3.35TB/s                      |
| NVLink  | SXM 기준 GPU당 900GB/s                    |
| 주요 특징   | FP8, Transformer Engine, 대규모 LLM 학습/추론 |
| 포지션     | A100 후속 AI 표준 GPU                      |


[H200](https://www.nvidia.com/en-us/data-center/h200/){:target="_blank" rel="noopener noreferrer"}은 완전히 새로운 아키텍처라기보다는 Hopper 기반에 HBM3e를 붙여 메모리 용량과 대역폭을 크게 늘린 모델이다. H200이 중요한 이유는 단순 FLOPS가 아니라 memory capacity + bandwidth이다. LLM inference에서는 KV cache가 커지고, batch/context가 길어질수록 HBM 용량과 대역폭이 latency/throughput을 지배한다.

| 항목      |  H100 |  H200 |
| ------- | --------- | -------------------------------------------------------- |
| 아키텍처    |     Hopper |                                                    Hopper |
| 대표 메모리  |  80GB HBM3 |                                               141GB HBM3e |
| 메모리 대역폭 | 약 3.35TB/s |                                                 약 4.8TB/s |
| 강점      |   학습/추론 범용 | LLM inference, 긴 context, KV cache, memory-bound workload |

A800처럼 H800은 중국 수출규제 환경에 맞춰 출시된 Hopper 기반 variant로 이해할 수 있다. 공개 보도 기준으로 H800은 H100 대비 chip-to-chip data transfer rate가 낮아진 제품으로 설명되며, 따라서 단일 GPU 연산보다 multi-GPU scaling에서 통신 병목이 더 중요해질 수 있다.

| 항목    | H100                     | H800                    |
| ----- | ------------------------ | ----------------------- |
| 아키텍처  | Hopper                   | Hopper                  |
| 목적    | 글로벌 데이터센터                | 중국 수출규제 대응              |
| 핵심 차이 | full interconnect        | interconnect/성능 일부 제한   |
| 실무 영향 | 대규모 multi-GPU scaling 유리 | scale-out 통신에서 불리할 수 있음 |

### **Blackwell 세대: B200, B300, GB200, GB300**

[Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/){:target="_blank" rel="noopener noreferrer"}의 핵심 키워드는 다음과 같다.

| 키워드                       | 의미                                            |
| ------------------------- | --------------------------------------------- |
| **FP4 / NVFP4**           | 초저정밀 inference/training 가속                    |
| **2-die GPU**             | 두 개의 대형 die를 고속 chip-to-chip interconnect로 연결 |
| **HBM3e 확장**              | 더 큰 모델과 KV cache 처리                           |
| **NVLink 5 / rack-scale** | GPU 하나가 아니라 랙 전체를 하나의 거대한 AI machine처럼 사용     |
| **LLM inference 최적화**     | attention, MoE, long-context workload 최적화     |

B200은 Blackwell 세대의 대표 데이터센터 GPU이다. 실제 제품명에서는 단일 GPU 카드만 말하는 것이 아니라 DGX B200, GB200, NVL72 같은 시스템 단위 이름으로 자주 등장한다. H100/H200은 FP8 시대의 대표 GPU라면, B200은 FP4/NVFP4 시대를 연 GPU이다. 특히 대규모 inference에서는 더 낮은 precision으로 더 많은 token/s를 뽑는 것이 중요해졌기 때문에 이와 같은 변화가 이루어졌다.

| 항목     | B200                                                  |
| ------ | ----------------------------------------------------- |
| 아키텍처   | Blackwell                                             |
| 주요 정밀도 | FP8, FP6, FP4/NVFP4                                   |
| 주요 강점  | LLM training/inference, MoE, long-context, rack-scale |
| 대표 시스템 | DGX B200, GB200 NVL72                                 |

[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/){:target="_blank" rel="noopener noreferrer"}은 8개의 Blackwell GPU, 총 1,440GB GPU memory, 64TB/s HBM3e bandwidth, FP4 sparse 기준 144PFLOPS, aggregate NVLink bandwidth 14.4TB/s를 제공한다.

GB200은 Grace CPU + Blackwell GPU를 묶은 superchip/system 계열이다.

| 항목     | GB200                                                 |
| ------ | ----------------------------------------------------- |
| 구성     | Grace CPU + Blackwell GPU                             |
| 대표 시스템 | GB200 NVL72                                           |
| 목적     | 랙스케일 AI supercomputer                                 |
| 핵심     | GPU만 빠른 것이 아니라 CPU, GPU, NVLink, NVSwitch를 시스템 단위로 설계 |

[NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/){:target="_blank" rel="noopener noreferrer"}는 36개의 Grace CPU와 72개의 Blackwell GPU를 하나의 rack-scale system으로 구성하며, 13.4TB HBM3e GPU memory, 576TB/s memory bandwidth, 130TB/s NVLink bandwidth를 제공한다.

B300은 [Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/){:target="_blank" rel="noopener noreferrer"} 계열로, B200의 단순 클럭 상승판이라기보다는 대규모 reasoning/inference workload를 위해 memory capacity, memory bandwidth, NVFP4 throughput, attention-layer execution을 강화한 Blackwell 계열 확장판으로 보는 것이 좋다.

| 항목    |            B200 |              B300 / Blackwell Ultra |
| ----- | -------------- | ---------------------------------- |
| 세대    |       Blackwell |                     Blackwell Ultra |
| 핵심 목적 |   FP4 기반 대규모 AI | 더 큰 모델, 더 긴 context, 더 강한 inference |
| 메모리 | HBM3e, B200 기준 약 180GB급 | 최대 288GB급 HBM3e per GPU |
| 대역폭   |           매우 높음 |                   GPU당 최대 8TB/s급 |
| 주 사용처 | DGX B200, GB200 |                     DGX B300, GB300 |

또한 NVLink 5는 GPU당 1.8TB/s 연결을 제공하고, NVL72 랙에서는 72 GPU를 130TB/s NVLink domain으로 묶는다. [DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/){:target="_blank" rel="noopener noreferrer"}은 8개의 Blackwell Ultra SXM GPU, 총 2.1TB GPU memory, FP4 Tensor Core dense 기준 108PFLOPS, sparse 기준 144PFLOPS, aggregate NVLink bandwidth 14.4TB/s를 제공한다.


[GB300](https://www.nvidia.com/en-us/data-center/dgx-gb300/){:target="_blank" rel="noopener noreferrer"}은 Grace Blackwell Ultra 계열의 rack-scale 시스템이다.

| 항목    |             GB200 |                          GB300 |
| ----- | ---------------- | ----------------------------- |
| GPU   |         Blackwell |                Blackwell Ultra |
| 구성    | Grace + Blackwell |        Grace + Blackwell Ultra |
| 주요 차이 |   Blackwell NVL72 | 더 큰 memory, 더 강한 FP4/attention |
| 목적    |     rack-scale AI |  더 큰 LLM/MoE/agentic inference |

[GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/){:target="_blank" rel="noopener noreferrer"}는 72개의 Blackwell Ultra GPU와 36개의 Grace CPU를 사용하며, Blackwell 대비 dense FP4는 1.5배, attention 성능은 2배 높다. 총 20TB GPU memory, 최대 576TB/s memory bandwidth, FP4 dense 1080PFLOPS, sparse 1440PFLOPS, NVLink 130TB/s를 제공한다.

### **빠른 비교**

지금까지의 모델들을 정리하면 아래와 같다:

| 모델        | 세대              | 정체                             | 핵심 차이                         |
| --------- | --------------- | ------------------------------ | ----------------------------- |
| **A100**  | Ampere          | 이전 세대 AI 표준 GPU                | HBM2e, FP16/BF16/TF32, MIG    |
| **A800**  | Ampere          | A100 규제 대응 variant             | interconnect/성능 일부 제한         |
| **H100**  | Hopper          | LLM 시대 대표 GPU                  | FP8, Transformer Engine, HBM3 |
| **H200**  | Hopper          | H100 memory 강화판                | 141GB HBM3e, 4.8TB/s          |
| **H800**  | Hopper          | H100 규제 대응 variant             | Hopper 기반, interconnect 제한 가능 |
| **B200**  | Blackwell       | Blackwell 대표 AI GPU            | FP4/NVFP4, HBM3e, rack-scale  |
| **GB200** | Blackwell       | Grace + Blackwell system       | 36 CPU + 72 GPU NVL72 구성 가능   |
| **B300**  | Blackwell Ultra | B200 강화판                       | 더 큰 HBM, 더 강한 attention/FP4   |
| **GB300** | Blackwell Ultra | Grace + Blackwell Ultra system | GB200의 Ultra 강화판              |


## **Inference Server GPU: T4, L4, A10, L40S**

데이터센터라고 해서 항상 H100/B200만 쓰는 것은 아니다. inference, video, cloud graphics, light AI serving에는 더 작은 GPU가 자주 쓰인다.

| 항목       | [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/){:target="_blank" rel="noopener noreferrer"}| [L4](https://www.nvidia.com/en-us/data-center/l4/){:target="_blank" rel="noopener noreferrer"}| [A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/){:target="_blank" rel="noopener noreferrer"}| [L40S](https://www.nvidia.com/en-us/data-center/l40s/){:target="_blank" rel="noopener noreferrer"}|
| -------- | -------------------------- | ------------------------- | ---------------------------- | --------------------------- |
| 세대       | Turing | Ada Lovelace              | Ampere| Ada Lovelace|
| 메모리      | 16GB GDDR6            | 24GB GDDR6 (300GB/s bandwidth) | 24GB GDDR6 (600GB/s bandwidth)| 48GB GDDR6 ECC (864GB/s bandwidth)|
| 전력       | 약 70W| 72W| 150W| 350W|
| 특징 및 용도 | 강점: 저전력 inference, video transcoding<br>약점: 대규모 LLM에는 메모리/대역폭 부족 | 강점: inference per watt, video, edge/cloud serving | 용도: graphics, VDI, inference, cloud GPU | 메모리 대역폭: 864GB/s<br>특징: PCIe, RT core, Tensor core, no MIG/NVLink |

L40S는 48GB GDDR6 ECC, 864GB/s memory bandwidth, 18,176 CUDA cores, 568 Tensor Cores, 350W, PCIe Gen4 x16 사양을 가져서 robotics policy inference, simulation rendering, VLA prototype profiling에 유리한 GPU이다.

## **Workstation / Professional RTX 계열**

Inference Server GPU는 서비스를 안정적으로 많이 처리하는 GPU이기 때문에, 정해진 모델을 가능한 낮은 비용과 전력으로 많이, 안정적으로, 오래 serving하는 것이 목표이다. 반면 Workstation / Professional RTX는 사람이 직접 붙어서 개발·렌더링·시뮬레이션·디자인·AI 연구를 하는 GPU이기 때문에 한 명 또는 소수의 엔지니어가 고성능 GPU를 직접 사용하면서 개발, 실험, 렌더링, 시뮬레이션을 빠르게 반복하는 것이 목표라는 점에서 다르다.

| 항목          | RTX 6000 Ada                               |
| ----------- | ------------------------------------------ |
| 세대          | Ada Lovelace                               |
| 메모리         | 48GB GDDR6 ECC                             |
| 용도          | CAD, rendering, simulation, AI workstation |
| GeForce와 차이 | ECC, pro driver, 안정성, 인증 SW                |

| 항목         | [RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf){:target="_blank" rel="noopener noreferrer"}                              |
| ---------- | --------------------------------------------------- |
| 세대         | Blackwell                                           |
| 메모리        | 96GB GDDR7 ECC                                      |
| 메모리 대역폭    | 1,792GB/s                                           |
| CUDA cores | 24,064                                              |
| 전력         | 600W                                                |
| 용도         | AI workstation, rendering, simulation, digital twin |

## **일반 사용자용 GeForce GPU**

게임용이지만, 개인 AI 연구/LoRA/소형 LLM inference/robotics prototype에도 많이 쓴다.

### **GeForce 이름 읽는 법**

예: RTX 4090

| 부분    | 의미                       |
| ----- | ------------------------ |
| RTX   | RT Core + Tensor Core 포함 |
| 40    | 세대: Ada Lovelace         |
| 90    | 등급: 최상위                  |
| Ti    | 강화형                      |
| SUPER | 중간 refresh 강화형           |

일반적으로 세대와 등급은 아래와 같다:

| 세대 number | 세대명 |
| ------ | --------------- |
| **50** | Blackwell |
| **40** | Ada Lovelace |
| **30** | Ampere |
| **20** | Turing |
| **10** | Pascal |



| 등급     | 의미              |
| ------ | --------------- |
| **90** | 최상위 enthusiast  |
| **80** | high-end        |
| **70** | upper mid-range |
| **60** | mainstream      |
| **50** | entry-level     |

일반적인 스펙으로 보면 RTX 4090/5090도 매우 빠른데, H100/B200등과 같은 Data Center GPU와 비교하면 아래와 같다:

| 항목                | GeForce RTX          | Data Center GPU    |
| ----------------- | -------------------- | ------------------ |
| 메모리               | GDDR6X/GDDR7         | HBM2e/HBM3/HBM3e   |
| 메모리 대역폭           | 높지만 HBM보다 낮음         | 매우 높음              |
| VRAM 용량           | 보통 8~32GB            | 80GB~288GB급        |
| NVLink            | 최근 consumer에서는 거의 없음 | 핵심 기능              |
| Multi-GPU scaling | 제한적                  | NVLink/NVSwitch 중심 |
| MIG               | 없음                   | A100/H100 등 지원     |
| ECC/안정성           | 제한적                  | 서버 안정성 중심          |
| 드라이버              | GeForce driver       | data center driver |
| 목적                | 게임/creator/개인 연구     | 대규모 학습/추론/HPC      |

## **최종 정리**

### **데이터센터 AI 계열**

| 세대              | 대표         | 한 줄 요약                               |
| --------------- | ---------- | ------------------------------------ |
| Pascal          | P100       | pre-Tensor Core HPC/AI 시대            |
| Volta           | V100       | Tensor Core 본격 도입                    |
| Turing          | T4         | 저전력 inference                        |
| Ampere          | A100       | TF32/BF16/MIG, 대규모 AI 표준             |
| Ada             | L4/L40S    | inference/graphics/server efficiency |
| Hopper          | H100       | FP8 + Transformer Engine             |
| Hopper refresh  | H200       | H100보다 훨씬 큰 HBM3e                    |
| Hopper export   | H800       | H100 기반 규제 대응 variant                |
| Blackwell       | B200/GB200 | FP4 + rack-scale AI                  |
| Blackwell Ultra | B300/GB300 | 더 큰 memory, 더 강한 attention/FP4       |


### **소비자 GPU 계열**

| 세대              | 대표     | 한 줄 요약                                  |
| --------------- | ------ | --------------------------------------- |
| Pascal          | GTX 10 | CUDA/gaming 명기, RTX 없음                  |
| Turing          | RTX 20 | RT/Tensor 소비자 최초                        |
| Turing cut-down | GTX 16 | RT/Tensor 없는 Turing                     |
| Ampere          | RTX 30 | RTX 3090 24GB가 개인 AI 연구 인기              |
| Ada             | RTX 40 | RTX 4090 24GB, 개인 연구 최상급                |
| Blackwell       | RTX 50 | RTX 5090 32GB, GDDR7, DLSS 4/5세대 Tensor |






{% include comments.html %}
