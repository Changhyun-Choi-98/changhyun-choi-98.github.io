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

> **Scope Note:** 이 글은 NVIDIA의 모든 GPU를 빠짐없이 cataloging하는 글이 아니라, AI training/inference, robotics/Physical AI, visualization, workstation 관점에서 자주 마주치는 대표 GPU 계열을 중심으로 정리한 글이다. 따라서 A2, A16, A30, A40, L40 같은 일부 데이터센터 GPU는 필요할 때 별도 글에서 다룬다.

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

> 참고로 여기서 2006년의 **Tesla**는 초기 CUDA GPU microarchitecture를 의미하고, `Tesla K80`, `Tesla P100`에서의 **Tesla**는 과거 NVIDIA data center accelerator brand를 의미한다. 같은 단어지만 문맥이 다르다.

CUDA 관점에서는 GPU마다 compute capability가 정해진다. Compute capability는 해당 GPU가 지원하는 CUDA hardware feature와 instruction capability를 나타낸다. 예를 들어 T4는 7.5, A100은 8.0, A10/RTX 30은 8.6, Jetson Orin 계열은 8.7, L4/L40S/RTX 40/RTX 6000 Ada는 8.9, H100/H200/GH200은 9.0, B200/GB200은 10.0, B300/GB300은 10.3, Jetson Thor 계열은 11.0, GeForce RTX 50 및 RTX PRO Blackwell 계열은 12.0으로 분류된다. Blackwell 이후에는 단순히 compute capability 숫자만 보는 것에 더해 CUDA compilation target도 중요해졌다. 예를 들어 `sm_100`은 Blackwell B200/GB200 계열, `sm_103`은 B300/GB300 계열을 가리킬 수 있다. CUDA 12.9 이후에는 `compute_100f`처럼 family-specific target도 제공되며, 이는 같은 Blackwell 10.x family 안에서 공통으로 지원되는 family-specific feature를 사용할 수 있게 해준다. 반면 `compute_100a` 같은 architecture-specific target은 특정 compute capability에 더 강하게 묶이므로 forward compatibility가 제한된다.

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

아래 수치는 대표적으로 SXM form factor 기준이며, PCIe/NVL variant에서는 메모리 용량, 대역폭, NVLink 구성, TDP가 달라질 수 있다.

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

| 항목      |  H100 SXM |  H200 SXM |
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

> **Terminology Note:** Blackwell 계열에서는 제품명이 GPU chip, superchip, server, rack-scale system을 모두 가리킬 수 있으므로 구분이 중요하다.
>
> - **B200 / B300:** Blackwell 또는 Blackwell Ultra GPU 자체
> - **GB200 / GB300 Superchip:** 1 Grace CPU + 2개의 Blackwell (Ultra) GPU를 묶은 CPU-GPU module
> - **DGX B200 / DGX B300:** 여러 개의 Blackwell GPU를 탑재한 NVIDIA의 server system
> - **GB200 NVL72 / GB300 NVL72:** 36개의 Grace CPU와 72개의 Blackwell (Ultra) GPU를 NVLink/NVSwitch로 묶은 rack-scale AI system
>
> 즉, B200은 GPU이고, GB200은 Grace CPU와 Blackwell GPU가 결합된 superchip/system 계열이며, NVL72는 rack-scale deployment 단위이다.

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

[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/){:target="_blank" rel="noopener noreferrer"}은 8개의 Blackwell GPU, 총 1,440GB GPU memory, 64TB/s HBM3e bandwidth, FP4 Tensor Core 기준 144PFLOPS sparse / 72PFLOPS dense, aggregate NVLink bandwidth 14.4TB/s를 제공한다.

GB200은 Grace CPU + Blackwell GPU를 묶은 superchip/system 계열이다.

| 항목     | GB200                                                 |
| ------ | ----------------------------------------------------- |
| 구성     | Grace CPU + Blackwell GPU                             |
| 대표 시스템 | GB200 NVL72                                           |
| 목적     | rack-scale AI supercomputer                                 |
| 핵심     | GPU만 빠른 것이 아니라 CPU, GPU, NVLink, NVSwitch를 시스템 단위로 설계 |

[NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/){:target="_blank" rel="noopener noreferrer"}는 36개의 Grace CPU와 72개의 Blackwell GPU를 하나의 rack-scale system으로 구성하며, 13.4TB HBM3e GPU memory, 576TB/s memory bandwidth, 130TB/s NVLink bandwidth를 제공한다.

B300은 [Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/){:target="_blank" rel="noopener noreferrer"} 계열로, B200의 단순 클럭 상승판이라기보다는 reasoning/inference workload를 위해 memory capacity, memory bandwidth, NVFP4 throughput, attention-layer execution을 강화한 확장판으로 보는 것이 좋다. Blackwell Ultra는 최대 288GB HBM3E, GPU당 최대 8TB/s HBM bandwidth, dense NVFP4 15PFLOPS, NVLink 5 기준 GPU당 1.8TB/s interconnect를 제공하며, attention layer의 softmax 관련 연산 병목을 줄이기 위한 SFU throughput 강화도 포함한다.

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

| 모델 | 세대 | 정체 | 핵심 차이 |
| --- | --- | --- | --- |
| **A100** | Ampere | 이전 세대 AI 표준 GPU | HBM2e, FP16/BF16/TF32, MIG |
| **A800** | Ampere | A100 규제 대응 variant | interconnect/성능 일부 제한 |
| **H100** | Hopper | LLM 시대 대표 GPU | FP8, Transformer Engine, HBM3 |
| **H200** | Hopper | H100 memory 강화판 | 141GB HBM3e, 4.8TB/s |
| **H800** | Hopper | H100 규제 대응 variant | Hopper 기반, interconnect 제한 가능 |
| **B200** | Blackwell | Blackwell GPU | FP4/NVFP4, HBM3e, NVLink 5 |
| **GB200 Superchip** | Blackwell | 1 Grace CPU + 2 Blackwell GPU | CPU-GPU tightly coupled module |
| **GB200 NVL72** | Blackwell | 36 Grace CPU + 72 Blackwell GPU rack | 13.4TB HBM3E, 130TB/s NVLink |
| **B300** | Blackwell Ultra | Blackwell Ultra GPU | 더 큰 HBM, 더 강한 attention/FP4 |
| **GB300 Superchip** | Blackwell | 1 Grace CPU + 2 Blackwell Ultra GPU | CPU-GPU tightly coupled module |
| **GB300 NVL72** | Blackwell Ultra | 36 Grace CPU + 72 Blackwell Ultra GPU rack | 20TB GPU memory, reasoning/test-time scaling 지향 |


## **Mainstream(PCIe) Data Center / Inference / Visualization GPU: T4, L4, A10, L40S**

데이터센터라고 해서 항상 H100/B200만 쓰는 것은 아니다. inference, video, cloud graphics, light AI serving에는 더 작은 GPU가 자주 쓰인다.

| 항목       | [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/){:target="_blank" rel="noopener noreferrer"}| [L4](https://www.nvidia.com/en-us/data-center/l4/){:target="_blank" rel="noopener noreferrer"}| [A10](https://www.nvidia.com/en-us/data-center/products/a10-gpu/){:target="_blank" rel="noopener noreferrer"}| [L40S](https://www.nvidia.com/en-us/data-center/l40s/){:target="_blank" rel="noopener noreferrer"}|
| -------- | -------------------------- | ------------------------- | ---------------------------- | --------------------------- |
| 세대       | Turing | Ada Lovelace              | Ampere| Ada Lovelace|
| 메모리      | 16GB GDDR6            | 24GB GDDR6 (300GB/s bandwidth) | 24GB GDDR6 (600GB/s bandwidth)| 48GB GDDR6 ECC (864GB/s bandwidth)|
| 전력       | 약 70W| 72W| 150W| 350W|
| 특징 및 용도 | 강점: 저전력 inference, video transcoding<br>약점: 대규모 LLM에는 메모리/대역폭 부족 | 강점: inference per watt, video, edge/cloud serving | 용도: graphics, VDI, inference, cloud GPU | 메모리 대역폭: 864GB/s<br>특징: PCIe, RT core, Tensor core, no MIG/NVLink |

L40S는 48GB GDDR6 ECC, 864GB/s memory bandwidth, 18,176 CUDA cores, 568 Tensor Cores, 350W, PCIe Gen4 x16 사양을 가져서 robotics policy inference, simulation rendering, VLA prototype profiling에 유리한 GPU이다.

## **Workstation / Professional RTX 계열**

Inference Server GPU는 서비스를 안정적으로 많이 처리하는 GPU이기 때문에, 정해진 모델을 가능한 낮은 비용과 전력으로 많이, 안정적으로, 오래 serving하는 것이 목표이다. 반면 Workstation / Professional RTX는 사람이 직접 붙어서 개발·렌더링·시뮬레이션·디자인·AI 연구를 하는 GPU이기 때문에 한 명 또는 소수의 엔지니어가 고성능 GPU를 직접 사용하면서 개발, 실험, 렌더링, 시뮬레이션을 빠르게 반복하는 것이 목표라는 점에서 다르다. 아래는 해당 시리즈 중 예시로 두 모델을 가져온 것이고 전체 목록은 [여기](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/){:target="_blank" rel="noopener noreferrer"}를 보면 된다.

| 항목          | [RTX 6000 Ada](https://www.nvidia.com/en-us/products/workstations/rtx-6000/){:target="_blank" rel="noopener noreferrer"}                                   |
| ----------- | ------------------------------------------ |
| 세대          | Ada Lovelace                               |
| 메모리         | 48GB GDDR6 ECC                             |
| 용도          | CAD, rendering, simulation, AI workstation |
| GeForce와 차이 | ECC, pro driver, 안정성, 인증 SW                |

| 항목         | [RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/){:target="_blank" rel="noopener noreferrer"}                              |
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

## **Embedded / Edge GPU: Jetson Orin, Jetson Thor**

Jetson 계열은 discrete GPU가 아니라 CPU, GPU, memory, multimedia engine, I/O가 하나의 SoC 안에 통합된 edge AI platform이다. 데이터센터 GPU처럼 최대 FLOPS만 보는 것이 아니라 전력 제한, thermal budget, camera/encoder/decoder pipeline, real-time control loop와의 통합이 중요하다.

| 계열 | 대표 제품 | 핵심 특징 |
| --- | --- | --- |
| [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/){:target="_blank" rel="noopener noreferrer"} | Jetson AGX Orin, Orin NX, Orin Nano | Ampere 기반 edge AI SoC, robotics/vision inference |
| [Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/){:target="_blank" rel="noopener noreferrer"} | Jetson AGX Thor Developer Kit, Jetson T5000, Jetson T4000 | Blackwell 기반 edge/robotics AI platform, Physical AI와 humanoid robot workload 지향 |

Jetson Thor는 Blackwell 기반 GPU와 fifth-generation Tensor Cores를 사용하며, Jetson T5000 기준 최대 2070 FP4 sparse TFLOPS, 128GB LPDDR5X, 273GB/s memory bandwidth, 40W~130W power range를 제공한다.

로봇에서는 GPU inference latency만 중요한 것이 아니라 sensor input, preprocessing, policy inference, control output까지의 end-to-end latency와 jitter가 중요하다. 따라서 Jetson 계열은 CUDA 성능뿐 아니라 I/O, memory sharing, thermal throttling, power mode, real-time scheduling까지 함께 봐야 한다.


## **Next-generation Platform: Vera Rubin**

2026년 기준으로 NVIDIA는 Blackwell 이후 세대로 [Vera Rubin platform](https://www.nvidia.com/en-us/data-center/technologies/rubin/){:target="_blank" rel="noopener noreferrer"}도 공개했다. Vera Rubin NVL72는 72개의 Rubin GPU와 36개의 Vera CPU를 NVLink 6 기반 rack-scale system으로 묶는 차세대 AI factory platform이다. 다만 이 글에서는 현재 널리 비교·사용되는 Blackwell Ultra까지를 중심으로 정리하고, Rubin 계열은 별도 글에서 다룬다.

## **Performance Engineering 관점에서 GPU를 볼 때 중요한 축**

GPU를 비교할 때 단순히 FLOPS만 보면 안 된다. 실제 AI workload의 성능 병목은 workload type에 따라 달라진다.

| 병목 유형 | 중요한 스펙 | 예시 workload | 유리한 GPU |
| --- | --- | --- | --- |
| **Compute-bound** | Tensor Core throughput, supported precision | 대형 GEMM, training, batched inference | H100, H200, B200, B300 |
| **Memory-bandwidth-bound** | HBM/GDDR bandwidth | LLM decode, KV cache access, embedding lookup | H200, B200, B300 |
| **Memory-capacity-bound** | VRAM/HBM capacity | long-context inference, large batch serving, large model residence | H200, B300, GB300 |
| **Communication-bound** | NVLink, NVSwitch, InfiniBand | tensor parallelism, expert parallelism, large-scale training | H100/H200 SXM, GB200/GB300 NVL72 |
| **Video/rendering-bound** | NVENC/NVDEC, RT Core, graphics driver support | video analytics, simulation rendering, digital twin | L4, L40S, RTX PRO |
| **Power-bound / edge-bound** | TDP, performance per watt, SoC integration | robot onboard inference, edge AI | Jetson Orin, Jetson Thor |
| **Latency / jitter-bound** | p50/p95/p99 latency, kernel launch overhead, CPU-GPU synchronization, scheduling jitter | real-time robot policy inference, VLA control loop, online perception-control pipeline | Jetson Orin/Thor, L4/L40S, RTX PRO, 단 실제 유리함은 software stack과 power/thermal setting에 의존 |

따라서 “어떤 GPU가 더 좋은가?”라는 질문은 항상 “어떤 workload에서, 어떤 batch/context/model size로, 단일 GPU인지 multi-GPU인지, latency가 중요한지 throughput이 중요한지”와 함께 판단해야 한다. 특히 robotics workload에서는 평균 latency보다 p95/p99 latency와 jitter가 더 중요할 수 있다. 제어 주기가 20ms 또는 50ms로 정해져 있다면, 평균이 빠른 GPU보다 tail latency가 안정적인 시스템 구성이 더 중요할 수 있다.

## **최종 정리**

### **데이터센터 AI 계열**

| 세대 | 대표 | 핵심 정밀도/기능 | 병목 관점의 강점 | 한 줄 요약 |
| --- | --- | --- | --- | --- |
| Volta | V100 | FP16 Tensor Core | training acceleration | Tensor Core 본격 도입 |
| Ampere | A100 | TF32/BF16/MIG | 범용 training/inference | 대규모 AI 표준 GPU |
| Hopper | H100 | FP8, Transformer Engine | LLM training/inference | FP8 시대의 대표 GPU |
| Hopper refresh | H200 | FP8, HBM3e 141GB | memory capacity/bandwidth | 긴 context/KV cache에 유리 |
| Blackwell | B200/GB200 | FP4/NVFP4, NVLink 5 | rack-scale inference/training | FP4 + AI factory 시대 |
| Blackwell Ultra | B300/GB300 | 더 큰 HBM, attention acceleration | reasoning/long-context inference | 더 큰 memory와 더 강한 inference |
| Rubin | Vera Rubin NVL72 | NVFP4, HBM4, NVLink 6 | 차세대 rack-scale AI / agentic inference | 2026년 이후 차세대 platform |


### **소비자 GPU 계열**

| 세대              | 대표     | 한 줄 요약                                  |
| --------------- | ------ | --------------------------------------- |
| Pascal          | GTX 10 | CUDA/gaming 명기, RTX 없음                  |
| Turing          | RTX 20 | RT/Tensor 소비자 최초                        |
| Turing cut-down | GTX 16 | RT/Tensor 없는 Turing                     |
| Ampere          | RTX 30 | RTX 3090 24GB가 개인 AI 연구 인기              |
| Ada             | RTX 40 | RTX 4090 24GB, 개인 연구 최상급                |
| Blackwell | RTX 50 | RTX 5090 32GB GDDR7, 5th-gen Tensor Core, DLSS 4 계열 |



{% include comments.html %}
