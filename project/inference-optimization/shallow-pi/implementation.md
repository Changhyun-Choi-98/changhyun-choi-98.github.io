---
layout: default
title: "1. Shallow-π implementation"
nav_exclude: true
section: project
subcategory: shallow-pi
date: 2026-05-20
tags:
  - Korean
  - Python
language: ko
summary: "π0 distillation을 통해 Shallow-π 구현 완료"
math: true
comments: true
comment_id: "project-shallow-pi-shallow-pi-implementation"
permalink: /project/inference-optimization/shallow-pi/shallow-pi-implementation/
---

# **1. Shallow-π implementation**

<aside class="series-preface" markdown="1">

- **OS:** Ubuntu 22.04.5 LTS, Linux kernel 5.15.0
- **CPU:** 2 × AMD EPYC 9354 32-Core Processor
  - 64 physical cores / 128 threads total
- **Memory:** 503 GiB RAM
- **GPU:** 7 × NVIDIA L40S
  - ~48 GB VRAM per GPU
  - Used 5 out of 7 GPUs for training
  - ~240 GB total GPU memory used for the training run
- **NVIDIA Driver / CUDA:** NVIDIA Driver 580.82.07 / CUDA 13.0
- **Storage:**
  - 2 TB NVMe SSD for the system drive
  - ~10.4 TB ext4 workspace/data storage
- **Usage context:** Shared multi-user research server
  - This machine is not a dedicated benchmark node.
  - Training used 5 out of 7 NVIDIA L40S GPUs, as listed above.
  - CPU, system memory, storage I/O, and OS-level background load may still be affected by other users.

</aside>

[Shallow-π의 공식 구현체](https://github.com/icsl-Jeon/openpi){:target="_blank" rel="noopener noreferrer"}를 이용해서 [π0](https://arxiv.org/abs/2410.24164){:target="_blank" rel="noopener noreferrer"}를 distill한 버전인 Shallow-π를 구현했다. 

이전의 distillation과 비교되는 Shallow-π의 대략적인 컨셉은 다음 사진과 같다:

![Shallow-π concept](/project/inference-optimization/shallow-pi/images/scope.png)
*Figure source: [Shallow-π: Knowledge Distillation for Flow-based VLAs](https://arxiv.org/abs/2601.20262){:target="_blank" rel="noopener noreferrer"}.*
{: .figure-caption}

GPU VRAM을 효율적으로 사용하기 위해서 BF16 데이터 타입을 사용했고, 하나의 GPU에 batch size가 64까지 올라가는 것을 확인하여 5개의 GPU로 batch size 320으로 distillation을 진행했다(총 30,000 step). Distillation 동안 수집한 metric 정보는 다음과 같다(다른 연구실 인원들과 같이 쓰는 서버이기 때문에, 학습 도중에 다른 workload들이 생겼다가 없어졌다가 했을 것이다):

![distillation_metric](/project/inference-optimization/shallow-pi/images/distillation_metrics.png)
*pi0_libero_l06 → pi0_libero teacher를 distillation해서 만든 Gemma depth 6짜리 shallow student*
{: .figure-caption}

제대로 distillation이 되었는지 확인하기 위해, 학습이 완료된 30k-step checkpoint를 사용해 `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` 총 4개의 LIBERO task suite에서 평가를 진행했다. Success rate는 아래와 같다:

| Task Suite | Episodes | Success Rate |
|---|---:|---:|
| libero_spatial | 500 | 89.4% |
| libero_object | 500 | 88.8% |
| libero_goal | 500 | 81.8% |
| libero_10 | 500 | 45.6% |
| Average | 2,000 | 76.4% |

{% include comments.html %}
