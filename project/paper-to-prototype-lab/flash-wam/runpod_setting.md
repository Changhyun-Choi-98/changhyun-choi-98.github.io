---
layout: post
title: "1. Runpod Server Manifest"
nav_exclude: true
section: project
subcategory: flash-wam
date: 2026-06-13
tags:
  - English
  - Korean
language: en
summary: "public-safe Runpod GPU server snapshot after initial setup"
math: true
comments: true
comment_id: "project-flash-wam-runpod-setting"
permalink: /project/paper-to-prototype-lab/flash-wam/runpod-setting/
---

<aside class="series-preface" markdown="1">

* The paper reports latency on L40S, but due to RunPod availability constraints, this project was conducted on an **A100-SXM4-80GB**. Therefore, the latency results in this project should be interpreted not as a hardware-exact reproduction of the paper’s latency results, but as **independent real-time profiling** in an A100 environment.

* 논문은 L40S 기준 latency를 보고하지만, Runpod availability 문제로 본 프로젝트는 **A100-SXM4-80GB**에서 수행했다. 따라서 본 프로젝트의 latency 결과는 paper latency의 hardware-exact reproduction이 아니라, A100 환경에서의 **independent real-time profiling**으로 해석한다.

</aside>


## TL;DR

This is a public-safe, one-time server snapshot collected from inside the current Runpod GPU server after initial setup. It records automatically verifiable hardware, storage, OS, CUDA, Python, PyTorch, and key ML package information. Pricing, billing metadata, region, template name, SSH details, public IP addresses, credentials, project repositories, checkpoints, datasets, and experiment results are intentionally excluded.

## Generation Metadata

| Item | Value |
| --- | --- |
| Generated at UTC | 2026-06-12T23:41:24Z |
| Generated at local time | 2026-06-12T23:41:24+00:00 |
| Manifest version | public-server-manifest-v1 |
| Snapshot note | One-time server snapshot created after initial Runpod setup. |

## Hardware Summary

| Item | Value | Source |
| --- | ---: | --- |
| GPU count | 1 | auto-detected |
| GPU model | NVIDIA A100-SXM4-80GB | auto-detected |
| VRAM per GPU | 80.0 GiB (81920 MiB) | auto-detected |
| NVIDIA Driver | 570.124.06 | auto-detected |
| CUDA Version from nvidia-smi | 12.8 | auto-detected |
| CPU model | AMD EPYC 7343 16-Core Processor | auto-detected |
| vCPU count | 64 | auto-detected |
| System RAM | 503Gi | auto-detected |
| /workspace storage | 2.1P | auto-detected |

## GPU Details

| GPU | Model | VRAM | Driver | CUDA from nvidia-smi | Power Limit | PCI Bus ID |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 0 | NVIDIA A100-SXM4-80GB | 80.0 GiB (81920 MiB) | 570.124.06 | 12.8 | 500.0 W | 00000000:C1:00.0 |

## CPU Details

| Item | Value |
| --- | ---: |
| Architecture | x86_64 |
| CPU model | AMD EPYC 7343 16-Core Processor |
| vCPU count | 64 |
| Sockets | 2 |
| Cores per socket | 16 |
| Threads per core | 2 |
| CPU max MHz | 3940.6250 |

## Memory

| Item | Value |
| --- | ---: |
| Total RAM | 503Gi |
| Used RAM | 26Gi |
| Available RAM | 464Gi |

## Storage

| Mount | Size | Used | Available | Use% | Filesystem |
| --- | ---: | ---: | ---: | ---: | --- |
| / | 80G | 1.5G | 79G | 2% | overlay |
| /workspace | 2.1P | 1.4P | 736T | 66% | fuse |
| /dev/shm | 58G | 0 | 58G | 0% | tmpfs |

## Operating System

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| OS ID | ubuntu |
| OS version | 22.04 |
| Kernel release | 6.8.0-52-generic |
| Machine architecture | x86_64 |

## Python Environment

| Item | Value |
| --- | --- |
| Python | 3.11.10 |
| Python executable | /usr/bin/python |
| pip | 24.2 |

## CUDA / PyTorch Runtime

| Item | Value |
| --- | --- |
| PyTorch | 2.4.1+cu124 |
| CUDA available in PyTorch | True |
| PyTorch CUDA version | 12.4 |
| cuDNN | 90100 |
| PyTorch CUDA device count | 1 |
| NVCC | release 12.4, build 12.4.131 |

## PyTorch CUDA Devices

| Device | Name | Total Memory | Compute Capability |
| ---: | --- | ---: | --- |
| 0 | NVIDIA A100-SXM4-80GB | 79.3 GiB | 8.0 |

## Key Package Versions

| Package | Version |
| --- | --- |
| torch | 2.4.1+cu124 |
| torchvision | 0.19.1+cu124 |
| torchaudio | 2.4.1+cu124 |
| numpy | 1.26.3 |
| scipy | not installed |
| pandas | not installed |
| matplotlib | not installed |
| opencv / cv2 | not installed |
| diffusers | not installed |
| transformers | not installed |
| accelerate | not installed |
| tokenizers | not installed |
| einops | not installed |
| flash_attn | not installed |
| huggingface_hub | not installed |
| safetensors | not installed |
| xformers | not installed |

## Publication Notes

- This manifest includes only information automatically detected from inside the server/container.
- This manifest is intended to be safe for public blog publication.
- Pricing, billing type, region, data center, template name, public IP address, SSH port, credentials, API tokens, project repositories, checkpoints, datasets, and experiment results are intentionally excluded.
- Project-specific reproducibility metadata should be documented separately in each project report.



{% comment %}{% include comments.html %}{% endcomment %}
