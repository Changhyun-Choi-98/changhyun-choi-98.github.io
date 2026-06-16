---
layout: post
title: "1. Runpod Server Manifest"
nav_exclude: true
section: project
subcategory: realtime-vla-flash
date: 2026-06-15
tags:
  - English
language: en
summary: "public-safe Runpod GPU server snapshot after initial setup"
math: true
comments: true
comment_id: "project-realtime-vla-flash-runpod-setting"
permalink: /project/paper-to-prototype-lab/realtime-vla-flash/runpod-setting/
---


## **TL;DR**

This is a public-safe, one-time server snapshot collected from inside the current Runpod GPU server after initial setup. It records automatically verifiable hardware, storage, OS, CUDA, Python, PyTorch, and key ML package information. Pricing, billing metadata, region, template name, SSH details, public IP addresses, credentials, project repositories, checkpoints, datasets, and experiment results are intentionally excluded.

## **Hardware Summary**

| Item | Value | Source |
| --- | ---: | --- |
| GPU count | 1 | auto-detected |
| GPU model | NVIDIA L40S | auto-detected |
| VRAM per GPU | 46068 MiB | auto-detected |
| NVIDIA Driver | 580.159.03 | auto-detected |
| CUDA Version from nvidia-smi | 13.0 | auto-detected |
| CPU model | AMD EPYC 9554 64-Core Processor | auto-detected |
| vCPU count | 256 | auto-detected |
| System RAM | 1.0Ti | auto-detected |
| /workspace storage | 378T total, 125T available | auto-detected |

## **GPU Details**

| GPU | Model | VRAM | Driver | CUDA from nvidia-smi | Power Limit | PCI Bus ID |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 0 | NVIDIA L40S | 46068 MiB | 580.159.03 | 13.0 | 262.00 W | 00000000:C3:00.0 |

## **CPU Details**

| Item | Value |
| --- | ---: |
| Architecture | x86_64 |
| CPU model | AMD EPYC 9554 64-Core Processor |
| vCPU count | 256 |
| Sockets | 2 |
| Cores per socket | 64 |
| Threads per core | 2 |
| CPU max MHz | 3764.2019 MHz |

## **Memory**

| Item | Value |
| --- | ---: |
| Total RAM | 1.0Ti |
| Used RAM | 40Gi |
| Available RAM | 966Gi |

## **Storage**

| Mount | Size | Used | Available | Use% | Filesystem |
| --- | ---: | ---: | ---: | ---: | --- |
| / | 100G | 1.2G | 99G | 2% | overlay |
| /workspace | 378T | 253T | 125T | 67% | fuse |
| /dev/shm | 58G | 0 | 58G | 0% | tmpfs |

## **Operating System**

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04.3 LTS |
| OS ID | ubuntu |
| OS version | 24.04 |
| Kernel release | 6.17.0-29-generic |
| Machine architecture | x86_64 |

## **Python Environment**

| Item | Value |
| --- | --- |
| Python | CPython 3.12.3 |
| Python executable | /usr/local/bin/python |
| pip | 25.2 |

## **CUDA / PyTorch Runtime**

| Item | Value |
| --- | --- |
| PyTorch | 2.8.0+cu128 |
| CUDA available in PyTorch | True |
| PyTorch CUDA version | 12.8 |
| cuDNN | 91002 |
| PyTorch CUDA device count | 1 |
| NVCC | release 12.8, build V12.8.93 |

## **PyTorch CUDA Devices**

| Device | Name | Total Memory | Compute Capability |
| ---: | --- | ---: | --- |
| 0 | NVIDIA L40S | 44.39 GiB | 8.9 |

## **Key Package Versions**

| Package | Version |
| --- | --- |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| torchaudio | 2.8.0+cu128 |
| numpy | 2.1.2 |
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

## **Publication Notes**

- This manifest includes only information automatically detected from inside the server/container.
- This manifest is intended to be safe for public blog publication.
- Pricing, billing type, region, data center, template name, public IP address, SSH port, credentials, API tokens, project repositories, checkpoints, datasets, and experiment results are intentionally excluded.
- Project-specific reproducibility metadata should be documented separately in each project report.






{% comment %}{% include comments.html %}{% endcomment %}
