---
layout: post
title: "1. Runpod Server Manifest"
nav_exclude: true
section: project
subcategory: flash-wam
date: 2026-06-12
tags:
  - English
language: en
summary: "public-safe Runpod GPU server snapshot after initial setup"
math: true
comments: true
comment_id: "project-flash-wam-runpod-setting"
permalink: /project/paper-to-prototype-lab/flash-wam/runpod-setting/
---


## TL;DR

This is a public-safe, one-time server snapshot collected from inside the current Runpod GPU server after initial setup. It records automatically verifiable hardware, storage, OS, CUDA, Python, PyTorch, and key ML package information. Pricing, billing metadata, region, template name, SSH details, public IP addresses, credentials, project repositories, checkpoints, datasets, and experiment results are intentionally excluded.

## Generation Metadata

| Item | Value |
| --- | --- |
| Generated at UTC | 2026-06-12T10:23:29+00:00 |
| Generated at local time | 2026-06-12T10:23:29+00:00 |
| Manifest version | public-server-manifest-v1 |
| Snapshot note | This manifest is a one-time server snapshot created after initial Runpod setup. |

## Hardware Summary

| Item | Value | Source |
| --- | --- | --- |
| GPU count | 1 | auto-detected |
| GPU model | NVIDIA L40S | auto-detected |
| VRAM per GPU | 44.99 GiB | auto-detected |
| NVIDIA Driver | 570.195.03 | auto-detected |
| CUDA Version from nvidia-smi | 12.8 | auto-detected |
| CPU model | AMD EPYC 9354 32-Core Processor | auto-detected |
| vCPU count | 128 | auto-detected |
| System RAM | 1.5Ti | auto-detected |
| /workspace storage | 1.2P | auto-detected |

## GPU Details

| GPU | Model | VRAM | Driver | CUDA from nvidia-smi | Power Limit | PCI Bus ID |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | NVIDIA L40S | 44.99 GiB | 570.195.03 | 12.8 | 350.00 W | 00000000:64:00.0 |

## CPU Details

| Item | Value |
| --- | --- |
| Architecture | x86_64 |
| CPU model | AMD EPYC 9354 32-Core Processor |
| vCPU count | 128 |
| Sockets | 2 |
| Cores per socket | 32 |
| Threads per core | 2 |
| CPU max MHz | unavailable |

## Memory

| Item | Value |
| --- | --- |
| Total RAM | 1.5Ti |
| Used RAM | 179Gi |
| Available RAM | 1.3Ti |

## Storage

| Mount | Size | Used | Available | Use% | Filesystem |
| --- | --- | --- | --- | --- | --- |
| / | 80G | 1.2G | 79G | 2% | overlay |
| /workspace | 1.2P | 742T | 460T | 62% | fuse |
| /dev/shm | 100G | 0 | 100G | 0% | tmpfs |

## Operating System

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| OS ID | ubuntu |
| OS version | 22.04 |
| Kernel release | 6.8.0-60-generic |
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
| NVCC | 12.4 (build cuda_12.4.r12.4/compiler.34097967_0) |

## PyTorch CUDA Devices

| Device | Name | Total Memory | Compute Capability |
| --- | --- | --- | --- |
| 0 | NVIDIA L40S | 44.40 GiB | 8.9 |

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
