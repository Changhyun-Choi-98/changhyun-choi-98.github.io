---
layout: post
title: "1. Runpod Server Manifest"
nav_exclude: true
section: project
subcategory: flash-wam
date: 2026-06-15
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
| Generated at UTC | 2026-06-15T15:02:53Z |
| Generated at local time | 2026-06-15T15:02:53+00:00 |
| Manifest version | public-server-manifest-v1 |
| Snapshot note | This manifest is a one-time server snapshot created after initial Runpod setup. |

## Hardware Summary

| Item | Value | Source |
| --- | ---: | --- |
| GPU count | 1 | auto-detected |
| GPU model | NVIDIA L40S | auto-detected |
| VRAM per GPU | 45.0 GiB | auto-detected |
| NVIDIA Driver | 550.127.05 | auto-detected |
| CUDA Version from nvidia-smi | 12.4 | auto-detected |
| CPU model | AMD EPYC 7702 64-Core Processor | auto-detected |
| vCPU count | 256 | auto-detected |
| System RAM | 1.0Ti | auto-detected |
| /workspace storage | 311T total, 140T available | auto-detected |

## GPU Details

| GPU | Model | VRAM | Driver | CUDA from nvidia-smi | Power Limit | PCI Bus ID |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 0 | NVIDIA L40S | 45.0 GiB | 550.127.05 | 12.4 | 350 W | 00000000:24:00.0 |

## CPU Details

| Item | Value |
| --- | ---: |
| Architecture | x86_64 |
| CPU model | AMD EPYC 7702 64-Core Processor |
| vCPU count | 256 |
| Sockets | 2 |
| Cores per socket | 64 |
| Threads per core | 2 |
| CPU max MHz | 2183.5930 |

## Memory

| Item | Value |
| --- | ---: |
| Total RAM | 1.0Ti |
| Used RAM | 53Gi |
| Available RAM | 945Gi |

## Storage

| Mount | Size | Used | Available | Use% | Filesystem |
| --- | ---: | ---: | ---: | ---: | --- |
| / | 300G | 1.2G | 299G | 1% | overlay |
| /workspace | 311T | 172T | 140T | 56% | fuse |
| /dev/shm | 58G | 0 | 58G | 0% | tmpfs |

## Operating System

| Item | Value |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| OS ID | ubuntu |
| OS version | 22.04 |
| Kernel release | 5.15.0-97-generic |
| Machine architecture | x86_64 |

## Python Environment

| Item | Value |
| --- | --- |
| Python | 3.11.10 |
| Python executable | /usr/bin/python |
| pip | pip 24.2 |

## CUDA / PyTorch Runtime

| Item | Value |
| --- | --- |
| PyTorch | 2.4.1+cu124 |
| CUDA available in PyTorch | True |
| PyTorch CUDA version | 12.4 |
| cuDNN | 90100 |
| PyTorch CUDA device count | 1 |
| NVCC | CUDA 12.4 (V12.4.131) |

## PyTorch CUDA Devices

| Device | Name | Total Memory | Compute Capability |
| ---: | --- | ---: | --- |
| 0 | NVIDIA L40S | 44.5 GiB | 8.9 |

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
