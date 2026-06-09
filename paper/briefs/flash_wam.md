---
layout: post
title: "Flash-WAM: Modality-Aware Distillation for World Action Models"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-05
tags:
  - Korean
  - WAM
  - inference-time
  - fine-tuning
  - distillation
language: ko
summary: "WAM의 video/action diffusion denoising을 각각의 noise regime에 맞게 다르게 distill해서, WAM을 거의 teacher 성능에 가깝게 유지하면서 real-time chunk-level control이 가능한 수준까지 가속하는 step-distillation method"
math: true
comments: true
comment_id: "paper-flash-wam"
permalink: /paper/briefs/flash-wam/
---

<aside class="series-preface" markdown="1">

- **Authors:** Arman Akbari¹\*, Ci Zhang², Arash Akbari¹, Lin Zhao¹, Yixiao Chen¹, Weiwei Chen³, Xuan Zhang¹, Geng Yuan², Yanzhi Wang¹³
- **Affiliations:** ¹Northeastern University, ²University of Georgia, ³EmbodyX Inc., \*Correspondence to: akbari.ar@northeastern.edu
- **Links:** [arXiv](https://arxiv.org/abs/2606.05254){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://flashwam.github.io/){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/NU-World-Model-Embodied-AI/Flash-WAM){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-03

</aside>

## **Overview Figure**

![flash_wam_overview](/paper/briefs/images/flash_wam_overview.png)

## **Summary**

1. 기존 WAM은 미래 video latent와 robot action chunk를 함께 생성하기 때문에 강한 manipulation 성능을 보이지만, 매 control chunk마다 video denoising과 action denoising을 수십 step 수행해야 해서 **real-time closed-loop control**에 부적합하다.
2. 대표 WAM인 LingBot-VA는 RoboTwin 2.0에서 **한 chunk를 생성하는 데 video 25 step, action 50 step을 사용하며 총 8.1초**가 걸린다.
3. 기존 LCM/consistency distillation을 그대로 video-action joint diffusion에 적용하면 **video와 action이 서로 다른 SNR-shifted noise schedule을 쓰기 때문에 action 쪽 low-$\sigma$ 구간에서 gradient signal이 사라져** 성능이 크게 붕괴한다.
4. Flash-WAM은 **action stream에는 low-noise regime에 맞는 linear-gradient-scaling consistency function**, **video stream에는 high-noise regime에 안정적인 LCM/Karras-style variance-preserving consistency function**을 사용한다.
5. 그 결과 LingBot-VA를 **1 video step / 1 action step** 또는 **1 video step / 2 action step**으로 압축하면서 RoboTwin, LIBERO, Unitree G1 real-world setup에서 기존 naive distillation보다 훨씬 높은 success rate를 유지한다.



{% include comments.html %}
