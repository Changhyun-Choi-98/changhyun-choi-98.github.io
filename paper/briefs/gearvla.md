---
layout: post
title: "GEAR-VLA: Learning Geometry-Aware Action Representations for Generalizable Robotic Manipulation"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-09
tags:
  - Korean
  - success-rate
  - VLA
  - fine-tuning
  - auxiliary-module-training
  - component-scratch-training
  - cross-embodiment
language: ko
summary: "Qwen2.5-VL 기반 VLA에 latent action token K/V cache-conditioned stop-gradient DiT flow action expert, VGGT 기반 3D spatial encoder, relative end-effector action 기반 embodiment canonicalization을 결합해 unseen object / background shift / pretraining-unseen robot embodiment transfer를 개선하는 geometry-aware manipulation policy"
math: true
comments: true
comment_id: "paper-gear-vla"
permalink: /paper/briefs/gear-vla/
---

<aside class="series-preface" markdown="1">

- **Authors:** Yuan Zhang¹³\*, Shiqi Zhang²\*, Yedong Shen², Shuai Dong², Jiajun Deng², Xin Zhang², Yuxuan Gao², Jiajia Wu³, Xin Nie³, Zhiyuan Cheng³, Jianmin Ji², Yanyong Zhang²†, Xingyi Zhang¹†, Jia Pan²†
- **Affiliations:** ¹Anhui University, ²University of Science and Technology of China, ³iFLYTEK, \*Equal contribution, †Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2606.08530){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/babynabeauty/GEAR-VLA){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-07

</aside>

## **Overview Figure**

![gearvla_overview](/paper/briefs/images/gearvla_overview.png)


![gearvla_overview_2](/paper/briefs/images/gearvla_overview_2.png)

## **Summary**

1. 기존 VLA는 action tokenization, 3D spatial feature, cross-embodiment learning을 각각 다루지만, **low-level trajectory token에 overfitting**되거나, **3D feature가 VLM semantic space와 어긋나**거나, **robot-specific prompt/head가 shared policy representation을 오염시키**는 문제가 있다.
2. 이 논문은 **semantically grounded + geometry-aware + embodiment-shareable action representation을 학습**하는 것이 real-world manipulation generalization의 핵심이라고 주장한다.
3. 핵심 아이디어는 먼저 **VLM을 embodied reasoning / [FAST](https://arxiv.org/abs/2501.09747){:target="_blank" rel="noopener noreferrer"}-style action token / video-derived latent action ID로 pretrain**한 뒤, **latent action token의 K/V cache만 DiT-based continuous action expert에 넘기고 stop-gradient를 걸어 continuous action loss가 VLM backbone을 직접 업데이트하지 못하게 하는 것**이다.
4. 여기에 **frozen 2D VLM visual pathway와 trainable VGGT 3D spatial encoder를 zero-initialized connector로 결합**하고, robot별 차이는 **embodiment-specific state projector와 embodiment-invariant relative end-effector action**으로 low-level interface에 가둔다.
5. LIBERO 평균 98.7%, zero-shot LIBERO-Plus 88.7%, RoboTwin 2.0 clean/randomized 91.1%/89.9%, AgileX real-world bimanual 85.9%, pretraining-unseen LDT-01 lightweight adaptation 81.0%, 212 unseen object universal grasping 90.1%를 달성한다.






{% comment %}{% include comments.html %}{% endcomment %}
