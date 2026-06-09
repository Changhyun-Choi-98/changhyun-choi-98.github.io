---
layout: post
title: "MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-09
tags:
  - Korean
  - inference-time
  - success-rate
  - WAM
  - fine-tuning
  - component-scratch-training
language: ko
summary: "Cosmos-Predict2.5 기반 Video DiT의 intermediate denoising feature를 Motion DiT action policy에 주입하고, SONIC 기반 unified whole-body motion token으로 humanoid의 상·하체를 한 action space에 묶어 Unitree G1에서 real-time loco-manipulation을 수행"
math: true
comments: true
comment_id: "paper-motion-wam"
permalink: /paper/briefs/motion-wam/
---

<aside class="series-preface" markdown="1">

- **Authors:** Jia Zheng¹²†, Teli Ma¹²†, Yudong Fan¹, Zifan Wang¹², Shuo Yang¹\*, Junwei Liang²³\*
- **Affiliations:** ¹Mondo Robotics, ²HKUST (GZ), ³HKUST, †Equal contribution, \*Corresponding author, Co-advising
- **Links:** [arXiv](https://arxiv.org/abs/2606.09215){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-08

</aside>

## **Overview Figure**

![motionwam_overview](/paper/briefs/images/motionwam_overview.png)

## **Summary**

1. 기존 **humanoid loco-manipulation**은 보통 **high-level manipulation policy가 upper body만 세밀하게 제어하고, lower body는 low-level locomotion controller가 velocity, torso height, orientation 같은 coarse command를 추종하는 hierarchical 구조**라서 upper/lower body action space가 불일치하고 **다리가 task-driven interaction에 적극적으로 쓰이지 못한다**.
2. 동시에 **WAM**은 video dynamics prior를 policy에 넣을 수 있어 temporal coherence와 physical grounding 측면에서 유망하지만, high-dimensional video-action latent를 반복 denoising해야 해서 **real-time humanoid control에는 너무 느리다**는 문제가 있다.
3. MotionWAM은 **Video DiT + Motion DiT의 dual-DiT 구조**를 사용하되, **fully denoised future video를 만들지 않고 Video DiT branch의 intermediate denoising feature를 single forward pass로 뽑아 Motion DiT action policy에 condition으로** 넣는다.
4. 학습은 **Stage 1 egocentric video pretraining**, **Stage 2 cross-embodiment action post-training**, **Stage 3 Unitree G1 whole-body teleoperation fine-tuning**의 3단계로 구성되며, **output은 SONIC 기반 unified whole-body motion token과 continuous end-effector/gripper channel**로 구성된다.
5. 실험에서는 9개 real-world Unitree G1 loco-manipulation task에서 MotionWAM이 strongest baseline인 GR00T-N1.7의 43.9% 대비 76.1% overall success rate를 달성하고, A100 기준 Cosmos Policy보다 7배 빠른 4.9 Hz chunk-wise inference frequency를 보고한다.







{% include comments.html %}
