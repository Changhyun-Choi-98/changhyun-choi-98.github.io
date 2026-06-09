---
layout: post
title: "τ0-WM: A Unified Video-Action World Model for Robotic Manipulation"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-02
tags:
  - Korean
  - WAM
  - success-rate
  - foundation-Model
language: ko
summary: "action generation, video prediction, action-conditioned evaluation을 하나의 shared video diffusion backbone 위에서 통합한 manipulation framework"
math: true
comments: true
comment_id: "paper-tau-0"
permalink: /paper/briefs/tau-0/
---

<aside class="series-preface" markdown="1">

- **Authors:** Pengfei Zhou²\*, Shengcong Chen²\*, Di Chen², Jiaxu Wang², Rongjun Jin², Bingwen Zhu¹², Yike Pan², Songen Gu², Kuanning Wang², Shufeng Nan², Xingyu Qiu², Chenhao Qiu², Pu Yang², Yunuo Cai¹², Jianxiong Gao², Yifan Li¹, Yanwei Fu¹², Xiangyu Yue², Zhi Chen², Jianlan Luo¹²†
- **Affiliations:** ¹Shanghai Innovation Institute, ²AGIBOT Finch, \*Equal contribution, †Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2606.01027){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://finch.agibot.com/research/tau0-wm){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/sii-research/tau-0-wm){:target="_blank" rel="noopener noreferrer"}, [Hugging Face](https://huggingface.co/sii-research/tau-0-wm){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-05-31

</aside>

## **Overview Figure**

![tau_0_overview](/paper/briefs/images/tau_0_overview.jpg)

## **Summary**

1. VAM(Video Action Model)은 current multi-view observation,  language instruction, robot state를 입력받아 **future video latent와 continuous action chunk를 함께 예측**한다.
2. ACVS(Action-Conditioned Video Simulator)는 후보 action chunk를 condition으로 future multi-view rollout과 dense task-progress reward를 예측해, **실행 전에 action의 결과를 평가**한다.
3. 추론 시에는 여러 action 후보를 샘플링하고 RCS(Re-denoising Consistency Score)로 **1차 선택한 뒤**, 불확실하면 LAR(Low-quality Action Rectification)로 ACVS가 고른 좋은 미래를 조건으로 **VAM을 다시 호출해 action을 수정**한다.



{% include comments.html %}