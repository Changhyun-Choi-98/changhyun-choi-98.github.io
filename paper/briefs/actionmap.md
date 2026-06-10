---
layout: post
title: "ActionMap: Robot Policy Learning via Voxel Action Heatmap"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-08
tags:
  - Korean
  - success-rate
  - VLA
  - fine-tuning
  - component-scratch-training
language: ko
summary: "VLA의 기존 single-point action decoder를 3D translation / 3D rotation / gripper voxel heatmap action head로 교체해, action space의 geometric proximity(인접성)를 학습 신호로 활용"
math: true
comments: true
comment_id: "paper-actionmap"
permalink: /paper/briefs/actionmap/
---

<aside class="series-preface" markdown="1">

- **Authors:** Pei Yang¹\*, Hai Ci¹\*†, Yanzhe Chen¹\*, Qi Lv¹, Han Cai², Mike Zheng Shou¹‡
- **Affiliations:** ¹Show Lab, National University of Singapore, ²NVIDIA, \*Equal contribution, †Project lead, ‡Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2606.06904){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/showlab/ActionMap){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-05

</aside>

## **Overview Figure**

![actionmap_overview](/paper/briefs/images/actionmap_overview.png)


![actionmap_overview_2](/paper/briefs/images/actionmap_overview_2.png)

## **Summary**

1. 기존 VLA는 backbone, dataset, training recipe는 빠르게 커졌지만, action decoder는 여전히 autoregressive token, L1 regression, flow-matching denoising처럼 **최종적으로 하나의 point action을 예측하는 구조**가 많아 action space의 spatial structure를 충분히 활용하지 못한다.
2. 이 논문은 **end-effector action이 연속적인 기하 공간에 놓여 있고, 서로 가까운 action들이 물리적으로 비슷한 의미를 갖는다는 점을 action head 설계에 직접 넣으려** 한다.
3. 핵심 아이디어는 action을 직접 회귀하지 않고, **translation voxel grid, rotation voxel grid, gripper distribution 위의 probability heatmap으로 예측**한 뒤 **Gaussian-blob soft target과 cross-entropy로 학습**하고 **top-k soft-argmax로 continuous action을 복원**하는 것이다.
4. 이 head는 OpenVLA-OFT의 L1 regression head와 $\pi_{0.5}$의 flow-matching action expert를 대체하는 **drop-in component**로 제안되며, backbone 자체는 크게 바꾸지 않는다.
5. 실험에서는 LIBERO와 real-world Franka에서 OpenVLA-OFT 및 $\pi_{0.5}$ baseline보다 success rate, data efficiency, convergence 면에서 개선을 보였고, 특히 OpenVLA-OFT + low-data / long-horizon setting에서 gain이 크다.



{% comment %}{% include comments.html %}{% endcomment %}
