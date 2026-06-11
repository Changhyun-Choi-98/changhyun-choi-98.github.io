---
layout: post
title: "DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action model"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-11
tags:
  - Korean
  - success-rate
  - inference-time
  - VLA
  - fine-tuning
  - component-scratch-training
language: ko
summary: "VLA의 synchronous clock 가정이 contact-rich manipulation의 multi-rate sensor structure와 맞지 않는다고 보고, modality별 asynchronous latent buffer + gated cross-attention으로 X-VLA를 100 Hz controller 기반 closed-loop execution에 맞춘다"
math: true
comments: true
comment_id: "paper-dam-vla"
permalink: /paper/briefs/dam-vla/
---

<aside class="series-preface" markdown="1">

- **Authors:** Pankhuri Vanjani¹, Zhuoyue Li¹, Jakub Suliga¹, Moritz Reuss², Gianluca Geraci¹, Xinkai Jiang¹, Rudolf Lioutikov¹³
- **Affiliations:** ¹Intuitive Robots Lab (IRL), Karlsruhe Institute of Technology (KIT), ²NVIDIA, ³Robotics Institute of Germany
- **Links:** [arXiv](https://arxiv.org/abs/2606.12105){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://intuitive-robots.github.io/DAM-VLA/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-10

</aside>

## **Overview Figure**

![damvla_overview](/paper/briefs/images/damvla_overview.png)
![damvla_overview_2](/paper/briefs/images/damvla_overview_2.png)

## **Summary**

1. 기존 VLA는 vision-language pretraining에서 온 single synchronous clock 가정을 그대로 사용해, **느린 vision은 과도하게 재처리하고 빠른 force/torque contact transient는 놓치는 문제**가 있다.
2. 이 논문은 physical interaction에서 **modality마다 의미 있는 update rate와 temporal horizon이 다르다**는 점, 예를 들어 force/torque는 100–500 Hz 수준의 contact transient를 담고 RGB는 훨씬 느리게 변한다는 점에 주목한다.
3. 핵심 아이디어는 각 modality를 하나의 token sequence로 강제 concat하지 않고, **per-modality latent buffer를 sensor rate별로 갱신한 뒤 action expert가 매 inference step에서 이를 읽게 하는 것**이다.
4. 모델 구조는 **[X-VLA backbone](https://arxiv.org/abs/2510.10274){:target="_blank" rel="noopener noreferrer"} 위에 visual memory pathway와 force/torque pathway를 추가**한다. 새로운 modality token을 pretrained self-attention에 직접 concat하지 않고, **gated cross-attention, GCA residual pathway로 action tokens만 조절**한다.
5. Franka Panda 기반 7개 real-world contact-rich task에서 DAM-VLA는 평균 성공률 95.2%를 달성해 strongest synchronous baseline인 X-VLA25의 40.95%를 크게 넘었고, 100 Hz controller에서 smooth하고 reactive한 실행을 보였다.
6. 다만 force/torque는 action chunk 내부를 직접 수정하는 feedback controller로 쓰이는 것이 아니라 representation을 보강하는 용도라, very contact-heavy task에서는 mid-chunk correction 한계가 남는다.


{% comment %}{% include comments.html %}{% endcomment %}