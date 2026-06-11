---
layout: post
title: "Dynamic Execution Horizon Prediction for Chunk-based Robot Policies"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-11
tags:
  - Korean
  - inference-time
  - success-rate
  - diffusion-policy
  - scheduler-training
  - auxiliary-module-training
language: ko
summary: "pretrained action-chunking robot policy의 action generator는 완전히 고정하고, 현재 observation과 예측된 action chunk를 보고 “이번에 몇 step을 open-loop로 실행할지”를 PPO로 학습하는 lightweight execution-horizon predictor"
math: true
comments: true
comment_id: "paper-dehp"
permalink: /paper/briefs/dehp/
---


<aside class="series-preface" markdown="1">

- **Authors:** Yuchi Zhao¹²†, Miroslav Bogdanovic³†, Arjun Sohal¹, Liyu Tao¹, Kourosh Darvish³, Alán Aspuru-Guzik¹²³⁴⁵⁶⁷⁸¹⁰, Florian Shkurti¹², Animesh Garg⁹
- **Affiliations:** ¹Department of Computer Science, University of Toronto, ²Vector Institute for Artificial Intelligence, ³Acceleration Consortium, ⁴Department of Chemistry, University of Toronto, ⁵Department of Materials Science & Engineering, University of Toronto, ⁶Department of Chemical Engineering & Applied Chemistry, University of Toronto, ⁷Institute of Medical Science, University of Toronto, ⁸Canadian Institute for Advanced Research (CIFAR), ⁹Georgia Institute of Technology, ¹⁰NVIDIA, †Both worked super hard!!!
- **Links:** [arXiv](https://arxiv.org/abs/2606.11408){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://dehp-chunking.github.io/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-09

</aside>

## **Overview Figure**

![dehp_overview](/paper/briefs/images/dehp_overview.png)
![dehp_overview_2](/paper/briefs/images/dehp_overview_2.png)

## **Summary**

1. 기존 chunk-based robot policy는 Diffusion Policy, flow policy, VLA처럼 여러 action을 한 번에 예측하지만, 실제 실행 시에는 고정된 개수의 action만 실행하는 fixed execution horizon에 의존한다.
2. 이 고정 horizon은 free-space motion에서는 효율적일 수 있지만, grasp alignment, insertion, contact-rich manipulation처럼 feedback이 중요한 단계에서는 open-loop 실행이 길어져 작은 오차가 누적되는 문제가 있다.
3. 이 논문은 execution horizon 자체를 state-dependent decision으로 보고, pretrained base chunk policy가 만든 full action chunk와 현재 observation을 입력으로 받아 실행 길이 $h \in \\\{1, \dots, H\\\}$를 선택하는 Dynamic Execution Horizon Prediction, DEHP를 제안한다.
4. 학습은 base policy $\pi\_{\text{act}}$를 frozen으로 유지한 채, categorical horizon policy $\pi\_{\text{len}}(h \| s, \mathbf{a}\_{1:H})$와 state-value critic만 online PPO로 학습하며, variable horizon rollout을 semi-Markov decision process로 정식화한다.
5. 실험에서는 state-based Diffusion Policy를 base policy로 사용하고, multi-stage peg insertion, bimanual needle-syringe insertion, FurnitureBench one-leg / round-table assembly에서 tuned fixed-horizon baseline보다 높은 success rate를 보였으며, learned horizon이 free-space에서는 길고 alignment/insertion에서는 짧아지는 해석 가능한 패턴을 보였다.



{% comment %}{% include comments.html %}{% endcomment %}
