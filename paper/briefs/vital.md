---
layout: post
title: "Inference-time Policy Steering via Vision and Touch"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-16
tags:
  - Korean
  - inference-time
  - success-rate
  - WAM
  - diffusion-policy
  - auxiliary-module-training
language: ko
summary: "frozen diffusion robot policy의 weights는 바꾸지 않고, action-conditioned visuo-tactile latent world model로 후보 action chunk의 future outcome을 예측한 뒤, long-horizon vision으로 global action mode를 선택하고 short-horizon touch로 local contact execution을 diffusion editing하는 inference-time steering method"
math: true
comments: true
comment_id: "paper-vital"
permalink: /paper/briefs/vital/
---

<aside class="series-preface" markdown="1">

- **Authors:** Yilin Wu, Zilin Si, Zeynep Temel, Oliver Kroemer, Andrea Bajcsy
- **Affiliations:** Carnegie Mellon University
- **Links:** [arXiv](https://arxiv.org/abs/2606.14981){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://yilin-wu98.github.io/vital_website/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-12

</aside>


## **Overview Figure**

![vital_overview](/paper/briefs/images/vital_overview.png)

## **Summary**

1. 기존 inference-time steering은 candidate action을 visual future로 rollout하고 VLM/reward로 검증하는 방식이 많았지만, pipetting, wiping, insertion 같은 contact-rich manipulation에서는 **force, slip, pressure, insertion alignment 같은 핵심 성공 요인이 이미지로 충분히 보이지 않는다**.
2. 이 논문은 **vision은 global semantic progress / mode selection에 강하고**, **touch는 local contact quality / force-sensitive execution에 강하다**는 observation에서 출발한다.
3. 핵심 아이디어는 **multimodal reward를 하나로 섞지 않고, bi-level optimization으로 나누어 먼저 visual sampling-and-verification으로 long-horizon action mode를 선택하고, 그 선택된 action anchor의 앞부분을 tactile-guided diffusion editing으로 짧게 수정**하는 것이다.
4. 이를 위해 **frozen [DINOv3](https://arxiv.org/abs/2508.10104){:target="_blank" rel="noopener noreferrer"} / [AnyTouch2](https://arxiv.org/abs/2602.09617){:target="_blank" rel="noopener noreferrer"} encoder 위에 action-conditioned visuo-tactile latent world model을 학습**하고, **[ROBOMETER](https://arxiv.org/abs/2603.02115){:target="_blank" rel="noopener noreferrer"} 기반 visual verifier**와 **text-conditioned tactile latent reward**를 사용한다.
5. 실제 Franka robot의 wiping, insertion, pipette transfer 3개 task에서 ViTaL은 논문 기준 base policy 대비 overall success +51% improvement를 달성하고, unimodal steering 대비 최소 33%, naive multimodal fusion 대비 최소 20% 높은 성능을 달성한다.




{% comment %}{% include comments.html %}{% endcomment %}
