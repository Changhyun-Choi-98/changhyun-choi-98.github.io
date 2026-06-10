---
layout: post
title: "SARM2: Multi-Task Stage Aware Reward Modeling for Self Improving Robotic Manipulation"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-10
tags:
  - Korean
  - success-rate
  - VLA
  - fine-tuning
  - auxiliary-module-training
  - MoE
language: ko
summary: "long-horizon robotic manipulation에서 VLA policy의 self-improvement를 위해, action-primitive stage estimator와 multi-gate MoE value head로 dense reward/value model을 만들고, 이를 SPIRAL의 offline-to-online residual RL data flywheel에 통합한다"
math: true
comments: true
comment_id: "paper-sarm2"
permalink: /paper/briefs/sarm2/
---

<aside class="series-preface" markdown="1">

- **Authors:** Qianzhong Chen¹, Hau Zheng¹, Justin Yu², Suning Huang¹, Jiankai Sun¹, Ken Goldberg², Chuan Wen³, Pieter Abbeel², Yide Shentu²⁴, Philipp Wu⁴, Mac Schwager¹
- **Affiliations:** ¹Stanford University, ²UC Berkeley, ³Shanghai Jiao Tong University, ⁴xdof.ai
- **Links:** [arXiv](https://arxiv.org/abs/2606.10305){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://qianzhong-chen.github.io/sarm2.github.io/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-09

</aside>

## **Overview Figure**

![sarm2_overview](/paper/briefs/images/sarm2_overview.png)
![sarm2_overview_2](/paper/briefs/images/sarm2_overview_2.png)
![sarm2_overview_3](/paper/briefs/images/sarm2_overview_3.png)

## **Summary**

1. 기존 VLA policy fine-tuning은 주로 behavior cloning(BC)에 의존해서 **고품질 demonstration이 많이 필요**하고, **policy가 demonstration distribution 근처에 묶이는 문제**가 있다.
2. 이 논문은 **long-horizon manipulation에서 dense, accurate, general한 reward model을 만들어 on-robot RL의 supervision으로 쓰는 문제**를 다룬다.
3. 핵심 아이디어는 task-specific stage label 대신 **task-agnostic action primitive vocabulary를 stage representation으로 쓰고**, **예측된 primitive group이 multi-gate Mixture-of-Experts(MMoE) value head의 gate를 선택하게 만드는 것**이다.
4. 모델은 frozen SigLIP-2 encoder, 4-layer causal Transformer action-primitive classifier, 6-layer causal Transformer value model, top-k MMoE regression head로 구성되며, **SPIRAL은 $\pi_{0.5}$에서 BC fine-tuning한 base VLA policy의 action chunk에 residual correction을 더해 reward-guided residual RL을 수행한다**.
5. 10-task reward-model benchmark에서 SARM2는 demo MSE와 rollout classification에서 baseline보다 우수하고, SPIRAL에 넣었을 때 Folding Shorts Flat 12/12, Crumpled 8/12, Cleaning Whiteboard 18/20까지 올라가 BC/offline RL/sparse reward self-improvement보다 높은 real-robot success를 보인다.




{% comment %}{% include comments.html %}{% endcomment %}
