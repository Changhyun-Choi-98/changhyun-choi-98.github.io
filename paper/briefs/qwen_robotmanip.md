---
layout: post
title: "Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-17
tags:
  - Korean
  - success-rate
  - foundation-model
  - VLA
  - training-data
language: ko
summary: "Qwen-VL 기반 VLA에 canonical state/action alignment, camera-frame EEF action, in-context policy adaptation, Human-to-Robot synthesis를 결합해 heterogeneous robot manipulation data를 coherent하게 scale하고 OOD task/scene·instruction·cross-embodiment generalization을 끌어올린 robot manipulation foundation model"
math: true
comments: true
comment_id: "paper-qwen-robotmanip"
permalink: /paper/briefs/qwen-robotmanip/
---

<aside class="series-preface" markdown="1">

- **Authors:** Qwen Team — Core Contributors: Haoqi Yuan\*, Zhixuan Liang\*, Anzhe Chen\*, Ye Wang\*, Haoyang Li\*, Pei Lin\*, Yiyang Huang\*, Zixing Lei\*, Tong Zhang\*, Jiazhao Zhang, Jie Zhang, Jingyang Fan, Gengze Zhou, Qihang Peng, Chenxu Lv, Xiaoyue Chen, An Yang, Fei Huang, Junyang Lin, Dayiheng Liu, Jingren Zhou, Chenfei Wu†, Xiong-Hui Chen†; Contributors: Jinhui Ye, Sicheng Xie, Hale Yin, Xudong Guo, Shuai Bai, Lulu Hu, Minying Zhang, Shurui Li, Wenhu Xiao, Yue Wang, Kun Yan, Xiao Xu, Jiahao Li, Xuancheng Ren
- **Affiliations:** Qwen Team, Alibaba Cloud, \*Equal contribution, †Corresponding authors
- **Links:** [arXiv](https://arxiv.org/abs/2606.17846){:target="_blank" rel="noopener noreferrer"}, [Blog](https://qwen.ai/blog?id=qwen-robotmanip){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/QwenLM/Qwen-RobotManip){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-16

</aside>


## **Overview Figure**

![Qwen-RobotManip_overview](/paper/briefs/images/Qwen-RobotManip_overview.png)
![Qwen-RobotManip_overview_2](/paper/briefs/images/Qwen-RobotManip_overview_2.png)
![Qwen-RobotManip_overview_3](/paper/briefs/images/Qwen-RobotManip_overview_3.png)
![Qwen-RobotManip_overview_4](/paper/briefs/images/Qwen-RobotManip_overview_4.png)

## **Summary**


1. 기존 VLA 모델들은 LIBERO, RoboTwin 같은 standard in-domain benchmark에서는 높은 점수를 보이지만, **다른 embodiment, 다른 camera, 다른 scene, 다른 instruction으로 넘어가면 일반화가 급격히 약해지는 문제**가 있다.
2. 이 논문은 **로봇 데이터가 embodiment, coordinate frame, action space, teleoperation setup별로 서로 달라서 단순히 데이터를 많이 섞는 것만으로는 scaling law가 생기지 않는다**는 문제를 다룬다.
3. 핵심 아이디어는 **alignment first, then scale**로, canonical 80D state-action vector, per-dimension binary mask, camera-frame delta EEF pose, structured embodiment prompt, in-context policy adaptation을 통해 **여러 robot morphology의 데이터를 하나의 일관된 물리 표현으로 맞춘 뒤 대규모 pretraining을 수행**하는 것이다.
4. 모델은 [Qwen3.5-4B](https://qwen.ai/blog?id=qwen3.5){:target="_blank" rel="noopener noreferrer"} / [Qwen-VL](https://arxiv.org/abs/2511.21631){:target="_blank" rel="noopener noreferrer"} 계열 **vision-language backbone과 flow-matching DiT action expert**로 구성되며, **manipulation data와 vision-language data를 dual-stream으로 co-training**하고, **downstream domain에서는 SFT 또는 mixed post-training으로 adaptation**한다.
5. 약 38,100시간 manipulation corpus와 OOD benchmark suite에서 RT-C2R Hard 69.4% vs. $\pi_{0.5}$ 47.9%, RoboTwin-IF 72.2% vs. 49.6%, RoboTwin-XE EEF 23.9% vs. 7.5%처럼 OOD robustness, instruction following, cross-embodiment transfer를 크게 끌어올렸지만, Human-to-Robot 합성 artifact, simulation 중심 OOD 평가, fixed action chunk와 latency 한계는 남아 있다.



{% comment %}{% include comments.html %}{% endcomment %}
