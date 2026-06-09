---
layout: post
title: "VLAMotor: Test-Guided Enhancement of Vision-Language-Action Models via Agent-Based Data Synthesis"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-02
tags:
  - Korean
  - VLA
  - success-rate
  - fine-tuning
language: ko
summary: "training distribution에서 멀고 서로 중복되지 않는 테스트 케이스로 VLA 실패를 적극적으로 찾고, 그 실패 trajectory를 VLM agent가 성공 trajectory로 고쳐 fine-tuning data로 쓰는 failure-driven VLA enhancement framework"
math: true
comments: true
comment_id: "paper-vla-motor"
permalink: /paper/briefs/vla-motor/
---

<aside class="series-preface" markdown="1">

- **Authors:** Zeqin Liao¹, Peifan Ren², Zixu Gao², Hongyu Gong², Lianyu Hu¹\*, Wenbing Tang³, Yuhong Nan², Zibin Zheng², Yang Liu¹
- **Affiliations:** ¹School of Computing and Data Science, Nanyang Technological University, ²School of Software Engineering, Sun Yat-sen University & GuangDong Engineering Technology Research Center of Blockchain, China, ³Northwest A&F University, \*Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2606.00053){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/scuama/VLAMotor){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-05-16

</aside>

## **Overview Figure**

![vla_motor_overview](/paper/briefs/images/vla_motor_overview.png)

## **Summary**

1. VLA는 training data coverage 밖의 edge-case에서 자주 실패하므로, VLAMotor는 fused hidden state 기준으로 **training sample과의 distance가 크고 uncertainty가 높은 test case를 우선 선택**하고, redundancy elimination으로 서로 중복되는 실패 케이스를 줄인다.
2. 선택된 failure trajectory는 **VLM-based agent**가 semantic abstraction으로 해석하고, perception/motion/interaction repair skill sequence로 바꾼 뒤 IK와 motion planning으로 **실행 가능한 성공 trajectory를 만든다**.
3. 이렇게 자동 생성된 **성공 trajectory를 원래 data와 합쳐 fine-tuning**하면 simulation에서 전체 success rate가 49.25%p 개선되고, real-world Franka 실험에서도 π0.5가 0/80에서 46/80 성공으로 개선된다.



{% include comments.html %}