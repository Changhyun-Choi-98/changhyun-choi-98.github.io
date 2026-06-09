---
layout: post
title: "Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-03
tags:
  - Korean
  - VLA
  - inference-time
  - training-free
language: ko
summary: "last denoising step들에서 clean-action estimate들의 variance를 future action별 stability proxy로 사용해, 안정적인 action prefix만 실행하고 고분산 구간 전에 replan"
math: true
comments: true
comment_id: "paper-dvac"
permalink: /paper/briefs/dvac/
---

<aside class="series-preface" markdown="1">

- **Authors:** Xiangdong Feng¹³\*, Yuxuan Cheng²³\*, Chen Shi², Boyao Han⁴², Yuxuan Yan⁵³, Yitong Hong⁶³, Zhuotao Tian⁷³, Li Jiang²³†
- **Affiliations:** ¹Beijing Institute of Technology, ²The Chinese University of Hong Kong, Shenzhen, ³Shenzhen Loop Area Institute, ⁴Hunan University, ⁵Xi’an Jiaotong University, ⁶Renmin University of China, ⁷Harbin Institute of Technology, Shenzhen, \*Equal contribution, †Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2606.03847){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-02

</aside>

## **Overview Figure**

![dvac_overview](/paper/briefs/images/dvac_overview.png)

## **Summary**

1. 기존 action chunking은 **fixed execution horizon**을 쓰기 때문에, free-space motion에서는 너무 자주 replan하고 contact/precision phase에서는 너무 긴 open-loop horizon로 실행할 수 있다.
2. DVAC는 **denoising 과정의 마지막 $L$ steps에서 각 future action index $k$의 clean-action estimate variance $V\_{s}(k)$를 계산**하고, rolling statistics 기반 (adaptive) threshold $\tau_s=\mu_s+\alpha\sigma_s$를 넘어가기 전 prefix만 실행한다.
3. $\pi_{0.5}$ 기반 실험에서 LIBERO success를 94.8% → 98.0%로 올리고 replanning count를 32.6 → 18.6으로 줄였으며, RoboTwin, CALVIN, real-world tasks에서도 전반적인 이득을 보인다.



{% include comments.html %}