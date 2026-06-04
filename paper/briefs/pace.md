---
layout: post
title: "PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-02
tags:
  - Korean
  - VLA
  - inference-time
  - training-free
language: ko
summary: "action chunking robot policy에서 고정 execution horizon 대신, predicted action chunk의 low-speed valley를 phase boundary로 사용해 매 query마다 실행 길이를 동적으로 선택하는 training-free test-time execution 방법"
math: true
comments: true
comment_id: "paper-pace"
permalink: /paper/briefs/pace/
---

<aside class="series-preface" markdown="1">

- **Authors:** Junnan Nie¹, Jiayi Li², Jiachen Zhang¹, Junyi Lao¹, Chenghao Liu¹, Tianle Zhang², Songfang Huang¹†
- **Affiliations:** ¹Peking University, ²JD Explore Academy, †Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2606.00537){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-05-30

</aside>

![pace_overview](/paper/briefs/images/pace_overview.png)
*Overview*
{: .figure-caption}

1. 기존 Diffusion Policy, VLA 계열의 action chunking은 한 번 query할 때 여러 future action을 예측하지만, 그중 몇 step을 실제로 실행할지는 보통 **고정 horizon `H`**로 정한다.
2. 논문은 이 `H`가 **task-dependent하고 non-monotonic**해서, 하나의 고정값으로는 안정적인 deployment rule이 되기 어렵다는 점을 보인다.
3. PACE는 predicted action chunk의 smoothed speed profile에서 **low-speed valley를 phase boundary로 검출해 그 지점까지의 prefix만 실행**하여 simulation 및 real-robot에서 고정 horizon보다 높은 성공률을 얻는다.



{% include comments.html %}