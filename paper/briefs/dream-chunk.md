---
layout: post
title: "DREAM-Chunk: Reactive Action Chunking with Latent World Model"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-18
tags:
  - Korean
  - inference-time
  - success-rate
  - VLA
  - auxiliary-module-training
language: ko
summary: "frozen action-chunking VLA가 샘플링한 N개 candidate chunk의 latent future를 lightweight world model로 예측하고, 매 control step마다 현재 observation과 가장 가까운 phase-aligned dreamed state의 action으로 전환해 VLA를 다시 호출하지 않고 within-chunk reactivity를 높이는 test-time scaling method"
math: true
comments: true
comment_id: "paper-dream-chunk"
permalink: /paper/briefs/dream-chunk/
---

<aside class="series-preface" markdown="1">

- **Authors:** Wenxi Chen¹, Kaidi Zhang¹\*, Chi Lin¹\*, Zhiyuan Zhang¹, Yu She¹, Yuejiang Liu², Raymond A. Yeh¹, Shaoshuai Mou¹, Yan Gu¹
- **Affiliations:** ¹Purdue University, ²Stanford University, \*Kaidi Zhang and Chi Lin made equal contributions to the experimental work
- **Links:** [arXiv](https://arxiv.org/abs/2606.18589){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://wenxichen2746.github.io/DREAM-Chunk/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-17

</aside>


## **Overview Figure**

<p style="text-align: center;">
  <img src="/paper/briefs/images/dream-chunk_overview.png" alt="dream-chunk_overview" width="50%">
</p>
![dream-chunk_overview_2](/paper/briefs/images/dream-chunk_overview_2.png)

## **Summary**

1. 기존 action chunking은 느린 VLA inference를 낮은 주기로 수행하면서 높은 control rate를 유지할 수 있지만, **chunk 후반부 action은 점점 오래된 observation에 기반하므로 stochastic dynamics, hardware execution error, partial observability에 취약하다**.
2. 이 논문은 **base VLA를 추가로 fine-tuning하거나 매 control step마다 다시 호출하지 않고도 within-chunk reactivity를 높이는 문제**를 다룬다.
3. 핵심 아이디어는 **현재 observation에서 여러 candidate action chunk를 한 번에 샘플링하고, auxiliary latent world model로 각 candidate의 phase-aligned latent rollout을 예측한 뒤, 실행 중 실제 observation과 가장 잘 맞는 candidate를 계속 선택하는 것**이다.
4. **Base policy는 고정**되며, 별도로 학습한 **observation encoder $e$와 action-conditioned latent dynamics model $f$가 candidate별 future state를 예측**한다. **매 control step마다 새 observation을 encoding하고, 현재 phase에서 가장 가까운 dreamed latent를 가진 candidate의 동일 phase action을 실행**한다.
5. Kinetix에서는 action noise가 강하고 candidate 수 $N$이 클수록 baseline 대비 상대적 이득이 커졌으며, corrective behavior가 포함된 demonstration으로 학습한 policy에서 test-time scaling 효과가 더 컸다. SO-101의 세 task에서도 성공률이 모두 향상되었고, Franka can insertion은 local $N=5$에서 10%에서 65%로 상승했지만, 더 큰 remote candidate pool에서는 sampling 및 communication latency가 이득을 상쇄했다.







{% comment %}{% include comments.html %}{% endcomment %}
