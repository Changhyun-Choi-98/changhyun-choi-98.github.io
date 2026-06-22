---
layout: post
title: "ENPIRE: Agentic Robot Policy Self-Improvement in the Real World"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-22
tags:
  - Korean
  - success-rate
  - training-data
  - VLA
language: ko
summary: "coding agent가 실제 로봇의 reset → rollout → verification → policy/code refinement research loop를 직접 운영하고, 여러 robot–agent worker가 Git으로 실험 지식을 공유하면서 task policy를 자동 개선하게 만든 physical autoresearch harness"
math: true
comments: true
comment_id: "paper-enpire"
permalink: /paper/briefs/enpire/
---

<aside class="series-preface" markdown="1">

- **Authors:** Wenli Xiao¹²\*, Jia Xie²\*, Tonghe Zhang²\*, Haotian Lin²\*, Letian “Max” Fu³, Haoru Xue³, Jalen Lu², Yi Yang², Cunxi Dai², Zi Wang¹, Jimmy Wu¹, Guanzhi Wang¹, S. Shankar Sastry³, Ken Goldberg³, Linxi “Jim” Fan¹†, Yuke Zhu¹†, Guanya Shi²†
- **Affiliations:** ¹NVIDIA, ²CMU, ³UC Berkeley, \*Equal contribution, †Equal advising
- **Links:** [arXiv](https://arxiv.org/abs/2606.19980){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://research.nvidia.com/labs/gear/enpire/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-18

</aside>


## **Overview Figure**

![enpire_overview](/paper/briefs/images/enpire_overview.png)

## **Summary**

1. 기존 real-world robot learning은 **demonstration 수집, 실패 판정, 환경 reset, algorithm tuning까지 상당한 human supervision이 필요해 scale-up의 병목이 된다**.
2. ENPIRE는 **coding agent가 실제 로봇을 대상으로 반복 가능한 physical hypothesis-testing loop를 운영하게 만드는 것**을 목표로 한다.
3. 핵심 아이디어는 먼저 **human feedback으로 safety constraints·automatic reset·binary verifier를 포함한 environment API를 만들고 고정**한 뒤, **coding agent가 그 API를 통해 policy algorithm과 training recipe를 자율적으로 수정**하는 것이다.
4. 시스템은 **Environment, Policy Improvement, Rollout, Evolution의 네 모듈로 구성**되며, **여러 robot–agent pair가 서로 다른 가설을 비동기적으로 시험하고 Git을 통해 성공적인 변경을 공유**한다.
5. 저자들은 Push-T, pin insertion, GPU insertion, zip-tie cutting에서 높은 성공률과 fleet scaling에 따른 wall-clock time 감소를 보였지만, 더 많은 agent를 사용할수록 robot utilization은 낮아지고 token 비용은 super-linear하게 증가한다. 

### **Further Analysis**

* 이 논문의 핵심 novelty는 새 policy architecture가 아니라, robot policy research process 전체를 executable closed loop로 바꾼 것이다.
* 일반적인 VLA 논문이 $(\text{image, language, state}) \rightarrow \text{action}$ 을 개선한다면, ENPIRE는 더 바깥쪽의 $\text{task specification} \rightarrow \text{experiment design} \rightarrow \text{training} \rightarrow \text{real rollout} \rightarrow \text{analysis} \rightarrow \text{code update}$를 자동화한다.
* 가장 중요한 engineering contribution은 policy보다 environment interface다. 실제 세계에서는 성공 판정과 reset이 자동화되지 않으면 아무리 좋은 coding agent도 반복 실험을 진행할 수 없다.
* “Autonomous”라는 표현은 Stage 1 이후에 한정해 이해해야 한다. 초기 safety boundary 설정, 몇 분 분량의 success/failure examples, human critique은 직접 정해줘야 한다.
* 실험에서 큰 성능 향상은 새로운 learning algorithm보다는 BC regularization, demo mixing, batch size, controller compensation 같은 known recipe를 올바르게 조합한 결과에 가깝다.
* 따라서 이 논문은 “AI가 새로운 robot learning theory를 발견했다”기보다, AI가 real-world robotics engineer의 반복 실험 업무를 상당 부분 수행할 수 있다는 실증으로 보는 것이 정확하다.




{% comment %}{% include comments.html %}{% endcomment %}
