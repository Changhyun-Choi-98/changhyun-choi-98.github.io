---
layout: post
title: "Grounding Generative Policies in Physics: Optimization-Guided Diffusion for Robot Control"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-25
tags:
  - Korean
  - inference-time
  - training-free
  - diffusion-policy
language: ko
summary: "Frozen task-space diffusion policy의 DDIM sampling noise를 무작위로 뽑는 대신, robot reachability·collision·controller trackability를 만족하도록 최적화하여 cross-embodiment deployment를 수행하는 inference-time constrained diffusion method"
math: true
comments: true
comment_id: "paper-optim-guide"
permalink: /paper/briefs/optim-guide/
---

<aside class="series-preface" markdown="1">

- **Authors:** Sabrina Bodmer¹\*, René Zurbrügg¹\*, Tifanny Portela¹, Hao Ma¹, Alexandre Didier¹, Marco Hutter¹, Colin Jones²†, Melanie Zeilinger¹†
- **Affiliations:** ¹ETH Zurich, ²EPFL, \*Equal contribution, †Equal supervision, Corresponding authors: Sabrina Bodmer and René Zurbrügg
- **Links:** [arXiv](https://arxiv.org/abs/2606.24208){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-23

</aside>


## **Overview Figure**

![optim-guide_overview](/paper/briefs/images/optim-guide_overview.png)

## **Summary**
1. Task-space diffusion policy는 grasp나 trajectory의 분포 자체는 잘 학습할 수 있지만, **특정 robot arm의 reachability, collision constraint, joint/torque limit, controller tracking capability까지 보장하지는 않는다**.
2. 이 논문은 **policy를 embodiment마다 다시 학습하지 않고도, frozen generative prior의 출력을 deployment-time physical feasibility에 맞추는 것**을 목표로 한다.
3. 핵심 아이디어는 **DDIM reverse step의 stochastic perturbation $\omega_k$를 constrained optimization으로 구한 correction $\delta_k$로 대체**하는 것이다.
4. **Correction 크기를 제한하여 learned prior에서 멀어지지 않게 하면서, $J_{\mathrm{IK}}$, $J_{\mathrm{coll}}$, $J_{\mathrm{dyn}}$ 등의 feasibility cost를 [IPOPT](https://link.springer.com/article/10.1007/s10107-004-0559-y){:target="_blank" rel="noopener noreferrer"}, [Theseus](https://arxiv.org/abs/2207.09442){:target="_blank" rel="noopener noreferrer"}, [L-BFGS](https://link.springer.com/article/10.1007/BF01589116){:target="_blank" rel="noopener noreferrer"}로 최적화**한다.
5. Dexterous grasping에서는 gradient guidance 대비 최대 약 20 percentage points, pick-and-place에서는 최대 23 percentage points의 task-success 향상을 보였지만, grasp optimization은 unguided DDIM보다 46–69배 느리고 real-world 검증은 제한적이다.


### **Further Analysis**

* **이 논문의 본질은 “생성 후 projection”이 아니라 “생성 trajectory 자체를 constrained inference 문제로 다시 푸는 것”이다** 
    * 최종 output만 feasible set으로 snap하면 원래 grasp manifold를 훼손할 수 있지만, 이 방법은 전체 reverse process에서 최소 correction을 찾는다.
* **$\delta_k$에 대한 $L_2$ regularization은 단순한 heuristic이 아니라, 원래 DDIM stochastic transition의 Gaussian negative log-prior로 해석된다**
    * 즉, feasibility를 얻되 pretrained sampler가 선호하는 trajectory에서 최소한으로 이탈하려는 MAP formulation이다.
* **“Training-free”는 policy weight 관점에서는 맞지만, 시스템 전체로 보면 완전한 training-free는 아니다**
    * Grasping 구현은 각 arm마다 100,000개의 IK-labeled target으로 작은 MLP를 학습한다.
* **“Cross-embodiment”도 end-effector까지 완전히 다른 multi-robot generalization이라기보다, 동일한 task-space behavior와 gripper를 서로 다른 arm kinematics에 연결하는 arm-level transfer에 가깝다**
* **제목의 “Physics”는 full contact dynamics나 differentiable rigid-body simulation보다는 kinematics, controller authority, tracking approximation과 geometric collision을 의미한다**
    * Contact-rich physics grounding까지 해결했다고 해석하면 과장이다.



{% comment %}{% include comments.html %}{% endcomment %}
