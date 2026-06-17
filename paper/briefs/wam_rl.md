---
layout: post
title: "WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-17
tags:
  - Korean
  - success-rate
  - WAM
  - fine-tuning
language: ko
summary: "pretrained WAM에서 actor만 RL fine-tuning하지 않고, successful online rollout으로 world model을 KL-regularized video SFT하며, actor는 imagined future와 executed future의 reconstruction consistency reward로 RL update하는 WAM post-training framework"
math: true
comments: true
comment_id: "paper-wam-rl"
permalink: /paper/briefs/wam-rl/
---

<aside class="series-preface" markdown="1">

- **Authors:** Zezhong Qian¹, Xiaowei Chi¹\*, Yu Qi², Haozhan Li³, Zhi Yang Chen¹, Shanghang Zhang¹†
- **Affiliations:** ¹State Key Laboratory of Multimedia Information Processing, School of Computer Science, Peking University, ²Northeastern University, ³Tsinghua University, \*Project Leader, †Corresponding Author
- **Links:** [arXiv](https://arxiv.org/abs/2606.17906){:target="_blank" rel="noopener noreferrer"}, [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Qian_WA-RL_World-Action_Model_Reinforcement_Learning_with_Reconstruction_Rewards_and_Online_CVPRW_2026_paper.html){:target="_blank" rel="noopener noreferrer"}, [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/papers/Qian_WA-RL_World-Action_Model_Reinforcement_Learning_with_Reconstruction_Rewards_and_Online_CVPRW_2026_paper.pdf){:target="_blank" rel="noopener noreferrer"}, [Supplementary](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/supplemental/Qian_WA-RL_World-Action_Model_CVPRW_2026_supplemental.pdf){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-16

</aside>


## **Overview Figure**

![wam-rl_overview](/paper/briefs/images/wam-rl_overview.png)

## **Summary**

1. 기존 WAM은 future observation과 action을 함께 모델링해 long-horizon decision making에 유리하지만, 대부분 expert trajectory 기반 supervised learning에 의존해 **demonstration distribution 밖의 fine-grained skill을 배우거나 interaction으로 지속 개선하기 어렵다**.
2. 이 논문은 **WAM에 RL을 넣을 때 actor만 update하면 world model latent space에 묶인 actor가 장기 누적 예측 오류를 고치지 못한다**는 문제를 다룬다.
3. 핵심 아이디어는 WAM의 주된 capability가 world model에서 나오고 actor는 latent prediction을 executable action으로 번역하는 translator에 가깝기 때문에, **world model과 actor를 함께 co-evolve시켜야 한다**는 것이다.
4. **world model을 successful online rollout으로 video self-supervised fine-tuning(SFT)하되 KL regularization으로 pretrained latent geometry를 보존하고, actor는 imagined future와 실제 실행 결과 사이의 similarity를 reconstruction-based dense reward로 삼아 policy gradient로 학습**한다.
5. 실험에서는 LIBERO-Object에서 Base 68%, actor-only $\pi_{\text{RL}}$ 78%, WAM-RL 82%, RLBench Water Plants에서 Base 19%, $\pi_{\text{RL}}$ 18%, WAM-RL 22%를 달성하며, 특히 video SFT가 failed grasp 이후 recovery behavior를 예측하게 만든다고 주장한다.



{% comment %}{% include comments.html %}{% endcomment %}
