---
layout: post
title: "WAM4D: Fast 4D World Action Model via Spatial Register Tokens"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-15
tags:
  - Korean
  - success-rate
  - WAM
  - fine-tuning
  - auxiliary-module-training
  - component-scratch-training
language: ko
summary: "4D geometry를 inference-time output으로 직접 만들지 않고, training-time spatial register token으로 future depth를 예측하게 만들어 geometric foundation prior를 causal video-action WAM에 distill한 뒤, deploy 시 geometry branch를 제거해 action chunk를 빠르게 생성"
math: true
comments: true
comment_id: "paper-wam4d"
permalink: /paper/briefs/wam4d/
---

<aside class="series-preface" markdown="1">

- **Authors:** Ying Li¹²\*, Xiaobao Wei¹³\*, Jiajun Cao¹³†, Hao Wang¹, Xiaowei Chi², Chengyu Bai¹, Qianpu Sun¹, Jiajun Li¹, Xiaojie Zhang², Jian Tang³, Sirui Han²‡, Shanghang Zhang¹³‡
- **Affiliations:** ¹Peking University, ²The Hong Kong University of Science and Technology, ³Beijing Innovation Center of Humanoid Robotics, \*Equal Contribution, †Project Leader, ‡Equal Corresponding Author
- **Links:** [arXiv](https://arxiv.org/abs/2606.14048){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/myendless1/wam4d){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-12

</aside>


## **Overview Figure**

![wam4d_overview](/paper/briefs/images/wam4d_overview.png)
![wam4d_overview_2](/paper/briefs/images/wam4d_overview_2.png)
![wam4d_overview_3](/paper/briefs/images/wam4d_overview_3.png)

## **Summary**

1. 기존 WAM은 future video/action을 함께 모델링하지만 대부분 2D video 또는 latent space에 머물러 manipulation에 중요한 **3D spatial constraint, occluded contact geometry, free space, object extent를 충분히 반영하지 못한다**.
2. 반대로 [TesserAct](https://arxiv.org/abs/2504.20995){:target="_blank" rel="noopener noreferrer"}, [Kinema4D](https://arxiv.org/abs/2603.16669){:target="_blank" rel="noopener noreferrer"}, [X-WAM](https://arxiv.org/abs/2604.26694){:target="_blank" rel="noopener noreferrer"}류의 4D world modeling은 dense geometry를 input/output 또는 inference target으로 다루기 때문에 **action inference 시 geometric decoding 비용과 latency가 커진다**.
3. WAM4D의 핵심 아이디어는 **geometry를 deployment-time output이 아니라 training-time auxiliary readout target으로만 사용**하고, **learnable spatial register tokens가 history video feature에서 future depth를 읽어내도록 학습시키는 것**이다.
4. 모델은 **[LingBot-VA](https://arxiv.org/abs/2601.21998){:target="_blank" rel="noopener noreferrer"} 기반 causal video-action MoT backbone에 spatial register 기반 depth extraction blocks, pretrained [Depth Anything 3](https://arxiv.org/abs/2511.10647){:target="_blank" rel="noopener noreferrer"} / DA3-GIANT-1.1 geometric head, causal mixture attention mask를 붙여 video/action/depth objective를 joint optimization**한다.
5. RoboTwin 2.0과 AstriBot S1 real-world task에서 WAM4D는 spatial consistency와 real-world sub-action success를 개선하지만, RoboTwin randomized average에서는 LingBot-VA/Fast-WAM보다 낮고, WAM 계열 자체도 VLA보다 latency가 크다는 한계가 있다.






{% comment %}{% include comments.html %}{% endcomment %}
