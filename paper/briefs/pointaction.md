---
layout: post
title: "PointAction: 3D Points as Universal Action Representations for Robot Control"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-03
tags:
  - Korean
  - WAM
  - fine-tuning
language: ko
summary: "pretrained video diffusion model이 RGB뿐 아니라 temporally consistent XYZ pointmap까지 생성하게 만들고, 이 3D point dynamics를 embodiment-specific diffusion action decoder가 action chunk로 변환"
math: true
comments: true
comment_id: "paper-pointaction"
permalink: /paper/briefs/pointaction/
---

<aside class="series-preface" markdown="1">

- **Authors:** Mutian Tong†, Han Jiang†\*, Qiao Feng, Lingjie Liu, Jiatao Gu
- **Affiliations:** University of Pennsylvania, †Equal contribution, \*Work done during internship at University of Pennsylvania
- **Links:** [arXiv](https://arxiv.org/abs/2606.03943){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://oriontmt.github.io/pointaction/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-02

</aside>

![pointaction_overview](/paper/briefs/images/pointaction_overview.png)
*Overview*
{: .figure-caption}

1. 기존 VAM(Video-Action Models)은 미래 RGB video rollout을 action reasoning trace로 쓰지만, **RGB만으로는 metric 3D motion, contact geometry, fine-grained spatial constraint가 under-specified되어 있어서 action grounding이 어렵다**.
2. PointAction은 **RGB-XYZ joint video generation model을 학습해 dynamic 3D pointmaps를 만들고, robot-centric points를 추출**한 뒤 PointNet-style encoder + lightweight DiT diffusion decoder로 low-level action chunk를 생성한다.
3. RoboCasa365, xArm7, YAM real robot에서 VLA/VAM baseline보다 높은 성능을 보였지만, **video model inference latency**와 **open-loop execution** 때문에 real-time closed-loop robot control 관점에서는 아직 한계가 있다.



{% include comments.html %}