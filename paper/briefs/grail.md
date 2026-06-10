---
layout: post
title: "GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-04
tags:
  - Korean
  - success-rate
  - data-generation
  - fine-tuning
  - auxiliary-module-training
  - sim2real
language: ko
summary: "3D asset과 video foundation model prior를 이용해 humanoid loco-manipulation용 4D human-object interaction 데이터를 완전 디지털로 생성하고, 이를 Unitree G1용 tracking policy와 egocentric visual policy로 변환해 실제 로봇에 배포하는 data-generation / sim-to-real framework"
math: true
comments: true
comment_id: "paper-grail"
permalink: /paper/briefs/grail/
---

<aside class="series-preface" markdown="1">

- **Authors:** Tianyi Xie¹²\*, Haotian Zhang¹\*, Jinhyung Park¹\*, Zi Wang¹\*, Bowen Wen¹, Jiefeng Li¹, Xueting Li¹, Qingwei Ben¹, Haoyang Weng¹, Yufei Ye¹, David Minor¹, Tingwu Wang¹, Chenfanfu Jiang², Sanja Fidler¹, Jan Kautz¹, Linxi Fan¹, Yuke Zhu¹, Zhengyi Luo¹†, Umar Iqbal¹†, Ye Yuan¹†
- **Affiliations:** ¹NVIDIA, ²UCLA, \*Co-First Authors, †Project Leads
- **Links:** [arXiv](https://arxiv.org/abs/2606.05160){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://research.nvidia.com/labs/dair/grail/){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/NVlabs/GRAIL){:target="_blank" rel="noopener noreferrer"}, [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-03

</aside>

## **Overview Figure**

![grail_data](/paper/briefs/images/grail_data.jpeg)
![gail_trajectory](/paper/briefs/images/grail_trajectory.jpeg)

## **Summary**

1. 기존 humanoid loco-manipulation 데이터는 teleoperation, motion capture, in-the-wild video reconstruction에 의존하는데, 이는 **물리적 세팅 변경, actor instrumentation, robot operation, monocular depth/scale/contact ambiguity 때문에 scale-up이 어렵다**.
2. GRAIL은 **“먼저 3D scene을 완전히 지정하고, 그 위에서 VFM(Video Foundation Model)이 interaction video를 만들게 한 뒤, 알려진 geometry/camera/scale/depth를 이용해 4D HOI(4D Human-Object Interaction, 시간에 따라 변하는 human pose와 object pose를 함께 복원하는 문제)를 복원한다”**는 방향으로 문제를 바꾼다.
3. 핵심 아이디어는 uncontrolled video를 4D로 억지 복원하는 대신, **privileged 3D configuration 안에서 video prior를 interaction prior로만 사용**하고 **metric reconstruction은 known 3D assets와 scene geometry에 anchor**하는 것이다.
4. 이후 **복원된 4D HOI를 Unitree G1에 retargeting**하고, **pretrained whole-body controller SONIC 위에 object-aware latent adaptor와 scene-aware tracker를 학습**해 task-general tracking policy를 만든다.
5. 결과적으로 GRAIL은 20,000개 이상의 generated sequence를 만들고, 이 데이터만으로 학습한 egocentric visual policy를 실제 Unitree G1에 배포해 object pick-up 84%, stair-climbing 90% real-world success를 보고한다.



{% comment %}{% include comments.html %}{% endcomment %}
