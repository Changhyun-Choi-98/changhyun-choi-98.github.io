---
layout: post
title: "Cosmos 3: Omnimodal World Models for Physical AI"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-03
tags:
  - Korean
  - WAM
  - success-rate
  - foundation-Model
language: ko
summary: "language, image, video, audio, action을 하나의 Mixture-of-Transformers (MoT) 기반 omnimodal world model로 통합해, VLM·video generator·forward/inverse dynamics·robot policy를 하나의 Physical AI backbone으로 다루는 NVIDIA의 대규모 foundation model"
math: true
comments: true
comment_id: "paper-cosmos3"
permalink: /paper/briefs/cosmos3/
---

<aside class="series-preface" markdown="1">

- **Authors:** NVIDIA¹
- **Affiliations:** NVIDIA, ¹Contributors and acknowledgments are listed in Appendix G
- **Links:** [Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://research.nvidia.com/labs/cosmos-lab/cosmos3/){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/nvidia/cosmos){:target="_blank" rel="noopener noreferrer"}, [Cosmos-Framework](https://github.com/nvidia/cosmos-framework){:target="_blank" rel="noopener noreferrer"}, [Hugging Face](https://huggingface.co/collections/nvidia/cosmos3){:target="_blank" rel="noopener noreferrer"}
- **Technical Report Released:** 2026-06-01

</aside>

## **Overview Figure**

![cosmos3_overview](/paper/briefs/images/cosmos3_overview.png)

## **Summary**

1. 기존 Physical AI pipeline은 VLM, video world model, VLA/WAM, forward dynamics model을 따로 붙이는 구조라서 비효율적인데, Cosmos 3는 이를 **AR Reasoner + Diffusion Generator 구조로 통합**한다.
2. 핵심은 **AR token subsequence와 diffusion token subsequence를 한 sequence로 묶고**, Reasoner는 causal attention, Generator는 AR+DM 전체에 full attention을 하도록 설계한 **dual-tower MoT**다.
3. 로봇 관점에서 가장 중요한 부분은 **action을 별도 head가 아니라 core modality로 넣어 forward dynamics, inverse dynamics, policy를 같은 sequence modeling 문제로 만든 점**이며, DROID policy는 4 diffusion steps로 32개 joint-position action을 15Hz에서 생성하도록 post-training된다.

| 기존 별도 모델                        | Cosmos 3에서의 역할                     |
| ------------------------------- | ---------------------------------- |
| VLM                             | AR Reasoner mode                   |
| Text-to-Image / Video generator | Diffusion Generator mode           |
| Forward dynamics model          | action-conditioned video denoising |
| Inverse dynamics model          | video-conditioned action denoising |
| VLA / WAM policy                | joint action-video denoising       |
| Synthetic data generator        | post-trained generator             |
| Robot policy                    | DROID post-trained policy          |

![cosmos3_MoT](/paper/briefs/images/cosmos3_MoT.png)



{% include comments.html %}