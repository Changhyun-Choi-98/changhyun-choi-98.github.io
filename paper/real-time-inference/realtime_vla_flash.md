---
layout: default
title: "Realtime-VLA FLASH: Speculative Inference Framework for Diffusion-based VLAs"
nav_exclude: true
section: paper
subcategory: real-time-inference
date: 2026-05-21
tags:
  - Korean
language: ko
summary: "diffusion / flow-matching 기반 VLA의 full inference의 대부분의 replanning round를 가벼운 draft + 빠른 verification으로 대체"
math: true
comments: true
comment_id: "paper-realtime-vla-flash"
permalink: /paper/real-time-inference/realtime-vla-flash/
---

# **Realtime-VLA FLASH: Speculative Inference Framework for Diffusion-based VLAs**

<aside class="series-preface" markdown="1">

- **Authors:** Jiahui Niu¹²*, Kefan Gu³⁴, Yucheng Zhao⁴†, Shengwen Liang¹‡, Tiancai Wang⁴‡, Xing Hu¹, Ying Wang¹, Huawei Li¹
- **Affiliations:** ¹State Key Lab of Processors, Institute of Computing Technology, CAS, ²University of Chinese Academy of Sciences, ³Nanjing University, ⁴Dexmal, *This work was done during the internship at Dexmal, †Project lead, ‡Corresponding author
- **Pages:** [arXiv](https://arxiv.org/abs/2605.13778){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://dexmal.github.io/realtime-vla-flash/){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/dexmal/realtime-vla-flash){:target="_blank" rel="noopener noreferrer"}, [Hugging Face](https://huggingface.co/Dexmal/RealtimeVLA-Flash){:target="_blank" rel="noopener noreferrer"}
- **Submitted:** 2026-03-13

</aside>

이 논문은 Diffusion-based Vision-Language-Action 모델(dVLAs)이 high latency로 인해 real-time deployment에서 제한된다는 문제를 **speculative inference framework**로 해결한다. 이 개념은 LLM의 speculative decoding과 컨셉을 공유하는데, 가벼운 draft model이 후보 action들을 만들어주면 기존의 무거운 모델이 이것을 parallel하게 verify하는 것이다. 하지만 dVLA는 continuous action chunk를 생성하고 explicit likelihood가 없기 때문에, draft token을 큰 모델이 확률적으로 accept/reject하는 것이 아니라 *draft action chunk가 main action expert의 flow field와 locally consistent한지* 검사한다는 점이 다르다.


{% include comments.html %}
