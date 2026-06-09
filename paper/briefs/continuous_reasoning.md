---
layout: post
title: "Continuous Reasoning for Vision-Language-Action"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-02
tags:
  - Korean
  - VLA
  - success-rate
  - fine-tuning
language: ko
summary: "VLA의 reasoning을 자연어 CoT가 아니라, 다른 VLA instance도 consume할 수 있는 WAE-regularized Gaussian continuous reasoning interface로 정의"
math: true
comments: true
comment_id: "paper-continuous-reasoning"
permalink: /paper/briefs/continuous-reasoning/
---

<aside class="series-preface" markdown="1">

- **Authors:** Yueh-Hua (Kris) Wu\*, Tatsuya Matsushima, Kei Ota
- **Affiliations:** AIRoA, \*Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2606.00229){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-05-29

</aside>

## **Overview Figure**

![cr_overview](/paper/briefs/images/cr_overview.jpg)

## **Summary**

1. 기존 language/text CoT는 task-level reasoning에는 좋지만, **robot action은 훨씬 더 fine-grained continuous control이므로 시간 granularity가 맞지 않는다**.
2. 이 논문은 **observation/instruction에서 continuous thought**를 만들고, 이를 **WAE(Wasserstein autoencoder)-regularized Gaussian latent space**로 정규화한 뒤, chunked flow-matching action generation의 **shared reasoning context**로 사용한다.
3. **EMA teacher**가 student의 latent thought를 consume해서 target action field를 예측하도록 self-verification objective를 넣었고, LIBERO-PRO와 TX-G2/HSR 실로봇에서 특히 **spatial retargeting**과 **task adaptation**이 개선된다.



{% include comments.html %}