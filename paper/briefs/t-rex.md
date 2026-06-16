---
layout: post
title: "T-Rex: Tactile-Reactive Dexterous Manipulation"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-16
tags:
  - Korean
  - success-rate
  - VLA
  - foundation-model
  - fine-tuning
  - auxiliary-module-training
  - training-data
language: ko
summary: "tactile-free human egocentric pretraining으로 얻은 visuomotor prior를 tactile-rich robot mid-training으로 contact dynamics에 맞춘 뒤, slow action expert와 fast tactile expert를 cascaded flow matching으로 연결해 action chunk 내부에서도 tactile feedback에 반응하는 tactile-reactive dexterous VLA"
math: true
comments: true
comment_id: "paper-t-rex"
permalink: /paper/briefs/t-rex/
---

<aside class="series-preface" markdown="1">

- **Authors:** Dantong Niu¹²\*, Zhuoyang Liu¹\*, Zekai Wang¹\*, Boning Shao¹, Zhao-Heng Yin¹, Anirudh Pai¹, Yuvan Sharma¹, Stefano Saravalle⁵, Ruijie Zheng², Jing Wang², Ryan Punamiya², Mengda Xu², Yuqi Xie², Yunfan Jiang²³, Letian Fu¹, Konstantinos Kallidromitis⁴, Matteo Gioia⁵⁶, Junyi Zhang¹, Jiaxin Ge¹, Haiwen Feng¹, Fabio Galasso⁵⁶, Wei Zhan¹, David M. Chan¹, Yutong Bai¹, Roei Herzig¹, Jiahui Lei¹, Fei-Fei Li³, Ken Goldberg¹, Jitendra Malik¹, Pieter Abbeel¹, Yuke Zhu², Danfei Xu², Jim (Linxi) Fan², Trevor Darrell¹
- **Affiliations:** ¹UC Berkeley, ²NVIDIA, ³Stanford, ⁴Panasonic, ⁵La Sapienza University, ⁶ItalAI, \*Equal Contribution
- **Links:** [arXiv](https://arxiv.org/abs/2606.17055){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://tactile-rex.github.io/){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/ZhuoyangLiu2005/T-Rex){:target="_blank" rel="noopener noreferrer"}, [Dataset Page](https://tactile-rex.github.io/dataset/){:target="_blank" rel="noopener noreferrer"}, [Hugging Face Dataset](https://huggingface.co/datasets/zekaiwang/trex_dataset){:target="_blank" rel="noopener noreferrer"}, [HF Pretrain Checkpoint](https://huggingface.co/miniFranka/T-Rex_pretrain_mecka22k_epoch1){:target="_blank" rel="noopener noreferrer"}, [HF Midtrain Checkpoint](https://huggingface.co/miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-15

</aside>


## **Overview Figure**

![t-rex_overview](/paper/briefs/images/t-rex_overview.png)
![t-rex_overview_2](/paper/briefs/images/t-rex_overview_2.png)

## **Summary**


1. 기존 VLA / dexterous manipulation policy는 대부분 vision 중심이라 force variation, micro-slip, deformation 같은 **tactile signal을 충분히 활용하지 못하고, tactile을 넣더라도 static encoder나 task-specific imitation learning에 머무르는 문제**가 있다.
2. T-Rex는 **contact-rich dexterous manipulation에서 필요한 high-frequency tactile reaction을 foundation-style VLA에 통합**하는 문제를 다룬다.
3. 핵심 아이디어는 **tactile-free human egocentric video pretraining으로 broad visuomotor prior를 얻고**, **100h tactile-synchronized robot data로 tactile-grounded mid-training을 수행**한 뒤, **inference에서는 slow action expert와 fast tactile expert를 cascaded flow matching으로 연결하**는 것이다.
4. 모델은 **latent expert, action expert, tactile expert로 구성된 Mixture-of-Transformer-Experts (MoT) 구조**이며, **action expert는 약 5 Hz로 coarse action chunk를 만들고 tactile expert는 약 20 Hz로 cached visual-language context를 재사용해 tactile-conditioned refinement를 수행**한다.
5. 실험에서는 12개 real-world tactile-reactive task에서 평균 성공률 65%를 기록해 가장 강한 baseline인 EgoScale 35%보다 +30 percentage points 높았지만, screw lightbulb, open lock, extract card, apply toothpaste 같은 tight contact / force-sensitive / long-horizon task에서는 object collision, slipping, excessive force, sliding misalignment가 여전히 남아 있다.



{% comment %}{% include comments.html %}{% endcomment %}
