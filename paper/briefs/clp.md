---
layout: post
title: "Finetuning Vision-Language-Action Models Requires Fewer Layers Than You Think"
nav_exclude: true
section: paper
subcategory: briefs
date: 2026-06-22
tags:
  - Korean
  - inference-time
  - VLA
  - fine-tuning
language: ko
summary: "pretrained π0·GR00T-N1.5·SmolVLA의 VLM backbone과 continuous-action head에서 Centered Kernel Alignment로 표현이 거의 변하지 않는 연속 Transformer layer를 찾아 fine-tuning 전에 정적으로 제거하고, 남은 작은 모델을 downstream fine-tuning하여 학습·추론 비용을 함께 줄이는 VLA structural pruning method"
math: true
comments: true
comment_id: "paper-clp"
permalink: /paper/briefs/clp/
---

<aside class="series-preface" markdown="1">

- **Authors:** Gia-Binh Nguyen¹², Trong-Bao Ho², Thien-Loc Ha², Khoa Vo³, Philip Lund Møller⁴, Quang T. Nguyen², Long Dinh¹², Tuan Dam⁵, Vu Duong¹, Tung M. Luu⁶, Trung Le⁷, Tran Nguyen Le⁴, Minh Vu¹², An Thai Le¹²¹³, Ngan Le³, Daniel Sonntag⁸⁹, James Zou¹², Jan Peters⁹¹³, Duy M. H. Nguyen⁹¹⁰¹¹†, Ngo Anh Vien¹²†
- **Affiliations:** ¹Center for AI Research, VinUniversity, ²VinRobotics, ³University of Arkansas, ⁴Technical University of Denmark, ⁵Hanoi University of Science and Technology, ⁶KAIST, ⁷Monash University, ⁸Oldenburg University, ⁹DFKI, ¹⁰University of Stuttgart, ¹¹IMPRS-IS, ¹²Stanford University, ¹³Technische Universität Darmstadt, †Project Leads
- **Links:** [arXiv](https://arxiv.org/abs/2606.20246){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://clpvla.github.io/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-18

</aside>


## **Overview Figure**

![clp_overview](/paper/briefs/images/clp_overview.png)

## **Summary**

1. 기존 VLA efficiency method들은 token pruning·cache처럼 **inference만 줄이거나**, dynamic layer skipping을 위해 router와 추가 objective를 학습해야 하므로 **downstream fine-tuning 비용과 architectural complexity를 동시에 해결하지 못한다**.
2. 이 논문은 [$\pi_{0}$](https://arxiv.org/abs/2410.24164){:target="_blank" rel="noopener noreferrer"}와 [GR00T-N1.5](https://arxiv.org/abs/2503.14734){:target="_blank" rel="noopener noreferrer"} 같은 **modern continuous-control VLA에서 fine-tuning 전에 상당수 Transformer layer를 제거해도 성능을 유지할 수 있는지**를 다룬다.
3. 핵심 아이디어는 **작은 calibration set에 대해 한 번 forward하고 인접 layer hidden representation의 Centered Kernel Alignment(CKA)를 측정하여, representation 변화가 거의 없는 연속 구간을 redundant block으로 판정**하는 것이다.
4. **각 high-CKA block의 첫 layer를 anchor로 유지하고 나머지 후보 중 similarity가 높은 layer를 정적으로(statically) 삭제**한 뒤, **축소된 VLM backbone과 flow/diffusion action head를 기존 native objective로 fine-tuning**한다.
5. 세 VLA backbone과 LIBERO·RoboCasa·SimplerEnv, 4종 robot embodiment의 10개 real-world task에서 약 28–32% inference latency 감소와 대체로 baseline 수준의 success rate를 보였으며, 특히 저데이터 setting에서는 작은 모델이 regularizer처럼 작동할 가능성을 제시한다.


<aside class="content-summary" markdown="1">
CKA는 [Centered Kernel Alignment](https://arxiv.org/abs/1905.00414){:target="_blank" rel="noopener noreferrer"}로, 직관적으로는 두 hidden representation $\mathbf{H}\_t$, $\mathbf{H}\_{\tau}$가 비슷한 구조를 갖는지 비교하는 similarity measure이다. 단순하게 pixel 차이를 보는 것이 아니라 hidden state 내부의 similarity pattern을 비교하기 때문에 semantic stability를 보는 데 더 적합하다. ([ElegantVLA](/paper/inference/elegant-vla/)에서도 나왔던 개념)
</aside>

### **Further Analysis**


* **Static compression**
    * CLP(CKA-guided Layer Pruning)는 매 입력마다 layer를 선택하는 dynamic routing이 아니다. Pruning 이후에는 항상 동일한 얕은 network를 실행하므로 runtime router나 early-exit decision이 없다.

* **CKA는 pruning 그 자체보다 layer 후보를 고르는 diagnostic이다**
    * 높은 CKA가 곧 “그 layer가 필요 없다”는 수학적 증명은 아니다. 논문도 이를 인정하며, 실제 성능은 pruning 이후 fine-tuning으로 검증한다.

* **Fine-tuning이 단순한 마무리가 아니라 CLP의 필수 구성요소다**
    * 위 figure는 pruning 직후 latent geometry가 크게 변하고 fine-tuning 후 다시 형성되는 양상을 보여준다. 따라서 이미 deploy 중인 VLA를 fine-tuning 없이 즉시 압축하는 방법으로 해석하면 안 된다.

* **low-data 성능 향상은 관찰된 현상이지만 원인은 아직 가설이다**
    * 저자들은 작은 capacity가 downstream noise에 대한 overfitting을 줄이는 implicit regularization이라고 해석한다. 그러나 capacity-matched baseline, stronger regularization, teacher distillation과의 비교가 없어 인과적으로 입증되지는 않았다.

* **논문의 가장 강한 결과는 accuracy 향상보다 cost–performance trade-off다**
    * Full-data LIBERO에서는 $\pi_{0}$가 94.6%→93.9%, GR00T-N1.5가 93.9%→93.0%로 소폭 하락한다. 즉 “더 좋은 policy”라기보다 “거의 같은 policy를 더 싸게 fine-tune하고 실행”하는 결과가 중심이다.




{% comment %}{% include comments.html %}{% endcomment %}
