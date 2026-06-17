---
layout: post
title: "Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement"
nav_exclude: true
section: paper
subcategory: inference
date: 2026-06-17
tags:
  - Korean
  - inference-time
  - success-rate
  - VLA
  - fine-tuning
  - Writing
language: ko
summary: "pretrained generalist robot policy를 stochastic generator로 두고, VLM 기반 visual verifier로 여러 action chunk 중 가장 그럴듯한 후보를 inference-time에 선택한 뒤, 성공한 verifier-guided rollout을 다시 behavior cloning으로 fine-tuning하는 “test-time steering + autonomous data flywheel” method"
math: true
comments: true
comment_id: "paper-veritas"
permalink: /paper/inference/veritas/
---

<aside class="series-preface" markdown="1">

- **Authors:** Mingtong Zhang, Dhruv Shah
- **Affiliations:** Princeton University
- **Links:** [arXiv](https://arxiv.org/abs/2606.18247){:target="_blank" rel="noopener noreferrer"}, [Project Page](https://veritas-improvement.github.io/){:target="_blank" rel="noopener noreferrer"}, [RSS 2026](https://roboticsconference.org/program/papers/79/){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-06-16

</aside>


## **Overview Figure**

![veritas_overview](/paper/inference/images/veritas_overview.png)
![veritas_overview_2](/paper/inference/images/veritas_overview_2.png)

## **Top-Down Summary**

1. 기존 robot foundation model / VLA는 **대규모 human demonstration에 크게 의존하고, human-in-the-loop correction이나 expert relabeling은 비용이 선형적으로 증가한다는 문제**가 있다.
2. 이 논문은 **추가 human demonstration 없이, 배포 중인 robot policy가 자신의 경험에서 성공 trajectory를 만들고 이를 학습 데이터로 재사용하는 self-improvement 문제**를 다룬다.
3. 핵심 아이디어는 **pretrained policy를 여러 action chunk를 샘플링하는 generator로 보고, VLM이 만든 pixel-space waypoint trace를 기준으로 candidate action을 평가하는 gradient-free visual verifier를 붙여 Best-of-N action selection을 수행**하는 것이다.
4. **Inference 때는 policy parameter를 고정하고 `sample → verify → select → execute`를 반복**하며, 이후 **성공한 verified rollout만 모아 $\mathcal{D}\_{\text{auto}}$를 만들고 standard behavior cloning objective로 policy를 fine-tuning**한다.
5. 실험에서는 simulation과 real-world DROID setup에서 inference-time steering이 base policy보다 성공률을 높이고, verified self-generated data로 fine-tuning한 policy가 human expert demonstration과 비슷한 data efficiency를 보인다.

### **Closed-loop control pipeline에서의 위치**

```text
Observation + instruction
      ↓
Pretrained VLA policy samples N action chunks
      ↓
Visual verifier scores candidate chunks
      ↓
Best-of-N selected chunk
      ↓
Low-level robot execution for H steps
      ↓
New observation → repeat
```

VERITAS는 policy output과 robot execution 사이에 들어가는 inference-time action selector이다.

### **Main Idea**

```text
기존 가정:
  VLA는 한 번 query해서 action을 실행한다.

문제점:
  generative policy는 좋은 action을 낮은 확률로 알고 있을 수 있지만,
  greedy / single sample deployment에서는 그 후보가 선택되지 않을 수 있다.

새 관점:
  policy를 deterministic controller가 아니라 stochastic generator로 보자.
  여러 action chunk를 sample하고, 외부 verifier가 task-aligned / physically plausible action을 고르자.

구조 반영:
  VLM이 initial observation + instruction으로 visual waypoint trace를 만든다.
  각 candidate action chunk의 projected EEF/gripper trajectory를 trace와 비교한다.
  가장 높은 verifier score의 action chunk를 실행한다.

학습 반영:
  성공한 verified rollout을 D_auto로 모아 behavior cloning fine-tuning한다.

Robot behavior 변화:
  즉시 성능은 Best-of-N filtering으로 올라가고,
  장기적으로는 verifier가 고른 행동 패턴이 policy weight에 distill된다.
```




## **Experiments**


## **Limitations**

### **1. Inference Compute Cost**
repeated sampling을 통해 task performance를 얻기 때문에 latency-critical application에서는 계산 비용이 부담될 수 있다.

### **2. Static Visual Trace**
현재 verifier는 episode 시작 시 생성한 static visual trace에 의존하므로 quasi-static manipulation에는 충분하지만, scene이 빠르게 바뀌는 dynamic environment에서는 어려울 수 있다.

### **3. Policy prior limitation**
verifier는 policy가 제안한 후보 중 best를 고를 뿐이므로, pretrained policy의 exploration prior 안에 좋은 후보가 없으면 개선할 수 없다.

### **Additional Points**
1. Visual trace는 task success와 같지 않다
    * EEF/gripper가 VLM waypoint trace를 따라가도 실제 object를 grasp하지 못하거나, contact force가 부족하거나, object가 미끄러지면 실패할 수 있다. Pixel-space path consistency는 success의 proxy이지 success 자체가 아니다.

2. 3D geometry / occlusion / camera calibration 취약성
    * verifier는 pixel-space trace와 EEF pixel을 비교하므로 camera calibration, perspective distortion, occlusion, depth ambiguity에 취약할 수 있다. 논문도 real-world setup에서 reliable visual verification을 위해 calibrated front-facing camera를 추가했다고 설명한다.

3. Long-horizon task에는 static trace가 부족할 수 있다
    * drawer opening, tool use, deformable object manipulation, multi-stage assembly처럼 intermediate state가 크게 바뀌는 task에서는 initial VLM trace가 빠르게 부정확해질 수 있다.

4. Failure data를 버리는 구조
    * 성공 rollout만 BC target으로 쓰면 positive-only imitation이 된다. 실패 trajectory에서 “무엇을 피해야 하는지”를 배우는 contrastive signal은 약하다.

5. Verifier hacking 가능성
    * policy가 fine-tuning을 반복하면 실제 task success보다 verifier score에 맞는 trajectory를 더 많이 생성할 수 있다. 예를 들어 waypoint를 따라가지만 object interaction은 실패하는 behavior가 강화될 가능성이 있다.

6. Safety / collision constraint 부재
    * visual trace proximity만으로는 collision, joint limit, force safety, human proximity 같은 safety constraint를 충분히 보장하지 못한다. Safety-critical deployment에는 별도 safety layer가 필요하다.

7. Real-world task 수가 제한적
    * real-world는 2개 policy에 대해 policy당 2개 task, 총 4개 task로 평가된다. “general robot self-improvement” claim을 완전히 뒷받침하기에는 task diversity가 아직 제한적이다.

8. Compute-success scaling이 포화됨
    * 논문은 $N > 8$ 이후 performance가 saturate한다고 보고한다. 단순 Best-of-N scaling만으로는 무한한 improvement를 기대하기 어렵다.



{% comment %}{% include comments.html %}{% endcomment %}
