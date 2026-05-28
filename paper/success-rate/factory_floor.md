---
layout: default
title: "A Factory-Floor Deployment Case Study of VLA Pipelines for Industrial Packaging Task: Workflow, Failures, and Lessons"
nav_exclude: true
section: paper
subcategory: success-rate
date: 2026-05-28
tags:
  - Korean
  - real-world
  - VLA
language: ko
summary: "데이터 수집·teleoperation·runtime·failure analysis 루프를 설계해서 pretrained π0.5를 실제 공장 포장 작업에 배포하는 시도, 그리고 거기서 얻은 교훈들"
math: true
comments: true
comment_id: "paper-factory-floor"
permalink: /paper/success-rate/factory-floor/
---

# **A Factory-Floor Deployment Case Study of VLA Pipelines for Industrial Packaging Task: Workflow, Failures, and Lessons**

<aside class="series-preface" markdown="1">

- **Authors:** Brian Zhu¹, Philipp Schmitt¹, Philine Meister¹, Lukas Gensler¹, Momen Khalil¹, Emmanuele Poggi¹, Johannes Hechtl¹, Carsten Braunroth¹, Kai Wurm¹, Gokul Narayanan¹, Eugen Solowjow¹, Georg von Wichert¹, Andre Scholz¹, Felix Albrecht¹, Maxmillian Metzner¹
- **Affiliations:** ¹Siemens Corporation, author list provided in Acknowledgments as Project Members
- **Pages:** [arXiv](https://arxiv.org/abs/2605.27461){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-05-25

</aside>

기존 VLA 연구들은 다양한 manipulation capability들을 보여주었지만, 대개 lab setting 중심이라 실제 공장 환경의 throughput, occlusion robustness, low latency, safety, downstream quality requirement를 충분하게 다루지 못하였다. 이 논문은 Siemens Factory의 실제 packaging task에서 pretrained [$\pi$0.5](https://arxiv.org/abs/2504.16054){:target="_blank" rel="noopener noreferrer"}를 single factory-floor task에 맞게 fine-tuning할 때 어떤 engineering effort와 실패 요인이 발생하는지를 분석한다. 저자들은 한 번에 full task 데이터를 모으는 대신, constrained scenario → constraint 제거 → unconstrained full task → targeted recovery data로 이어지는 iterative data collection / curation / fine-tuning / evaluation loop를 사용한다. 최종 fine-tuned policy의 success rate는 기대에 미치지 못했지만 저자들은 이 과정에서 얻은 lesson들을 소개했다.

## **Target Task: Industrial Packaging Task**

이 논문의 핵심 질문은 아래와 같다:

> **Pretrained VLA를 하나의 실제 산업 작업에 맞게 fine-tuning해서 배포하려면, 실제로 무엇이 필요한가? 그리고 reliability를 제한하는 bottleneck은 무엇인가?**

로봇이 해야 하는 일은 다음과 같다:

```text
1. Bin 안에 쌓여 있는 여러 개의 transparent accessory bag 중 하나를 집는다.
2. Bag 안에는 user manual과 industrial connector cable이 들어 있다.
3. 이미 primary product가 들어 있는 cardboard package의 남은 cavity에 bag을 넣는다.
4. Bag과 내부 부품이 product surface 위로 튀어나오지 않게 눌러서 box가 flat하게 닫히도록 만든다.
```

겉보기에는 위 task가 단순한 pick-and-place처럼 보이지만, 아래와 같은 어려움들이 존재한다:

| 어려움                           | 구체적 이유                                                    |
| ----------------------------- | --------------------------------------------------------- |
| Transparent object perception | 투명 비닐 bag의 경계가 camera image에서 잘 드러나지 않음                   |
| Cluttered singulation         | 여러 bag이 겹쳐 있어 하나만 집기 어려움                                  |
| Grasp point variation         | bag 방향과 내부 부품 위치에 따라 안전한 grasp point가 달라짐                 |
| Empty-region grasping         | gripper가 manual/cable을 직접 집지 않도록 bag의 “empty side”를 잡아야 함 |
| Deformable dynamics           | 비닐 bag과 내부 cable/manual의 모양이 계속 변함                        |
| Insertion precision           | package cavity에 정확히 넣어야 함                                 |
| Downstream closure constraint | 최종적으로 box가 평평하게 닫혀야 하므로 튀어나온 부품이 있으면 실패                   |

추가적으로 저자들은 policy rollout을 smooth하고 accurate하게 실행하기 위해 [asynchronous inference with real-time chunking](https://arxiv.org/abs/2506.07339){:target="_blank" rel="noopener noreferrer"}을 구현했다.

## **Workflow**

```text
1. Mock cell에서 hardware/software/teleop stack 검증
2. Teleoperate하기 쉬운 execution strategy 설계
3. Constrained task data collection
4. Manual data curation
5. π0.5 fine-tuning
6. Evaluation
7. Failure mode 분석
8. Targeted recovery data collection
9. Constraint를 하나씩 제거하며 task difficulty 증가
10. Unconstrained full-task data로 full fine-tuning
11. Factory-floor evaluation
12. Failure taxonomy와 lessons learned 정리
```

한 번에 full production setting으로 가는 것이 아니라, task difficulty를 staged curriculum처럼 올렸다.

## **HW & SW Setup**

### **Robot HW**

| 구성요소       | 내용                                                    |
| ---------- | ----------------------------------------------------- |
| Robot arms | Pair of **UR7e robots**                               |
| Grippers   | **Robotiq 2F-85** grippers                            |
| Fingertips | Plastic bag 조작을 쉽게 하기 위한 custom 3D-printed fingertips |
| Cameras    | 양팔 wrist-mounted cameras + base camera                |
| Lighting   | controlled lighting을 위한 extra light panels            |
| Fixtures   | cardboard box와 bin 위치를 고정하기 위한 fixture                |
| Compute    | Siemens Industrial PC, **RTX 5090 GPU**               |

### **Teleoperation stack**

Teleoperation은 Meta Quest 3 tracking을 이용해 robot end-effector pose를 제어하는 방식이다. 저자들은 teleoperation stack을 smooth, responsive, safe하게 만들기 위해 다음 기능들을 구현했다:

| 기능                                       | 목적                                         |
| ---------------------------------------- | ------------------------------------------ |
| Low-frequency command interpolation      | UR robot이 기대하는 500 Hz command로 변환          |
| Impedance control                        | compliance 확보                              |
| Motion planning with collision avoidance | safety 확보                                  |
| Responsive control loop                  | teleoperator의 perception-action loop 왜곡 방지 |


## **Execution Strategy**

사람이 가방을 넣을 때 자연스럽게 할 수 있는 동작 일부는 아래와 같은 이유로 VLA(robot)가 재현하기 어려웠다:

| 사람에게 쉬운 동작                  | 왜 robot/VLA에는 어려운가                               |
| --------------------------- | ------------------------------------------------ |
| Bag을 흔들어 내용물을 아래로 settle시키기 | 반복적이고 빠른 motion이며, VLA가 memory 없이 재현하기 어려움       |
| Bag을 말거나 구부려 cavity에 넣기     | 높은 dexterity가 필요하고 teleoperation consistency가 낮음 |

따라서 논문의 저자들은 teleoperation-friendly하고 VLA-friendly한 형태로 execution strategy를 설계했다:

### **1. One-arm grasp**

한쪽 arm이 bin 안의 pile에서 accessory bag 하나를 잡는다. 이때 gripper가 bag 내부 contents, 즉 manual이나 cable을 직접 집지 않도록 해야한다.

### **2. Gravity-assisted settling and insertion**

Arm이 가방을 들어올린다. 이때 gravity를 이용해 가방 안의 내용물들이 아래쪽으로 내려가게 한다. 이후 package cavity로 bag를 넣는다.

### **3. Second-arm pushing and flattening**

다른 arm이 cavity 밖으로 튀어나온 accessory를 밀어 넣고, 가방을 product 위에 눌러 box closing이 가능하도록 flatten한다.

![factory_floor_task](/paper/success-rate/images/factory_floor_task.png)
*Task Overview*
{: .figure-caption}

## **Data Collection Strategy**

논문에서 검증하려고 했던 핵심 behavior들은 아래와 같다:

| Behavior                  | 의미                                                         |
| ------------------------- | ---------------------------------------------------------- |
| Bag edge perception       | dense pile에서 transparent bag의 edge를 보고 grasp target을 찾는 능력 |
| Safe grasp / reposition   | bin wall과 충돌하거나 contents를 집을 위험이 있으면 bag을 reposition하는 능력  |
| Post-placement correction | bag을 놓은 뒤 contents가 cavity 밖에 있으면 다시 밀어 넣는 능력              |


2주라는 제한된 시간 내에 full production setting만 반복하면 실패 원인을 분석하기 어렵기 때문에, 논문의 저자들은 environment constraint를 걸어 task를 단순화한 뒤, 하나씩 constraint를 제거하는 curriculum-like data collection을 사용했다. 총 3가지 constraint를 정의했다:

| 약어     | 이름            | 의미                                                   |
| ------ | ------------- | ---------------------------------------------------- |
| **SP** | Settled Parts | bag contents가 한쪽에 몰려 있어 empty side와 filled side가 구분됨 |
| **NR** | No Reposition | bag이 bin 안에서 안전하게 바로 grasp 가능한 위치에 놓임                |
| **RC** | Reduced Count | bin 안 bag 수를 줄여 edge 구분이 쉬움                          |

위 constraint들을 조합해 구성한 data collection round는 아래와 같다:

|    Round | Episodes | Constraints                          | 목적                                                                      |
| -------: | -------: | ------------------------------------ | ----------------------------------------------------------------------- |
|        1 |      693 | SP + NR + RC                         | reduced clutter와 favorable bag configuration에서 nominal strategy 검증      |
|        2 |      199 | SP + RC                              | No Reposition 제거, corrective reposition behavior 검증                     |
| Recovery |      242 | constrained + unconstrained recovery | failed grasp, multi-bag pick, imperfect placement 등 failure recovery 보강 |
|        3 |     1401 | none                                 | dense pile, scrambled bag을 포함한 production-like setting                  |

각 data collection round 이후 모든 episode를 사람이 직접 검토했고, erroneous trajectory는 제거했다. 제거 비율은 전체 데이터의 5% 미만이다.

## **Training**

초기 round에서는 특정 behavior가 되는지 빠르게 검증하는 것이 목적이었기 때문에 LoRA fine-tuning을 사용했고, full production-like dataset을 모은 뒤 full fine-tuning으로 전환했다:

| 단계         | Fine-tuning 방식   | Batch size |           Steps | Compute                          |
| ---------- | ---------------- | --------- | -------------- | -------------------------------- |
| Round 1 이후 | LoRA fine-tuning |         32 |             30k | RTX 5090 IPC                     |
| Round 2 이후 | LoRA fine-tuning |         32 |             30k | RTX 5090 IPC                     |
| Round 3 이후 | Full fine-tuning |        128 | 60k, 약 4 epochs | Siemens internal compute cluster |

policy가 evaluation에서 최소 70% success를 달성해야 다음 data collection round로 넘어갔다.


## **Evaluation**

### **Setup**

| 항목                  | 내용                                          |
| ------------------- | ------------------------------------------- |
| Evaluation scenario | 30개의 randomly placed bag이 있는 bin을 비우는 상황    |
| Episode definition  | 하나의 bag에 대한 single pick-and-place operation |
| Time limit          | episode당 1분                                 |
| Trial 1             | Settled Parts constraint 포함                 |
| Trial 2, 3          | Fully unconstrained                         |
| Failure analysis 대상 | Trial 2와 Trial 3의 failed episodes           |
| Bag removal rule    | 성공/실패와 무관하게 episode 종료 후 target bag 제거      |


### **Results**

아래 표는 전체 episode 기준의 failure rate가 아니라, Trial 2와 Trial 3에서 실패한 episode들만 대상으로 각 failure mode가 얼마나 자주 등장했는지를 나타낸 것이다. 하나의 episode에 여러 failure mode가 동시에 포함될 수 있다.

| Failure Type                        | Trial 2 | Trial 3 | Overall |
| ----------------------------------- | ------: | ------: | ------: |
| **Bag Contents Remain on Product**  |     62% |     69% | **65%** |
| **Multiple Bags Grasped**           |     23% |     23% | **23%** |
| **Bag Not Fully Inserted into Box** |     23% |    7.7% | **15%** |
| **Poor or Failed Grasps of Bag**    |     23% |    7.7% | **15%** |

## **Lessons**

위 결과에서 알 수 있듯이, 이 논문에서의 실험들은 많은 failure를 포함한다. 이로 인해 저자들이 제시한 lessons는 다음과 같다:


1. Responsive control stack은 high-quality data collection과 performant policy를 위해 중요하다.
2. Data collection, training, evaluation은 iterative loop로 진행되어야 한다.
3. Detailed cataloging과 manual review는 dataset error를 줄이고 data segment별 영향을 이해하는 데 중요하다.
4. Qualitative failure analysis는 binary success rate보다 더 actionable하다.
5. HW setup에서 morphology gap을 줄이는 것이 중요하다.
6. Recovery data는 실제 policy-induced failure distribution에서 모아야 한다. (human-in-the-loop evaluation workflow)
7. Camera view가 policy decision에 충분한 정보를 제공하는지 검증해야 한다.
8. Memory 없는 VLA는 반복적·dexterous behavior에 약하다.
9. Binary success rate뿐 아니라 sub-task success rate, progress completion 같은 granular metric이 필요하다.


8번은 사실 [이런 논문에서](https://arxiv.org/abs/2603.03596){:target="_blank" rel="noopener noreferrer"} 해결하고 있는 것 같다. 찾아보면 더 많이 있을듯..


## **Personal Idea**

엄청나게 기발한 아이디어가 돋보이는 논문도 아니고, success rate가 높지도 않았다. 하지만 lessons를 보면서 어떻게 보면 당연하게 생각할 수도 있는 것들을 다시 한번 곱씹어볼 기회를 가졌다는 것 만으로도 의미가 있다고 생각한다. 또한 $\pi$0.5를 이렇게 학습시켰을 때 이런 task에서 어느 정도 성공하는지 감을 잡을 수 있는 것도 valuable하다고 생각한다.


{% include comments.html %}
