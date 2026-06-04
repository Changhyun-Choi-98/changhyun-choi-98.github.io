---
layout: post
title: "Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments"
nav_exclude: true
section: paper
subcategory: success-rate
date: 2026-06-01
tags:
  - Korean
  - VLA
  - foundation-Model
language: ko
summary: "Qwen3.5 VLM + DiT flow-matching action decoder / embodiment-aware prompt / joint pretraining → generalist VLA (manipulation, navigation, human egocentric motion, trajectory prediction)"
math: true
comments: true
comment_id: "paper-qwen-vla"
permalink: /paper/success-rate/qwen-vla/
---

<aside class="series-preface" markdown="1">

- **Authors:** Qiuyue Wang\*, Mingsheng Li\*, Jian Guan\*, Jinhui Ye, Sicheng Xie, Yitao Liu, Junhao Chen, Zhixuan Liang, Jie Zhang, Xintong Hu, Xuhong Huang, Pei Lin, Junyang Lin, Dayiheng Liu, Shuai Bai†, Jingren Zhou, Jiazhao Zhang, Haoqi Yuan, Gengze Zhou, Hang Yin, Ye Wang, Yiyang Huang, Zixing Lei, Wujian Peng, Delin Chen, Yingming Zheng, Jingyang Fan, Xianwei Zhuang, Xin Zhou, Haoyang Li, Anzhe Chen, Tong Zhang, Xuejing Liu, Yuchong Sun, Ruizhe Chen, Zhaohai Li, Chenxu Lü, Zhibo Yang, Tao Yu, Xionghui Chen
- **Affiliations:** Qwen Team, \*Equal contribution, †Corresponding author
- **Links:** [arXiv](https://arxiv.org/abs/2605.30280){:target="_blank" rel="noopener noreferrer"}, [Blog](https://qwen.ai/blog?id=qwenvla){:target="_blank" rel="noopener noreferrer"}, [GitHub](https://github.com/QwenLM/Qwen-VLA){:target="_blank" rel="noopener noreferrer"}
- **arXiv Submitted:** 2026-05-28

</aside>

기존 embodied AI/VLA 모델들은 manipulation, navigation, dexterous control처럼 특정 task나 embodiment에 맞춰 따로 설계되는 경우가 많아서 task 간 transfer와 robot embodiment 간 generalization이 제한적이었다.

Qwen team은 겉으로는 서로 달라 보이는 manipulation action, navigation waypoint, human hand/wrist trajectory, trajectory prediction을 모두 **future action/trajectory prediction conditioned on visual observation + language instruction + embodiment description** 문제로 통합할 수 있다고 보았다. 이를 위해 Qwen3.5-4B vision-language backbone 위에 약 1.15B parameter 규모의 DiT-style flow-matching action expert를 붙이고, robot-specific textual prompt로 현재 robot platform, arm configuration, control frequency, action convention, prediction horizon을 알려준다. 학습은 **T2A(text-to-action) → CPT(continued pretraining) → SFT → RL** 순서로 진행되며, 먼저 text만으로 action prior를 만든 뒤 vision grounding, downstream specialization, closed-loop success optimization을 단계적으로 수행한다.

## **Introduction**

기존에는 task에 따라 모델이 분리되어 있었고, 이들은 output format과 더불어 observation format, control frequency, prediction horizon, action dimensionality, evaluation protocol 등에서 차이가 있었다. 하지만 논문의 저자들은 여기에 다음과 같은 common computational structure가 있다고 보았다:

> **An embodied agent must condition on visual observations, language instructions, and embodiment-specific constraints, then predict future actions or trajectories that are physically and semantically aligned with the task.**

그래서 Qwen-VLA는 모든 task를 다음과 같은 conditional prediction framework로 통합한다:

{::nomarkdown}
\[
    p_{\theta}(y_{t:t+H-1} | o_{t}, x, e, z)
\]
{:/nomarkdown}

$o_t$는 image/video/history observation, $x$는 language instruction, $e$는 embodiment description, $z$는 optional task identifier, $y_{t:t+H-1}$는 예측할 future action/trajectory sequence이다. $H$는 action chunk length로 manipulation benchmark에서는 $H = 16$을 사용했다. Manipulation action, navigation waypoint, human hand/body trajectory, driving-style trajectory를 모두 이 unified action-and-trajectory space 안에서 다룬다.

## **Architecture**

Qwen-VLA는 크게 두 부분으로 나뉘어져 있는데, 하나는 Qwen3.5 vision-language backbone이고 나머지 하나는 DiT-style flow-matching action expert이다.

![qwenvla_overview](/paper/success-rate/images/qwenvla_overview.png)
*Architecture Overview*
{: .figure-caption}

### **Qwen3.5 vision-language backbone**

Qwen3.5는 native multimodal backbone이고, visual token을 text token stream에 interleave해서 images/videos/language를 하나의 Transformer에서 처리한다. 또한 long multimodal sequence encoding을 위해 gated linear attention과 grouped-query softmax attention을 혼합한다. 이 backbone의 역할은 아래와 같다:

1. image/video observation encoding
2. instruction understanding
3. object grounding
4. spatial reasoning
5. multi-step instruction following
6. embodiment prompt understanding


### **DiT-style flow-matching action expert**

input은 VLM hidden states, noisy action chunk, timestep embedding이다. Positional encoding은 multi-section RoPE를 썼다 (aligned with the backbone).

Qwen-VLA에서는 VLM의 pretrained capability를 망가뜨리지 않으면서 action generation 능력을 붙이기 위해 backbone과 action expert를 decoupled 시켰다. VLM backbone은 perception/reasoning을 보존하고, action expert는 continuous action distribution의 multi-modality와 high-frequency dynamics를 담당한다.



## **Embodiment-aware prompt conditioning**

서로 다른 로봇은 action semantics가 다르다. 예를 들어:

| 로봇/데이터                         | 가능한 action convention                          |
| ------------------------------ | ---------------------------------------------- |
| WidowX                         | delta end-effector + gripper                   |
| Franka Panda                   | delta EEF 또는 absolute joint + gripper          |
| Mobile ALOHA                   | dual-arm EEF/joint + gripper                   |
| AgiBot / Galaxea / humanoid 계열 | absolute joint, dexterous hand 등               |
| Navigation agent               | $(\Delta x, \Delta y, \Delta \theta)$ waypoint |
| Human egocentric data          | wrist pose, hand articulation                  |

이것을 하나의 action head로 처리하려면, 모델이 “지금 출력해야 하는 action vector가 무엇을 의미하는지” 알아야 한다. Qwen-VLA는 별도 per-robot head를 붙이는 대신, textual prompt로 알려준다. 논문에서 제시한 prompt template는 다음과 같다:

```text
The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile base].
The control frequency is {FPS} Hz.
Please predict the next {chunk_size} control actions to execute the following task:
{ori_instruction}.
```

이 prompt에는 robot platform, arm configuration, control convention, control frequency, prediction horizon이 들어간다. 이 정보가 VLM token으로 처리되고, 그 hidden state가 DiT action expert의 condition이 된다.

Qwen-VLA는 tensor interface와 masking scheme은 통합하지만, 모든 embodiment들을 single physical action semantic space로 강제 변환하지는 않는다. 즉, 각 dataset의 native control convention을 유지하고, 그 convention은 prompt와 dataset-specific normalization으로 알려준다.

## **Unified action representation**

Qwen-VLA는 모든 action target들을 고정된 tensor 형태로 맞춘다:

{::nomarkdown}
\[
    \mathbf{Y} \in \mathbb{R}^{H \times K}
\]
{:/nomarkdown}

여기서 $H$는 prediction horizon, $K$는 전체 control mode를 담기 위한 fixed channel dimension(실제 task가 쓰는 channel 수는 $c \leq K$이고 사용하지 않는 channel은 zero padding 함)이다. Valid channel/time step은 mask $\mathbf{M} \in \\\{0, 1\\\}^{H \times K}$로 표시한다. 예를 들어 어떤 single-arm robot은 7차원 action만 쓸 수 있고, 다른 humanoid/bimanual robot은 훨씬 높은 차원의 action을 쓸 수 있다. Qwen-VLA는 짧은 action vector를 앞쪽 c개 channel에 넣고 나머지를 0으로 padding한다. Loss 계산 시에는 mask로 padding된 위치를 제외한다. 이와 같은 설계로 다음과 같은 장점을 얻을 수 있다:

1. per-embodiment output head가 필요 없다.
2. 한 DiT action expert가 여러 action dimensionality를 처리한다.
3. mask를 쓰기 때문에 padding이 gradient를 오염시키지 않는다.
4. control convention은 prompt와 normalization으로 구분할 수 있다.

다만 이 방식은 "모든 로봇의 action semantics가 하나로 align된다”는 의미는 아니다. 더 정확히는 action tensor format을 통일하고, semantic interpretation은 prompt에 맡기는 방식이다.

## **(Training) Objective function**

Qwen-VLA는 두 종류의 loss를 함께 사용한다. Flow-matching action loss와 Vision-language next-token loss이다.

### **Flow-matching action loss**

Clean target action을 $\mathbf{Y}\_{0}$, Gaussian noise를 $\mathbf{Y}\_{1}$이라고 하고 $\mathbf{Y}\_{\tau} = ( 1 - \tau) \mathbf{Y}\_{0} + \tau \mathbf{Y}\_1$을 만든다. 모델 $v\_{\theta}$는 다음 velocity $\mathbf{Y}\_{1} - \mathbf{Y}\_{0}$을 예측한다. 다만 action tensor에는 padding이 있기 때문에, 단순히 전체 MSE를 평균내면 channel 수가 많은 embodiment나 padding 구조가 gradient를 왜곡할 수 있다. 그래서 논문은 먼저 active channel별 MSE를 계산하고, 그다음 active channel들에 대해 uniform average를 구한다. 이렇게 하면 각 control dimension이 동등하게 gradient에 기여하고 padding은 제외된다.

### **Vision-language next-token loss**

Qwen-VLA는 action만 학습하지 않는다. Auxiliary VL data, fine-grained embodied action captions, autonomous driving VQA, general VL pretraining corpora에 대해 next-token prediction loss도 유지한다:

{::nomarkdown}
\[
    \mathcal{L}_{\text{vl}} = - \sum_{i} \log p_{\theta}(w_i | w_{< i}, o_{1:t})
\]
{:/nomarkdown}

이 loss의 목적은 heavy embodied co-training 중에도 VLM backbone의 language grounding, visual reasoning, perception capability가 망가지지 않게 하는 것이다.

Joint objective는 위 두 loss의 weighted sum이다. mini-batch 안에서는 manipulation, VLN trajectory, vision-language sample을 고정 sampling ratio로 섞어서 backbone과 action expert를 동시에 업데이트한다.

## **Training recipe: T2A → CPT → SFT → RL**

![qwenvla_training](/paper/success-rate/images/qwenvla_training.png)
*Training Overview*
{: .figure-caption}

논문은 VLM backbone과 DiT action decoder가 매우 비대칭적인 상태에서 학습을 시작한다고 지적한다. VLM backbone은 이미 강하게 pretrained되어 있지만, DiT action decoder는 random initialization부터 시작한다. 이 상태에서 바로 image-conditioned joint training을 하면 decoder는 action distribution, language conditioning, embodiment conditioning, flow dynamics, visual grounding을 한꺼번에 배워야 하고, 동시에 매 step마다 image encoding 비용도 든다. 또한 fresh decoder에서 나온 noisy gradient가 pretrained VLM representation을 흔들 수도 있다. 그래서 Qwen-VLA는 학습을 4 단계로 나눈다.

### **Stage I: T2A, text-to-action DiT pretraining**

T2A에서는 VLM을 freeze하고 DiT만 학습한다. 입력에는 text instruction과 embodiment prompt만 주고, image는 의도적으로 제거한다. 즉, 모델은 다음을 학습한다:

{::nomarkdown}
\[
    \text{language instruction} + \text{embodiment prompt} \rightarrow \text{action trajectory prior}
\]
{:/nomarkdown}

논문은 이를 structured decompression problem으로 설명한다. “pick up the red cup” 같은 language instruction은 몇 token에 불과하지만, 실제 action trajectory는 수백~수천 개의 high-dimensional joint/action value로 이루어진다. 논문은 이 compressed task description이랑 full action signal 사이에 vast dimensionality gap이 있음을 지적한다. T2A는 compact text에서 dense action trajectory를 복원하는 decompressor를 먼저 학습시키는 단계다. 이 단계의 효과는 다음과 같다:

1. DiT가 action distribution의 모양을 먼저 배운다.
2. language가 action space의 어느 region을 선택하는지 배운다.
3. embodiment prompt가 같은 task intent를 robot-specific motor program으로 바꾸는 법을 배운다.
4. visual grounding이 들어오기 전에 action prior가 생긴다.

### **Stage II: CPT, continued pretraining**

CPT에서는 VLM과 DiT를 모두 unfreeze하고, heterogeneous multimodal data로 joint training한다. 이 단계의 목적은 T2A에서 배운 text-conditioned action prior를 실제 visual observation에 grounding하는 것이다.

즉, T2A가 “명령만 보고 가능한 action prior”를 배웠다면, CPT는 “현재 장면에서 그 action을 어떻게 실행할지”를 배운다. CPT가 simulation과 real-robot trajectory를 모두 포함하기 때문에, 이후 SFT/RL이 simulation 또는 real domain으로 specialization될 수 있다.

### **Stage III: SFT**

SFT는 두 branch로 진행된다:

#### **1. Multi-task SFT**
VQA, spatial grounding, manipulation, navigation을 함께 fine-tuning한다. embodiment-balanced, task-balanced sampling을 사용한다.

#### **2. Real-robot SFT**
CPT checkpoint에서 시작해 in-house teleoperation data로 real robot deployment용 fine-tuning을 한다.

SFT objective는 vision-language next-token loss와 action flow-matching loss를 함께 사용하며, loss weight는 VL next-token 0.1, manipulation/navigation action 1.0으로 설정한다.


### **Stage IV: RL**

SFT는 demonstration likelihood를 최적화한다. 하지만 실제 로봇 제어에서 중요한 것은 “demonstration처럼 그럴듯한 action을 냈는가”가 아니라 "closed-loop로 task success를 달성했는가"이다. 그래서 Qwen-VLA는 SFT checkpoint에서 PPO + GAE로 RL fine-tuning을 수행한다. reward는 simulator에서 주는 sparse binary reward이며, task goal이 episode 끝에 달성되면 $R = 1$, 아니면 $R = 0$이다. 이 RL stage는 SimplerEnv라는 단일 simulation environment에서만 rollout을 수집해 수행되며, 최종 모델 Qwen-VLA-Instruct를 만든다.

PPO 학습을 위해서는 log probability가 필요하다. Autoregressive token policy는 softmax 확률이 있으므로 log-probability가 바로 나오지만, flow-matching action decoder는 deterministic ODE 기반 implicit density라서 log-probability가 직접 나오지 않는다. 논문은 Euler denoising step마다 controlled noise를 넣어 deterministic probability-flow ODE를 SDE처럼 만들고, 각 transition을 Gaussian으로 해석해 analytic log-probability를 계산한다. rollout 때 intermediate denoising state를 저장하고, PPO update 때 current parameter로 velocity field를 재평가해 Gaussian log-probability를 다시 계산한다.

## **Pretraining data**

| Data source                          |     비율 |
| ------------------------------------ | -----: |
| Robot manipulation trajectories      |  74.2% |
| Human egocentric trajectories        |   6.0% |
| Navigation trajectories              |   7.5% |
| Synthetic simulation trajectories    |   3.7% |
| General vision-language data         |   3.4% |
| Spatial grounding 2D                 |   2.5% |
| Autonomous driving VQA               |   2.4% |
| Fine-grained embodied action caption |   0.2% |
| Total                                | 100.0% |


로봇 manipulation trajectory가 대부분을 차지하지만, navigation, egocentric human data, driving VQA, general VL data도 섞는다. 논문의 의도는 low-level motor prior와 high-level semantic/spatial reasoning을 동시에 유지하는 것이다.

### **Robot manipulation trajectories**

Robot manipulation data는 전체 pretraining mixture의 74.2%로 핵심이다. 공개 데이터로는 RobotSet, Galaxea, AgiBot World, RoboCOIN, RoboMIND V1/V2, RDT-1B, DROID, BridgeData V2, RH20T, RT-1, BC-Z 등을 사용하며, tabletop, mobile manipulation, bimanual, dexterous hand, in-the-wild execution을 포함해 10,000시간 이상의 heterogeneous interaction data를 구성한다. 또한 in-house real-robot trajectory 1,000시간 이상을 추가하고, 자체 scalable simulation pipeline으로 8M개 이상의 synthetic simulation trajectory를 생성한다. Representative embodiment에는 WidowX, Google Robot, Franka Panda, ARX5, Fourier GR-1, Mobile ALOHA, AgiBot A2-D, Galaxea R1, AIRBOT MMK2, TienKung, Real Human 등이 포함된다.

### **Human egocentric data**

Human egocentric data는 robot teleoperation보다 훨씬 scalable한 real-world manipulation prior를 제공하기 위한 것이다. Ego4D/EPIC-KITCHENS 기반 VITRA, EgoDex, EgoVerse, Xperience 등을 사용한다. Action representation은 양손 wrist motion과 hand articulation으로 구성되며, 각 hand에 대해 wrist SE(3) relative motion을 translation + axis-angle로 표현하고, 45D hand joint pose는 PCA로 10D eigengrasp coefficient로 줄인다. 그래서 bimanual human data의 action dimension은 time step당 32D가 된다.

### **Synthetic simulation data**

논문은 synthetic data를 두 종류로 나눈다:

#### **1. Vision-language-action data**
image observation + language instruction → robot action

#### **2. Language-action data**
language instruction만 보고 action trajectory 예측

Vision-conditioned synthetic data는 IsaacLab과 cuRobo 기반 pipeline으로 생성한다([RoboInF](https://xlang.ai/blog/roboinf){:target="_blank" rel="noopener noreferrer"}). 20개 tabletop scene, 각 scene당 10개 object initial-pose configuration, 총 200개 base scene configuration을 만들고, 450개 manipulation task를 생성한다. 각 task당 300개 successful trajectory를 생성하며, lighting, camera pose, background, table texture, robot initial state, object pose, controller dynamics 등을 randomization한다.

![qwenvla_data_example](/paper/success-rate/images/qwenvla_data_example.png)
*Examples of data generated through RoboInF*
{: .figure-caption}

Language-action synthetic data는 T2A pretraining의 주 corpus로 쓰인다. 여섯 종류의 single-arm manipulation primitive, 즉 pick-and-place, linear pushing, linear pulling, rotation with repositioning, rotation toward viewpoint direction, positional swapping을 여러 robot configuration에서 생성한다. 이 부분은 image 없이 language-action mapping을 먼저 학습시키는 데 중요하다.


### **Navigation data**
Navigation data는 전체 7.5%이며, long-horizon trajectory와 rich visual information을 제공한다. 구성은 instruction following, object searching, target tracking이다. Navigation action은 mobile robot이 2D plane에서 움직인다고 보고 $(\Delta x, \Delta y, \Delta \theta)$ waypoint로 표현한다. 이 데이터가 Qwen-VLA를 단순 arm manipulation policy가 아니라 VLN까지 포함하는 generalist embodied model로 만든다.


### **Vision-language data**

Vision-language data는 catastrophic forgetting을 막고, object grounding/spatial reasoning/instruction following을 유지하는 목적이다.

기존 robot dataset에는 “pick up, rotate, and place the ceramic bowl” 같은 coarse label만 있는 경우가 많다. 하지만 같은 label이라도 어디를 잡고, 어느 방향으로 돌리고, 어디에 놓는지가 다를 수 있다. Qwen-VLA는 13개 dimension의 dense action description을 생성하고 human review까지 수행해 약 48,000개 fine-grained video-caption pair를 만든다. 예를 들어 coarse label을 “ceramic bowl을 집고, 시계방향으로 두 바퀴 돌리고, table center에 놓는다” 같은 step-by-step caption으로 바꾼다.

Autonomous driving VQA도 넣는다. 이 데이터는 temporal scene understanding, surround-view spatial reasoning, language-grounded localization, planning-aware reasoning을 강화하기 위한 보조 supervision이다.











## **Experiments**

논문은 Qwen-VLA를 크게 다섯 관점에서 평가한다:

1. simulation manipulation benchmark에서 specialist model들과 비교
2. ALOHA bimanual robot에서 real-world in-domain / OOD 성능 평가
3. vision-language navigation benchmark에서 VLN 성능 평가
4. static / dynamic manipulation OOD benchmark에서 generalization 평가
5. T2A, VL co-training, projection design, RL, state conditioning에 대한 ablation study

평가 모델은 크게 두 가지이다:

| Model | 의미 |
| --- | --- |
| **Qwen-VLA-Base** | T2A + CPT까지 거친 large-scale pretrained generalist VLA |
| **Qwen-VLA-Instruct** | Qwen-VLA-Base에 SFT + RL post-training을 추가한 최종 모델 |

핵심 질문은 단순히 “각 benchmark에서 성능이 높은가?”가 아니라, **하나의 generalist VLA가 서로 다른 task, environment, robot embodiment에서 specialist model 수준의 성능을 낼 수 있는가?**이다.

### **Simulation manipulation**

먼저 LIBERO, Simpler-WidowX, RoboCasa-GR1, RoboTwin 2.0에서 manipulation 성능을 평가한다. 이 benchmark들은 single-arm tabletop manipulation부터 dual-arm / humanoid kitchen task까지 포함한다. 모든 benchmark에서 action chunk length는 `H = 16`으로 설정한다.

| Method | Type | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| π0 | Specialist | 94.4 | - | - | 65.9 | 58.4 |
| StarVLA-OFT | Specialist | 96.6 | 48.8 | 64.6 | 50.4 | - |
| GR00T N1.6 | Specialist | 97.2 | 49.9 | 63.2 | 47.6 | - |
| π0.5 | Specialist | 97.6 | 37.0 | 46.9 | 82.7 | 76.8 |
| ABot-M0 | Specialist | 98.6 | 58.3 | - | 86.0 | 85.0 |
| Being-H0.5 | Specialist | 97.6 | 53.3 | - | - | - |
| **Qwen-VLA-Base** | Generalist | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 |
| **Qwen-VLA-Instruct** | Generalist | **97.9** | **56.7** | **73.7** | **86.1** | **87.2** |

Qwen-VLA-Instruct는 하나의 generalist model임에도 LIBERO에서 97.9%, Simpler-WidowX에서 73.7%, RoboTwin-Easy/Hard에서 86.1/87.2%를 달성한다. 특히 Simpler-WidowX와 RoboTwin-Hard에서는 여러 specialist baseline보다 높은 성능을 보인다.

여기서 중요한 점은 Qwen-VLA가 benchmark별로 별도의 specialist model을 학습한 것이 아니라는 점이다. Qwen-VLA는 다양한 embodiment와 task를 함께 학습한 뒤, embodiment-aware prompt를 통해 각 platform에 맞는 action을 생성한다. 즉, 성능 표의 핵심은 단순한 SOTA 비교라기보다 **multi-embodiment co-training이 task-specific performance를 크게 희생하지 않는다는 것**이다.

또한 Qwen-VLA-Base에서 Qwen-VLA-Instruct로 넘어가면서 모든 benchmark에서 성능이 오른다. 상승폭은 LIBERO +7.1pp, RoboCasa-GR1 +16.3pp, Simpler-WidowX +9.4pp, RoboTwin-Easy +21.8pp, RoboTwin-Hard +20.8pp이다. 이는 large-scale pretraining이 transferable manipulation prior를 만들고, SFT/RL post-training이 이를 실제 benchmark execution에 맞게 정렬한다는 것을 보여준다.

### **Real-world ALOHA**

논문은 real-world bimanual manipulation 성능도 ALOHA platform에서 평가한다. ALOHA는 두 개의 6-DoF arm과 parallel-jaw gripper를 가진 bimanual robot이고, observation으로는 두 개의 wrist camera와 하나의 first-person-view RGB camera를 사용한다.

평가 task는 in-domain과 OOD로 나뉜다. In-domain task는 pick-and-place, table cleaning, bowl stacking, bowl/object placing, towel folding, fine-grained manipulation을 포함한다. OOD setting은 unseen color, unseen instance, unseen position, unseen background, unseen instruction generalization을 평가한다.

#### **In-domain real-world results**

| Model | Pick and Place | Table Cleaning | Bowl Stacking | Bowl Pick & Place | Towel Folding | Fine-grained | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GR00T N1.6 | 30.8 | 38.5 | 53.8 | 19.2 | 19.2 | 10.3 | 28.6 |
| π0.5 | 73.1 | 84.6 | 88.5 | 69.2 | 80.8 | 33.3 | 71.6 |
| Qwen-VLA-aloha w/o pretrain | 30.8 | 53.8 | 61.5 | 64.1 | 50.0 | 30.8 | 48.5 |
| **Qwen-VLA-aloha w/ pretrain** | **96.2** | **92.3** | **98.7** | **87.2** | 65.4 | **61.5** | **83.6** |

Qwen-VLA-aloha w/ pretrain은 평균 83.6%로 가장 높은 성능을 보인다. 같은 architecture를 scratch로 학습한 Qwen-VLA-aloha w/o pretrain은 48.5%에 그치므로, 성능 향상이 단순히 model architecture에서 나온 것이 아니라 **Qwen-VLA-Base의 large-scale embodied pretraining에서 온다**고 볼 수 있다.

특히 pick-and-place, table cleaning, bowl stacking, bowl pick & place, fine-grained manipulation에서 pretrained model의 이득이 크다. 이는 Qwen-VLA가 사전학습 과정에서 object grounding, spatial reasoning, manipulation primitive를 함께 학습했기 때문에 real-world fine-tuning에서도 더 빠르게 안정적인 policy로 수렴한 것으로 해석할 수 있다.

#### **OOD real-world results**

| Model | Color | Instance | Position | Background | Instruction | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GR00T N1.6 | 46.2 | 38.5 | 3.8 | 19.2 | 19.2 | 25.4 |
| π0.5 | 57.7 | 61.5 | 19.2 | 26.9 | 42.3 | 41.5 |
| Qwen-VLA-aloha w/o pretrain | 42.3 | 30.8 | 34.6 | 30.8 | 42.3 | 36.2 |
| **Qwen-VLA-aloha w/ pretrain** | **88.5** | **76.9** | **53.8** | **80.8** | **84.6** | **76.9** |

OOD setting에서도 Qwen-VLA-aloha w/ pretrain은 평균 76.9%를 달성한다. π0.5의 41.5%보다 +35.4pp 높고, scratch version의 36.2%보다 +40.7pp 높다.

특히 background generalization과 instruction generalization에서 각각 80.8%, 84.6%를 달성한 점이 중요하다. 이는 Qwen-VLA의 pretraining이 단순히 motor skill만 학습한 것이 아니라, visual-language grounding과 instruction following 능력을 action generation에 잘 연결했다는 근거로 볼 수 있다.

논문은 추가로 qualitative OOD rollout도 보여준다. Qwen-VLA-Base는 color-conditioned ball grasping, novel object grasping, compositional “clean up the table” task, unseen object approach, yellow background에서의 pen uncapping 등을 수행한다. 이 결과는 general VL data와 embodied action data를 함께 학습한 것이 novel object recognition과 manipulation transfer에 도움을 줄 수 있음을 보여준다.

### **Navigation**

Qwen-VLA는 manipulation만 다루는 모델이 아니라, navigation trajectory도 같은 action-and-trajectory prediction framework로 처리한다. 논문은 VLN-CE의 R2R, RxR Val-Unseen split에서 Qwen-VLA를 평가한다.

| Method | R2R NE ↓ | R2R OS ↑ | R2R SR ↑ | R2R SPL ↑ | RxR NE ↓ | RxR SR ↑ | RxR SPL ↑ | RxR nDTW ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NaVid | 5.7 | 49.2 | 41.9 | 36.5 | 5.7 | 45.7 | 38.2 | - |
| Uni-NaVid | 5.6 | 53.3 | 47.0 | 42.7 | 6.2 | 48.7 | 40.9 | - |
| NaVILA | 5.2 | 62.5 | 54.0 | 49.0 | 6.8 | 49.3 | 44.0 | 58.8 |
| StreamVLN | 5.0 | 64.2 | 56.9 | **51.9** | 6.2 | 52.9 | 46.0 | **61.9** |
| Qwen-VLA-Base | 5.2 | 61.7 | 53.8 | 49.4 | 6.4 | 55.1 | 45.8 | 56.2 |
| **Qwen-VLA-Instruct** | 5.1 | **69.0** | **57.5** | 51.2 | **5.8** | **59.6** | **47.8** | 57.1 |

Qwen-VLA-Instruct는 R2R Val-Unseen에서 OS(Oracle Success rate) 69.0, SR 57.5를 기록하고, RxR Val-Unseen에서는 SR 59.6, SPL(Success weighted by Path Length) 47.8을 기록한다. 이 결과는 Qwen-VLA가 robot arm manipulation뿐 아니라 mobile navigation trajectory prediction도 하나의 framework 안에서 처리할 수 있음을 보여준다.

다만 navigation에서 모든 metric이 압도적으로 최고인 것은 아니다. 예를 들어 R2R SPL은 StreamVLN이 51.9로 Qwen-VLA-Instruct의 51.2보다 조금 높고, RxR nDTW도 StreamVLN이 더 높다. 따라서 navigation 결과는 “Qwen-VLA가 VLN specialist를 완전히 대체한다”기보다는, **manipulation 중심의 generalist VLA가 VLN에서도 competitive한 성능을 낸다**는 쪽으로 해석하는 것이 더 적절하다.

### **Static OOD manipulation: SimplerEnv-OOD**

논문은 fine-tuning distribution 바깥의 manipulation generalization을 보기 위해 SimplerEnv-OOD를 구성한다. Fine-tuning은 Bridge training split의 simple pick-and-place data만 사용하고, evaluation에서는 unseen spatial relation과 unseen manipulation primitive를 요구한다.

| Method | MoveAway | MoveRight | PlaceNear | PlaceRight | PutFront | StackYellow | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| π0.5 | 26.1 | 0.0 | 0.0 | 32.1 | 13.0 | 4.2 | 12.6 |
| Qwen-VLA-Base | 31.3 | 31.6 | 16.7 | 47.1 | 6.3 | 18.8 | 25.3 |
| **Qwen-VLA-Instruct** | **43.8** | **33.3** | **39.6** | **47.9** | 4.2 | **22.9** | **32.0** |

Qwen-VLA-Instruct는 평균 32.0%로 π0.5의 12.6%보다 높다. 특히 MoveRight와 PlaceNear에서 π0.5는 0.0%이지만, Qwen-VLA-Instruct는 각각 33.3%, 39.6%를 달성한다. 이는 Qwen-VLA가 단순히 seen task를 imitation하는 것이 아니라, language instruction에 포함된 spatial relation을 어느 정도 action으로 옮길 수 있음을 보여준다.

StackYellow에서는 training data에는 green block을 yellow block 위에 쌓는 경우만 있지만, evaluation에서는 yellow block을 green block 위에 쌓아야 한다. Qwen-VLA-Instruct는 이 reversed color-order setting에서 22.9%를 달성해 π0.5의 4.2%보다 높다. 이는 object-color binding과 spatial goal interpretation 측면에서 Qwen-VLA의 pretraining이 도움을 준다는 신호다.

다만 PutFront에서는 Qwen-VLA-Instruct가 4.2%로 낮고, Qwen-VLA-Base의 6.3%보다도 낮다. 따라서 OOD generalization이 모든 spatial primitive에 대해 균일하게 좋아진 것은 아니며, direction-specific relation이나 viewpoint-dependent instruction에서는 여전히 취약할 수 있다.

### **Dynamic manipulation: DOMINO**

DOMINO는 moving object dynamics가 있는 dynamic manipulation benchmark이다. Qwen-VLA는 dynamic manipulation data에 fine-tuning하지 않고 zero-shot으로 평가된다.

| Method | Setting | SR ↑ | MS ↑ |
| --- | --- | ---: | ---: |
| π0.5 | Fine-tuned on dynamic data | 9.6 | 26.2 |
| StarVLA-OFT | Fine-tuned on dynamic data | 10.9 | 30.5 |
| PUMA | Fine-tuned on dynamic data | 17.2 | 35.0 |
| LingBot-VA | Zero-shot | 24.1 | 36.1 |
| Qwen-VLA-Base | Zero-shot | 21.1 | 37.4 |
| **Qwen-VLA-Instruct** | **Zero-shot** | **26.6** | **39.5** |

Qwen-VLA-Instruct는 dynamic manipulation data 없이 zero-shot으로 SR 26.6%, MS(Manipulation Score) 39.5를 달성한다. 이는 DOMINO-specific fine-tuning을 한 PUMA보다도 SR 기준 +9.4pp, MS 기준 +4.5 높다.

논문은 이 결과를 두 가지로 해석한다. 첫째, flow-matching action decoder가 coherent action chunk를 생성하기 때문에 dynamic object를 다룰 때 hesitation이 줄어든다. 둘째, manipulation, navigation, trajectory prediction, vision-language data를 함께 학습하면서 visual grounding과 spatial-to-kinematic prior가 더 넓게 형성된다. 즉, DOMINO 결과는 Qwen-VLA의 unified action-and-trajectory pretraining이 static manipulation을 넘어 dynamic setting에도 일부 transfer될 수 있음을 보여준다.

### **Ablation studies**

#### **T2A pretraining**

T2A ablation은 이 논문의 training recipe가 단순한 engineering detail이 아니라 성능에 직접적인 영향을 준다는 것을 보여준다.

핵심 결과는 다음과 같다:

| Ablation point | 결과 |
| --- | --- |
| Data composition | real-only 51.04%, synthetic-only 64.06%, 20% synthetic + 80% real 71.09% |
| Prediction mode | full-sequence prediction이 chunk prediction보다 좋음 |
| Image during T2A | image를 넣으면 오히려 성능이 하락 |
| Timestep distribution | T2A는 Sigmoid-Normal, CPT/SFT는 Beta가 가장 좋음 |
| T2A duration | 2k steps가 best, 40k steps는 overfitting으로 하락 |

가장 중요한 메시지는 **T2A는 visual grounding 단계가 아니라 language-action prior를 만드는 단계**라는 점이다. 그래서 image를 넣는 것보다 image를 제거하고 text + embodiment prompt만으로 action trajectory를 복원하게 하는 것이 더 좋다. 또한 chunk 단위보다 full trajectory 단위로 학습하는 것이 long-horizon temporal structure를 더 잘 학습하게 한다.

#### **Vision-language co-training**

VL data를 action learning 중에 계속 섞는 것이 interference를 만들 수도 있지만, 논문 결과에서는 오히려 복잡한 benchmark에서 도움이 된다.

| Benchmark | VLA-only | VL+VLA |
| --- | ---: | ---: |
| LIBERO | 거의 동일 | 거의 동일 |
| Simpler-WidowX | 거의 동일 | 거의 동일 |
| RoboCasa-GR1 | 51.1 | 56.0 |
| RoboTwin-2.0 | 81.8 | 86.4 |

LIBERO나 Simpler-WidowX처럼 상대적으로 단순한 benchmark에서는 VLA-only와 VL+VLA의 차이가 작다. 하지만 fine-grained object recognition과 compositional instruction parsing이 더 중요한 RoboCasa-GR1, RoboTwin-2.0에서는 VL+VLA가 각각 +4.9pp, +4.6pp 개선된다. 이는 VLM backbone의 semantic grounding을 유지하는 것이 action prediction에도 도움이 된다는 것을 보여준다.

#### **Projection design for heterogeneous embodiments**

서로 다른 embodiment는 action dimension과 semantics가 다르기 때문에, 이를 DiT latent space에 어떻게 넣을지가 중요하다. 논문은 Multi-MLP, Concatenation, Zero-Padding을 비교한다.

| Training setup | Bridge | Robocasa |
| --- | ---: | ---: |
| Single-embodiment training | 62.8 | 53.4 |
| Multi-MLP | 63.3 | 52.1 |
| Concatenation | 63.0 | 52.8 |
| Zero-Padding | 63.0 | 53.2 |

결과적으로 세 projection design의 성능 차이는 작다. Bridge와 Robocasa 모두 single-embodiment training과 multi-embodiment co-training의 차이가 크지 않고, projection 방식 간 차이도 1.2%p 이내이다. 따라서 논문은 parameter 수가 가장 적은 Zero-Padding을 default로 사용한다.

이 결과는 Qwen-VLA의 핵심이 projection trick 자체라기보다, **embodiment-aware prompt + shared action expert + heterogeneous pretraining**의 조합에 있음을 시사한다.

#### **Effect of RL post-training**

RL stage는 SimplerEnv에서만 rollout을 수집해 수행된다. 따라서 중요한 질문은 RL이 SimplerEnv 성능만 올리고 다른 benchmark를 망가뜨리는지, 아니면 다른 benchmark에도 유지 또는 transfer되는지이다.

| Stage | Simpler | RoboCasa | RoboTwin-E | RoboTwin-H | LIBERO | Simpler-OOD | DOMINO SR | DOMINO MS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPT | 64.3 | 40.4 | 64.3 | 66.4 | 90.8 | 25.3 | 21.1 | 37.4 |
| + SFT | 70.8 | 56.0 | 86.3 | 87.1 | 97.8 | 31.6 | 25.7 | 39.1 |
| + RL | 73.7 | 56.7 | 86.1 | 87.2 | 97.9 | 32.0 | 26.6 | 39.5 |

SFT는 거의 모든 benchmark에서 큰 폭의 성능 향상을 만든다. RL은 그 위에 SimplerEnv에서 +2.9pp를 추가로 올린다. 흥미로운 점은 RL rollout이 SimplerEnv에서만 수집되었는데도, RoboCasa, RoboTwin-Hard, LIBERO, Simpler-OOD, DOMINO 성능이 유지되거나 약간 개선된다는 것이다.

이는 RL이 catastrophic forgetting을 크게 만들지 않고, closed-loop execution을 조금 더 decisive하게 만드는 방향으로 작동했음을 시사한다. 다만 상승폭은 SFT에 비해 작기 때문에, 이 논문에서 가장 큰 성능 개선은 RL보다는 large-scale pretraining과 SFT에서 나온다고 보는 것이 적절하다.

#### **State conditioning**

마지막으로 논문은 proprioceptive state, 즉 joint angle 같은 explicit robot state를 넣는 것이 도움이 되는지도 비교한다.

| Conditioning | RoboTwin-Easy | RoboTwin-Hard |
| --- | ---: | ---: |
| No State | 88.7 | 87.4 |
| State in VLM Prompt | 89.3 | 88.7 |
| State in DiT | 89.4 | 88.3 |

State conditioning의 이득은 매우 작다. Easy에서는 최대 +0.7pp, Hard에서는 최대 +1.3pp 정도이다. 논문은 multi-view visual observation이 이미 robot configuration을 충분히 보여주고, action decoder가 relative action displacement를 예측하기 때문에 explicit state의 필요성이 작다고 해석한다.

다만 이것을 “proprioception은 필요 없다”로 일반화해서는 안 된다. 이 결과는 multi-view camera가 있고, end-effector가 보이며, benchmark task가 상대적으로 vision으로 현재 robot configuration을 추정하기 쉬운 setting에서 나온 것이다. Occlusion이 많거나 force/contact-rich manipulation, precise insertion, tactile feedback이 중요한 task에서는 proprioception이나 tactile/force sensing이 더 중요해질 수 있다.

### **Experiment summary**

실험 결과를 종합하면 Qwen-VLA의 핵심 주장은 어느 정도 뒷받침된다. Qwen-VLA-Instruct는 하나의 generalist model로 simulation manipulation, real-world ALOHA, VLN, static OOD manipulation, dynamic manipulation에서 모두 경쟁력 있는 성능을 보인다. 특히 real-world ALOHA OOD와 DOMINO zero-shot 결과는 large-scale heterogeneous pretraining이 단순 in-domain imitation을 넘어 visual / spatial / instruction generalization에 도움을 줄 수 있음을 보여준다.

하지만 Qwen-VLA가 모든 task에서 압도적인 것은 아니고, navigation 일부 metric이나 SimplerEnv-OOD의 PutFront처럼 약한 부분도 있다. 따라서 이 논문의 가장 중요한 contribution은 “모든 benchmark를 완전히 정복했다”가 아니라, **manipulation, navigation, egocentric motion, trajectory prediction을 하나의 VLA training framework로 묶었을 때 generalist policy가 specialist 수준에 가까운 성능과 의미 있는 OOD generalization을 보일 수 있다**는 점이다.






## **Limitations**

### **1. Embodied action data는 여전히 vision-language pretraining data보다 훨씬 작고 덜 다양하다**
따라서 long-tail object, long-tail environment, unseen embodiment, contact-rich interaction에 대한 robustness는 제한될 수 있다.

### **2. Vision-language understanding, navigation, action generation을 joint training하면 objective 간 trade-off가 생긴다**
action-oriented training이 policy learning에는 좋지만, pure VL 또는 navigation evaluation 일부를 regression시킬 수 있어 objective balancing, data curriculum, modular specialization이 필요하다.

### **3. 현재 evaluation은 여전히 benchmark 중심이고 short-horizon 성격이 강하다**
long-duration real-world deployment와 failure recovery는 열린 문제로 남아 있다.

### **4. Qwen-VLA는 “unified model”이라고 하지만 action semantics 자체를 완전히 공통 물리 공간으로 정렬한 것은 아니다**
실제로는 각 dataset의 native action convention을 유지하고, prompt와 padding/masking으로 구분한다. 따라서 새로운 robot embodiment로 확장할 때 prompt만 바꾸면 완전히 해결된다고 보기보다는, 해당 embodiment의 normalization, action layout, data coverage가 중요할 가능성이 높다.

### **5. State conditioning의 이득이 작다고 해서 proprioception이 항상 불필요하다는 뜻은 아니다**
이 결과는 multi-view camera에서 end-effector가 보이고, 상대 action prediction을 쓰는 benchmark 조건에서 나온 것이다. occlusion이 많거나 force/contact-rich task, precise insertion, tactile feedback이 중요한 task에서는 proprioception/tactile/force가 더 중요해질 수 있다.







{% include comments.html %}
