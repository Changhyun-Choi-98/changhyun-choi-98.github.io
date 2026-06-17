---
layout: post
title: "4. Prefix Fixed-Cost Breakdown and Masked Image Skip"
nav_exclude: true
section: project
subcategory: shallow-pi
date: 2026-06-17
tags:
  - Korean
  - Python
  - Profiling
language: ko
summary: "prefix fixed cost를 image embedding과 prefill 단계로 분해한 뒤, LIBERO에서 mask 처리된 right-wrist image branch가 여전히 vision tower를 통과하는 낭비를 찾아 제거"
math: true
comments: true
comment_id: "project-shallow-pi-prefix-optim"
permalink: /project/inference-optimization/shallow-pi/prefix-optim/
---

<aside class="series-preface" markdown="1">

- **OS:** Ubuntu 22.04.5 LTS, Linux kernel 5.15.0
- **CPU:** 2 × AMD EPYC 9354 32-Core Processor
  - 64 physical cores / 128 threads total
- **Memory:** 503 GiB RAM
- **GPU:** 7 × NVIDIA L40S
  - ~48 GB VRAM per GPU
  - Used 1 out of 7 GPUs for inference & profiling
- **NVIDIA Driver / CUDA:** NVIDIA Driver 580.82.07 / CUDA 13.0
- **Storage:**
  - 2 TB NVMe SSD for the system drive
  - ~10.4 TB ext4 workspace/data storage
- **Usage context:** Shared multi-user research server
  - This machine is not a dedicated benchmark node.
  - Profiling was pinned to a single NVIDIA L40S, and that GPU was kept for this experiment during measurement.
  - CPU, system memory, storage I/O, and OS-level background load may still be affected by other users.

</aside>

이 서버는 여러 사용자가 함께 쓰는 공용 GPU 서버이므로, 전체 시스템을 독점한 dedicated benchmark 환경은 아니다. 따라서 이번 profiling에서는 실행 GPU를 1개의 L40S로 고정하고, 해당 GPU는 실험 중 단독으로 사용하기로 합의했다. 다만 CPU, memory, storage I/O, OS background load는 다른 사용자의 작업 영향을 받을 수 있으므로, 이후 latency 수치는 절대적인 서버 최대 성능이라기보다 shared-server 환경에서의 병목 분석용 측정값으로 해석한다.



[지난 게시물](/project/inference-optimization/shallow-pi/nsight-systems/)에서 Nsight Systems 분석으로 발견한 첫 번째 bottleneck은 prefix/prefill 자체가 아니라, denoise loop 내부의 CUDA tensor while-condition이 만든 per-step synchronization pattern이었다. `while time >= -dt / 2`를 `for range(num_steps)`로 바꾸자 `num_steps + 1`개의 graph launch / sync / D2H 패턴이 사라졌고, `T(N)`의 slope가 0.795 ms/step에서 0.628 ms/step으로 줄었다.

다만 after-patch scaling에서도 intercept는 약 13.94 ms로 여전히 크다. 따라서 다음 단계는 baseline latency 분석에서 가장 큰 항으로 드러난 prefix/prefill fixed cost를 분해하는 것이다.

[지난 게시물](/project/inference-optimization/shallow-pi/baseline-latency/)에서 구했던 `prefix-ish fixed cost`는 아래를 포함한다:

```text
sample_actions()
├─ observation preprocess
├─ embed_prefix()
│  ├─ image_0/base image embedding
│  ├─ image_1/left wrist image embedding
│  ├─ image_2/right wrist image embedding
│  ├─ language token embedding
│  └─ prefix token concat / mask construction
├─ prefix attention mask / position id build
├─ prefix prefill forward
│  └─ prefix KV cache 생성
└─ denoise loop
```

[지난](/project/inference-optimization/shallow-pi/baseline-latency/) [게시물들](/project/inference-optimization/shallow-pi/nsight-systems/)에서 구한 linear fit의 intercept로 구한 `14.14ms`는 주로

> `image embedding` + `language embedding` + `prefix prefill` + `KV cache 생성`

이다.

<details markdown="1">
<summary><code>profile_shallow_pi_prefix_breakdown.py</code></summary>
```python
#!/usr/bin/env python3

import argparse
import json
import math
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any, Callable

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.training import config as _config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pi0_libero_l06")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=str, default="profiles/latency/prefix_breakdown.json")
    return parser.parse_args()


def summarize(xs):
    xs = sorted(xs)
    n = len(xs)

    def pct(p):
        i = min(n - 1, int(round((p / 100.0) * (n - 1))))
        return xs[i]

    return {
        "count": n,
        "mean_ms": statistics.mean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": pct(90),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def sync_time_ms(fn: Callable[[], Any]):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def make_policy(args):
    train_config = _config.get_config(args.config)
    return policy_config.create_trained_policy(
        train_config,
        args.ckpt,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_steps},
    )


def prepare_observation(policy, example: dict[str, Any], device: str):
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
        inputs,
    )
    return _model.Observation.from_dict(inputs)


def make_fixed_noise(policy, observation, device: str, seed: int):
    bsize = observation.state.shape[0]
    horizon = policy._model.config.action_horizon
    action_dim = policy._model.config.action_dim
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(
        (bsize, horizon, action_dim),
        dtype=torch.float32,
        device=device,
        generator=gen,
    )


@torch.inference_mode()
def manual_prefix(policy, model, observation, times):
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation,
        train=False,
    )

    times["num_images"].append(float(len(images)))

    embs = []
    pad_masks = []
    att_masks = []

    # Image branches.
    for i, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):
        mask_true = int(img_mask.sum().detach().cpu())
        times[f"image_{i}_mask_true"].append(float(mask_true))

        img_emb, ms = sync_time_ms(lambda img=img: model.paligemma_with_expert.embed_image(img))
        times[f"image_{i}_embed"].append(ms)

        bsize, num_img_embs = img_emb.shape[:2]
        times[f"image_{i}_num_tokens"].append(float(num_img_embs))

        embs.append(img_emb)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
        att_masks += [0] * num_img_embs

    # Language branch.
    def lang_fn():
        lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        return lang_emb * math.sqrt(lang_emb_dim)

    lang_emb, ms = sync_time_ms(lang_fn)
    times["lang_embed"].append(ms)
    times["lang_num_tokens"].append(float(lang_emb.shape[1]))

    embs.append(lang_emb)
    pad_masks.append(lang_masks)
    att_masks += [0] * lang_emb.shape[1]

    # Concat prefix.
    def concat_fn():
        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
        bsize = prefix_pad_masks.shape[0]
        prefix_att_masks = prefix_att_masks[None, :].expand(bsize, len(att_masks))
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    (prefix_embs, prefix_pad_masks, prefix_att_masks), ms = sync_time_ms(concat_fn)
    times["prefix_concat"].append(ms)
    times["prefix_total_tokens"].append(float(prefix_embs.shape[1]))
    times["prefix_valid_tokens"].append(float(prefix_pad_masks.sum().detach().cpu()))

    return state, prefix_embs, prefix_pad_masks, prefix_att_masks


@torch.inference_mode()
def manual_prefix_prefill(model, prefix_embs, prefix_pad_masks, prefix_att_masks, times):
    def mask_fn():
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        return prefix_att_2d_masks_4d, prefix_position_ids

    (prefix_att_2d_masks_4d, prefix_position_ids), ms = sync_time_ms(mask_fn)
    times["prefix_mask_position_build"].append(ms)

    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    def prefill_fn():
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values

    past_key_values, ms = sync_time_ms(prefill_fn)
    times["prefix_prefill_forward"].append(ms)

    return past_key_values


@torch.inference_mode()
def manual_denoise_loop(model, device, state, prefix_pad_masks, past_key_values, noise, num_steps, times):
    bsize = state.shape[0]
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time_t = torch.tensor(1.0, dtype=torch.float32, device=device)

    step_times = []

    for _ in range(num_steps):
        def step_fn():
            nonlocal x_t, time_t
            expanded_time = time_t.expand(bsize)
            v_t = model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time_t += dt
            return x_t

        _, ms = sync_time_ms(step_fn)
        step_times.append(ms)

    times["denoise_loop_total"].append(sum(step_times))
    times["denoise_step_mean"].append(sum(step_times) / len(step_times))
    times["denoise_step_max"].append(max(step_times))

    return x_t


@torch.inference_mode()
def run_once(policy, observation, noise, device, num_steps, times):
    model = policy._model

    state, prefix_embs, prefix_pad_masks, prefix_att_masks = manual_prefix(
        policy, model, observation, times
    )

    past_key_values = manual_prefix_prefill(
        model, prefix_embs, prefix_pad_masks, prefix_att_masks, times
    )

    out = manual_denoise_loop(
        model, device, state, prefix_pad_masks, past_key_values, noise, num_steps, times
    )

    return out


def main():
    args = parse_args()

    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    policy = make_policy(args)
    example = libero_policy.make_libero_example()
    observation = prepare_observation(policy, example, args.device)
    noise = make_fixed_noise(policy, observation, args.device, args.seed)

    print("[INFO] device:", torch.cuda.get_device_name(0))
    print("[INFO] num_steps:", args.num_steps)

    # Warmup.
    warm_times = defaultdict(list)
    for _ in range(args.warmup):
        _ = run_once(policy, observation, noise, args.device, args.num_steps, warm_times)
    torch.cuda.synchronize()

    times = defaultdict(list)

    for _ in range(args.iters):
        _ = run_once(policy, observation, noise, args.device, args.num_steps, times)

    result = {
        "config": args.config,
        "ckpt": args.ckpt,
        "device": args.device,
        "num_steps": args.num_steps,
        "warmup": args.warmup,
        "iters": args.iters,
        "stages": {k: summarize(v) for k, v in sorted(times.items())},
    }

    print(json.dumps(result, indent=2))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
```
{: style="margin-left: 1rem;" }

</details>

Prefix breakdown 결과는 아래와 같다:

| Stage                        |                            Median |
| ---------------------------- | --------------------------------: |
| `image_0_embed`              |                          5.931 ms |
| `image_1_embed`              |                          5.639 ms |
| `image_2_embed`              |                      **5.600 ms** |
| `lang_embed`                 |                          0.058 ms |
| `prefix_mask_position_build` |                          0.137 ms |
| `prefix_prefill_forward`     |                      **8.091 ms** |

[Repository 코드](https://github.com/icsl-Jeon/openpi/blob/a5940ac510c7f5d94918f238cfc3722be1a2c5c8/src/openpi/models_pytorch/preprocessing_pytorch.py#L11){:target="_blank" rel="noopener noreferrer"} 기준으로 `IMAGE_KEYS` 순서는 다음과 같다:

```python
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)
```

즉 `image_2`는 `right_wrist_0_rgb`이다.

지금 실험을 진행하고 있는 LIBERO input transform은 이 image를 실제 image가 아니라 `np.zeros_like(base_image)`로 채우고, `pi0` 모델에서는 mask를 `False`로 둔다. 그런데 `PI0Pytorch.embed_prefix()`는 image mask를 보기 전에 모든 image에 대해 먼저 `self.paligemma_with_expert.embed_image(img)`를 호출한다. 이 부분을 제거하면 valid token은 유지하면서 prefix length를 줄일 수 있을 것이다. (이것을 발견하고 논문을 다시 확인했는데, 이 작업을 저자들도 했던 것 같은데 공식 repository에는 반영을 하지 않은 것 같다..) 해당 작업을 진행한 결과는 아래와 같다. 전체 policy latency 기준:

```text
23.74 ms → 18.49 ms
약 5.25 ms 감소
약 22.1% latency reduction
약 1.28× speedup
```

model-only 기준:

```text
Before image skip, after for-loop: 20.03 ms
After image skip:                  15.89 ms
약 4.14 ms 감소
약 20.7% latency reduction
약 1.26× speedup
```

지금까지 진행한 optimization 누적 효과는 다음과 같다:

| State                   | Sync wall median |           p95 | 해석       |
| -------------------- | --------------- | ------------ | -------- |
| Original policy      |        23.741 ms |     25.037 ms | 최초 기준    |
| After `while→for`    |        22.816 ms |     22.858 ms | tail 안정화 |
| After `image_2 skip` |    **18.487 ms** | **18.526 ms** | 큰 폭 개선   |


이 상태에서 scaling을 통해 linear regression을 수행한 결과는 아래와 같다:

| 상태                  |     Intercept |              Slope |
| ------------------- | ------------: | -----------------: |
| original            |    14.1388 ms |     0.7950 ms/step |
| after `while→for`   |    13.9442 ms |     0.6277 ms/step |
| after `image2 skip` | **9.7422 ms** | **0.6125 ms/step** |


```text
while→for:
  denoise step당 반복 overhead를 줄임

image2 skip:
  prefix fixed cost를 크게 줄임
```

다시 policy stage를 breakdown한 결과는 아래와 같다:

| Stage                   |    Median | Full 대비 |
| ----------------------- | --------: | ------: |
| `sample_actions`        | 15.806 ms |  86.61% |
| `observation_from_dict` |  1.941 ms |  10.64% |
| `tensorize_h2d`         |  0.214 ms |   1.17% |
| `input_transform`       |  0.120 ms |   0.66% |
| `to_cpu_numpy`          |  0.098 ms |   0.54% |
| `output_transform`      |  0.079 ms |   0.43% |


즉 아직도 main bottleneck은 `sample_actions()`이다. 하지만 `observation_from_dict`도 이제 충분히 커졌다. `Observation.from_dict()`는 uint8 torch image를 float32로 변환하고 `NHWC → NCHW` permute 및 `[-1, 1]` scaling을 수행한다. 이 작업이 모든 image key에 대해 수행된다.




{% comment %}{% include comments.html %}{% endcomment %}
