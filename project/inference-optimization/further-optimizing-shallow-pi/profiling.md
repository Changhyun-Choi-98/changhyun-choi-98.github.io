---
layout: default
title: "Shallow-π Profiling"
nav_exclude: true
section: project
subcategory: further-optimizing-shallow-pi
date: 2026-05-22
tags:
  - Korean
  - Python
  - Writing
language: ko
summary: "기존의 Shallow-π의 inference를 profiling해서 주요 병목지점 찾기"
math: true
comments: true
comment_id: "project-further-optimizing-shallow-pi-profiling"
permalink: /project/inference-optimization/further-optimizing-shallow-pi/profiling/
---

# **Shallow-π Profiling**

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

[Shallow-$\pi$가 잘 구현되었다는 것은 확인했으니](/project/inference-optimization/further-optimizing-shallow-pi/shallow-pi-implementation/), 다음으로 이 모델의 inference 과정에서 주요 bottleneck이 어디인지를 알아보기 위해 profiling을 수행한다. 이 서버는 여러 사용자가 함께 쓰는 공용 GPU 서버이므로, 전체 시스템을 독점한 dedicated benchmark 환경은 아니다. 따라서 이번 profiling에서는 실행 GPU를 1개의 L40S로 고정하고, 해당 GPU는 실험 중 단독으로 사용하기로 합의했다. 다만 CPU, memory, storage I/O, OS background load는 다른 사용자의 작업 영향을 받을 수 있으므로, 이후 latency 수치는 절대적인 서버 최대 성능이라기보다 shared-server 환경에서의 병목 분석용 측정값으로 해석한다.

## 1. **baseline latency script**

Profiler를 켜면 overhead가 추가적으로 생기므로, 먼저 profiler 없는 순수 latency, 즉 model-only latency baseline를 알아야 나중에 profiler 결과를 해석하기에 유리하다. 따라서 아래와 같은 파이썬 코드를 만들었다:

<details markdown="1">
<summary><code>profile_shallow_pi_latency.py</code></summary>

```python
#!/usr/bin/env python3

import argparse
import json
import pathlib
import statistics
import time
from typing import Any

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import policy_config
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="pi0_libero_l06")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--mode", type=str, choices=["model", "policy"], default="model")
    parser.add_argument("--fixed-noise", action="store_true")

    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out-json", type=str, default="profiles/latency/shallow_pi_latency.json")

    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    n = len(values)

    def percentile(p: float) -> float:
        idx = min(n - 1, int(round((p / 100.0) * (n - 1))))
        return values[idx]

    return {
        "count": n,
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(90),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def make_policy(args: argparse.Namespace):
    train_config = _config.get_config(args.config)

    # create_trained_policy() detects PyTorch checkpoint by model.safetensors.
    # sample_kwargs is passed to model.sample_actions().
    policy = policy_config.create_trained_policy(
        train_config,
        args.ckpt,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_steps},
    )

    return policy


def make_example() -> dict[str, Any]:
    # Random LIBERO-style dummy observation.
    # This checks model inference path, not task success.
    return libero_policy.make_libero_example()


def prepare_model_only_observation(policy, example: dict[str, Any], device: str):
    """
    Reproduce the PyTorch branch of Policy.infer(), but only once.

    Purpose:
      - input transform once
      - CPU -> GPU copy once
      - Observation object once
      - repeated benchmark measures only sample_actions()
    """
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)

    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
        inputs,
    )

    observation = _model.Observation.from_dict(inputs)
    return observation, inputs


def make_fixed_noise(policy, observation, device: str, seed: int):
    bsize = observation.state.shape[0]
    action_horizon = policy._model.config.action_horizon
    action_dim = policy._model.config.action_dim

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    return torch.randn(
        (bsize, action_horizon, action_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


@torch.inference_mode()
def run_model_only(policy, device: str, observation, noise, num_steps: int):
    if noise is None:
        return policy._sample_actions(device, observation, num_steps=num_steps)

    return policy._sample_actions(device, observation, noise=noise, num_steps=num_steps)


@torch.inference_mode()
def run_policy(policy, example: dict[str, Any]):
    return policy.infer(example)


def measure_cuda_event_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()

    _ = fn()

    end.record()
    end.synchronize()

    return float(start.elapsed_time(end))


def measure_sync_wall_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    _ = fn()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0


def main() -> None:
    args = parse_args()

    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This profiling script expects a CUDA GPU.")

    print("[INFO] config:", args.config)
    print("[INFO] ckpt:", args.ckpt)
    print("[INFO] device:", args.device)
    print("[INFO] num_steps:", args.num_steps)
    print("[INFO] mode:", args.mode)
    print("[INFO] fixed_noise:", args.fixed_noise)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)
    print("[INFO] visible devices:", torch.cuda.device_count())
    print("[INFO] device name:", torch.cuda.get_device_name(0))

    policy = make_policy(args)
    example = make_example()

    if args.mode == "model":
        observation, _ = prepare_model_only_observation(policy, example, args.device)
        noise = make_fixed_noise(policy, observation, args.device, args.seed) if args.fixed_noise else None

        def fn():
            return run_model_only(policy, args.device, observation, noise, args.num_steps)

    else:
        def fn():
            return run_policy(policy, example)

    print(f"[INFO] warmup start: {args.warmup}")
    for _ in range(args.warmup):
        _ = fn()
    torch.cuda.synchronize()
    print("[INFO] warmup done")

    # One output shape sanity check.
    out = fn()
    torch.cuda.synchronize()

    if isinstance(out, dict):
        print("[INFO] output keys:", list(out.keys()))
        if "actions" in out:
            print("[INFO] output actions shape:", np.asarray(out["actions"]).shape)
    else:
        print("[INFO] output tensor shape:", tuple(out.shape))
        print("[INFO] output dtype:", out.dtype)

    cuda_event_ms = []
    sync_wall_ms = []

    print(f"[INFO] measurement start: {args.iters}")
    for i in range(args.iters):
        cuda_event_ms.append(measure_cuda_event_ms(fn))
        sync_wall_ms.append(measure_sync_wall_ms(fn))

    result = {
        "config": args.config,
        "ckpt": args.ckpt,
        "device": args.device,
        "num_steps": args.num_steps,
        "mode": args.mode,
        "fixed_noise": args.fixed_noise,
        "warmup": args.warmup,
        "iters": args.iters,
        "cuda_event": summarize(cuda_event_ms),
        "sync_wall": summarize(sync_wall_ms),
    }

    print(json.dumps(result, indent=2))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
```

</details>

위 코드는 [실제 LIBERO observation 대신 shape만 같은 랜덤한 observation과 "do something"이라는 prompt를 줘서](https://github.com/icsl-Jeon/openpi/blob/a5940ac510c7f5d94918f238cfc3722be1a2c5c8/src/openpi/policies/libero_policy.py#L10){:target="_blank" rel="noopener noreferrer"} inference path에만 집중한다. 핵심 작동 원리는 아래와 같다:

```text
1. argument parsing
2. policy / model checkpoint load
3. dummy LIBERO observation 생성
4. mode에 따라 측정 함수 fn() 구성

   mode=model:
     - input transform 1회
     - GPU tensor 변환 1회
     - Observation 생성 1회
     - 반복 측정에서는 sample_actions()만 실행
     - model-only latency baseline

   mode=policy:
     - 반복 측정마다 policy.infer(example) 전체 실행
     - input transform, H2D copy, sample_actions, D2H copy, output transform 포함

5. warmup 수행
   - CUDA context initialization 제거
   - torch.compile 첫 compile overhead 제거
   - allocator/autotune overhead 제거

6. output shape sanity check
7. CUDA event latency 측정
8. synchronized wall-clock latency 측정
9. mean / median / p90 / p95 / p99 저장
10. JSON으로 결과 저장
```
{: style="margin-left: 1rem;" }

argument 중에서 유의해야 하는 것들 중 하나로 `fixed-noise`가 있다. 이 flag가 있으면 denoising을 시작하는 noise를 매번 새로 만들지 않고, 같은 noise를 반복해서 사용한다. 만약 매 iteration마다 noise를 새로 만들면 latency 안에 random number generation 비용이 포함된다. 첫 profiling에서는 denoising compute 자체를 보는 것이 목적이므로 `fixed-noise`를 주는 것을 고려해볼 수 있다. 이 flag가 없는 것이 실제 inference에 더 가깝긴 하다. 두 상황의 결과 차이가 작으면 무시해도 되고 (RNG 비용이 작음) 크면 이것도 profiling 대상으로 봐야 한다.

또한 `warmup`이라는 argument가 있다. 처음 inference를 하면 다음 overhead가 섞이게 된다:
```text
CUDA context initialization
model first run overhead
torch.compile compilation
Inductor / Triton autotuning
memory allocator warmup
kernel cache warmup
```
{: style="margin-left: 1rem;" }
위와 같은 overhead를 measurement에 포함시키지 않기 위해 `warmup` 횟수만큼 먼저 실행시킨다.

`prepare_model_only_observation` 함수는 `mode`가 `model` 일때 작동하는데, `Policy.infer()`는 원래 아래와 같은 매번 한다:
```text
raw observation
→ input transform
→ torch tensor 변환
→ GPU로 이동
→ Observation.from_dict
→ sample_actions()
→ CPU로 가져오기
→ output transform
```
{: style="margin-left: 1rem;" }
model-only profiling의 경우 `sample_actions()`만 measure한다.

`measure_cuda_event_ms()`를 이용해서 CUDA event 기반 latency 측정을 수행한다. GPU inference timing에서 가장 자주 쓰는 방식 중 하나이다. `measure_sync_wall_ms()`를 이용해서 CPU wall-clock 기준 latency를 측정한다. 두 latency의 의미는 아래와 같다:

| 측정 방식 | 의미 |
|---|---|
| CUDA event | CUDA stream 기준 elapsed time |
| synchronized wall | CPU 입장에서 `fn()` 전체가 끝날 때까지 걸린 시간 |

두 값을 비교해서 아래와 같은 insight를 얻을 수 있다:

```text
cuda_event ≈ sync_wall
  → GPU compute 중심, CPU overhead 작음

sync_wall >> cuda_event
  → CPU preprocessing, Python overhead, launch overhead, synchronization 문제 가능성

cuda_event가 심하게 흔들림
  → GPU clock, 다른 process, dynamic compilation, memory allocator 영향 가능성
```
{: style="margin-left: 1rem;" }


실행시킨 shell prompt(model-only / fixed-noise smoke test)와 결과는 아래와 같다:

<details markdown="1">
<summary>shell prompt & result json</summary>

```shell
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=0
export UV_LINK_MODE=copy

RUN_NAME=distill_l06_bf16_gb320_20260514_184612
STEP=30000
CKPT=./checkpoints/pi0_libero_l06/${RUN_NAME}/${STEP}

mkdir -p profiles/latency

uv run python scripts/profiling/profile_shallow_pi_latency.py \
  --config pi0_libero_l06 \
  --ckpt "${CKPT}" \
  --device cuda:0 \
  --num-steps 10 \
  --mode model \
  --fixed-noise \
  --warmup 5 \
  --iters 5 \
  --out-json profiles/latency/smoke_model_fixed_noise_numsteps10.json
```
{: style="margin-left: 1rem;" }

```json
{
  "config": "pi0_libero_l06",
  "ckpt": "./checkpoints/pi0_libero_l06/distill_l06_bf16_gb320_20260514_184612/30000",
  "device": "cuda:0",
  "num_steps": 10,
  "mode": "model",
  "fixed_noise": true,
  "warmup": 5,
  "iters": 5,
  "cuda_event": {
    "count": 5,
    "mean_ms": 20.894195556640625,
    "median_ms": 20.858688354492188,
    "p90_ms": 21.053152084350586,
    "p95_ms": 21.053152084350586,
    "p99_ms": 21.053152084350586,
    "min_ms": 20.847936630249023,
    "max_ms": 21.053152084350586
  },
  "sync_wall": {
    "count": 5,
    "mean_ms": 20.95017380779609,
    "median_ms": 20.850844972301275,
    "p90_ms": 21.413158043287694,
    "p95_ms": 21.413158043287694,
    "p99_ms": 21.413158043287694,
    "min_ms": 20.776993012987077,
    "max_ms": 21.413158043287694
  }
}
```
{: style="margin-left: 1rem;" }

</details>
`cuda_event`와 `sync_wall`이 거의 같기 때문에, 이 smoke test 구간에서는 Python overhead나 CPU-GPU synchronization overhead가 크게 보이지 않는다. 즉 현재 측정 대상인 model-only / fixed-noise / sample_actions()는 preliminary하게 GPU compute 중심으로 보인다. 하지만 sample 수가 5회로 아직 작고 지금은 smoke test, 즉 "실행 가능성 확인"을 위한 것이었고 이제 정식으로 fixed-noise baseline 100회를 실행시킨다:

<details markdown="1">
<summary>shell prompt & result json</summary>

```shell
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=0
export UV_LINK_MODE=copy

RUN_NAME=distill_l06_bf16_gb320_20260514_184612
STEP=30000
CKPT=./checkpoints/pi0_libero_l06/${RUN_NAME}/${STEP}

mkdir -p profiles/latency

uv run python scripts/profiling/profile_shallow_pi_latency.py \
  --config pi0_libero_l06 \
  --ckpt "${CKPT}" \
  --device cuda:0 \
  --num-steps 10 \
  --mode model \
  --fixed-noise \
  --warmup 30 \
  --iters 100 \
  --out-json profiles/latency/model_fixed_noise_numsteps10_100iters.json
```
{: style="margin-left: 1rem;" }

```json

```
{: style="margin-left: 1rem;" }

</details>


{% include comments.html %}
