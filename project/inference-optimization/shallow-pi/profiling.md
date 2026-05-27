---
layout: default
title: "2. Shallow-π Baseline Latency Check"
nav_exclude: true
section: project
subcategory: shallow-pi
date: 2026-05-27
tags:
  - Korean
  - Python
language: ko
summary: "Profiling tool들을 사용하기 전에 profiler 없는 순수 latency를 먼저 확인"
math: true
comments: true
comment_id: "project-shallow-pi-baseline-latency"
permalink: /project/inference-optimization/shallow-pi/baseline-latency/
---

# **2. Shallow-π Baseline Latency Check**

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

[Shallow-$\pi$가 잘 구현되었다는 것은 확인했으니](/project/inference-optimization/shallow-pi/shallow-pi-implementation/), 다음으로 이 모델의 inference 과정에서 주요 bottleneck이 어디인지를 알아보기 위해 profiling을 수행해야 한다. 하지만 Profiler를 켜면 overhead가 추가적으로 생기므로, 먼저 profiler 없는 순수 latency, 즉 model-only latency baseline를 알아야 나중에 profiler 결과를 해석하기에 유리하다.

## **code**

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

## **model-only / fixed-noise smoke test**

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
`cuda_event`와 `sync_wall`이 거의 같기 때문에, 이 smoke test 구간에서는 Python overhead나 CPU-GPU synchronization overhead가 크게 보이지 않는다. 즉 현재 측정 대상인 model-only / fixed-noise / sample_actions()는 preliminary하게 GPU compute 중심으로 보인다. 하지만 sample 수가 5회로 아직 작고 지금은 smoke test, 즉 "실행 가능성 확인"을 위한 것이었고 다음에 정식으로 fixed-noise baseline 100회를 실행시킨다.

## **model-only / fixed-noise**

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
{
  "config": "pi0_libero_l06",
  "ckpt": "./checkpoints/pi0_libero_l06/distill_l06_bf16_gb320_20260514_184612/30000",
  "device": "cuda:0",
  "num_steps": 10,
  "mode": "model",
  "fixed_noise": true,
  "warmup": 30,
  "iters": 100,
  "cuda_event": {
    "count": 100,
    "mean_ms": 21.493796825408936,
    "median_ms": 21.271471977233887,
    "p90_ms": 22.424543380737305,
    "p95_ms": 22.73289680480957,
    "p99_ms": 23.950016021728516,
    "min_ms": 20.864831924438477,
    "max_ms": 24.368127822875977
  },
  "sync_wall": {
    "count": 100,
    "mean_ms": 21.45980772911571,
    "median_ms": 21.263867005473003,
    "p90_ms": 22.26925897412002,
    "p95_ms": 22.52989000407979,
    "p99_ms": 24.125280033331364,
    "min_ms": 20.887919003143907,
    "max_ms": 24.429485958535224
  }
}
```
{: style="margin-left: 1rem;" }

</details>

`pi0_libero_l06 / model-only / fixed-noise / num_steps=10`의 baseline은 아래와 같이 결론짓는다:

```text
model-only fixed-noise median latency ≈ 21.27 ms
p95 ≈ 22.73 ms
p99 ≈ 23.95 ms
```

Smoke test때와 마찬가지로, `cuda_event`와 `sync_wall`이 거의 동일하므로 현재 `sample_actions()` 측정 구간은 host-side overhead보다 GPU execution 중심이라는 것을 알 수 있다. 지금까지의 내용을 정리하면 아래와 같다:

<aside class="content-summary" markdown="1">

- Model: `pi0_libero_l06`
- Checkpoint: `distill_l06_bf16_gb320_20260514_184612` / step 30000
- GPU: NVIDIA L40S, physical GPU 6 exposed as cuda:0
- Mode: model-only
- Noise: fixed
- num_steps: 10
- Warmup: 30
- Iterations: 100
- Output shape: (1, 50, 32)

| Metric | CUDA event | Sync wall |
|---|---:|---:|
| mean | 21.494 ms | 21.460 ms |
| median | 21.271 ms | 21.264 ms |
| p95 | 22.733 ms | 22.530 ms |
| p99 | 23.950 ms | 24.125 ms |
| min | 20.865 ms | 20.888 ms |
| max | 24.368 ms | 24.429 ms |

- Interpretation
  - CUDA event latency and synchronized wall-clock latency are nearly identical, so the model-only fixed-noise path is dominated by GPU execution rather than host-side overhead.

</aside>

## **model-only / random-noise**

다음으로 model-only random-noise baseline을 측정한다. 즉 fixed noise를 제거해서 `sample_actions()` 내부의 random noise generation(`torch.randn`)까지 포함했을 때 latency가 얼마나 증가하는지 확인해야 한다.

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
  --warmup 30 \
  --iters 100 \
  --out-json profiles/latency/model_random_noise_numsteps10_100iters.json
```
{: style="margin-left: 1rem;" }

```json
{
  "config": "pi0_libero_l06",
  "ckpt": "./checkpoints/pi0_libero_l06/distill_l06_bf16_gb320_20260514_184612/30000",
  "device": "cuda:0",
  "num_steps": 10,
  "mode": "model",
  "fixed_noise": false,
  "warmup": 30,
  "iters": 100,
  "cuda_event": {
    "count": 100,
    "mean_ms": 21.373190784454344,
    "median_ms": 21.259360313415527,
    "p90_ms": 21.956031799316406,
    "p95_ms": 22.308992385864258,
    "p99_ms": 22.99951934814453,
    "min_ms": 20.86854362487793,
    "max_ms": 23.820735931396484
  },
  "sync_wall": {
    "count": 100,
    "mean_ms": 21.40404676378239,
    "median_ms": 21.179622504860163,
    "p90_ms": 22.070512000937015,
    "p95_ms": 22.383352043107152,
    "p99_ms": 23.382120009046048,
    "min_ms": 20.842222031205893,
    "max_ms": 24.246756045613438
  }
}
```
{: style="margin-left: 1rem;" }

</details>

fixed-noise와 random-noise 결과가 거의 같으므로 (오히려 random-noise run이 약간 더 빠르게 나왔는데, 이정도는 측정 noise / 공유 서버 상태 등의 범위로 보는 것이 합리적임) random noise generation overhead는 negligible하다는 것을 확인할 수 있다. 즉 `torch.randn` / RNG path는 지금 단계에서 optimization target이 아니다.

<aside class="content-summary" markdown="1">

- Model: `pi0_libero_l06`
- Checkpoint: `distill_l06_bf16_gb320_20260514_184612` / step 30000
- GPU: NVIDIA L40S, physical GPU 6 exposed as cuda:0
- num_steps: 10
- Warmup: 30
- Iterations: 100
- Output shape: (1, 50, 32)

| Mode | CUDA event median | Sync wall median | 해석 |
|---|---:|---:|---|
| fixed-noise | 21.271 ms | 21.264 ms | denoising compute 중심 |
| random-noise | 21.259 ms | 21.180 ms | RNG overhead negligible |

Conclusion:
The model-only inference latency is approximately 21.2–21.3 ms for num_steps=10. Random noise generation does not measurably contribute to latency.

</aside>

## **policy-level**

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
  --mode policy \
  --warmup 30 \
  --iters 100 \
  --out-json profiles/latency/policy_numsteps10_100iters.json
```
{: style="margin-left: 1rem;" }

```json
{
  "config": "pi0_libero_l06",
  "ckpt": "./checkpoints/pi0_libero_l06/distill_l06_bf16_gb320_20260514_184612/30000",
  "device": "cuda:0",
  "num_steps": 10,
  "mode": "policy",
  "fixed_noise": false,
  "warmup": 30,
  "iters": 100,
  "cuda_event": {
    "count": 100,
    "mean_ms": 23.97035972595215,
    "median_ms": 23.76371192932129,
    "p90_ms": 24.5696964263916,
    "p95_ms": 25.10211181640625,
    "p99_ms": 25.959232330322266,
    "min_ms": 23.543039321899414,
    "max_ms": 26.087392807006836
  },
  "sync_wall": {
    "count": 100,
    "mean_ms": 23.963603607262485,
    "median_ms": 23.740898992400616,
    "p90_ms": 24.71128397155553,
    "p95_ms": 25.037412997335196,
    "p99_ms": 26.103081996552646,
    "min_ms": 23.56435399269685,
    "max_ms": 26.144322007894516
  }
}
```
{: style="margin-left: 1rem;" }

</details>

| Metric            | Model-only random-noise | Policy-level |       증가량 |
| ----------------- | ----------------------: | -----------: | --------: |
| CUDA event mean   |               21.373 ms |    23.970 ms | +2.597 ms |
| CUDA event median |               21.259 ms |    23.764 ms | +2.504 ms |
| CUDA event p95    |               22.309 ms |    25.102 ms | +2.793 ms |
| CUDA event p99    |               23.000 ms |    25.959 ms | +2.960 ms |
| Sync wall mean    |               21.404 ms |    23.964 ms | +2.560 ms |
| Sync wall median  |               21.180 ms |    23.741 ms | +2.561 ms |
| Sync wall p95     |               22.383 ms |    25.037 ms | +2.654 ms |
| Sync wall p99     |               23.382 ms |    26.103 ms | +2.721 ms |


Policy-level overhead는 약 `+2.56ms`이다. 즉 현재 inference latency는 다음과 같이 분해된다:

```text
model-only random-noise median  ≈ 21.18 ms
policy-level median             ≈ 23.74 ms
extra policy wrapper overhead   ≈  2.56 ms
```

즉 전체 latency의 약 89%는 `sample_actions()` 내부 model inference이고, 나머지 11%는 `policy.infer()` wrapper 경로의 preprocessing / tensorization / H2D / D2H / output transform overhead이다.

<aside class="content-summary" markdown="1">

- Model: `pi0_libero_l06`
- Checkpoint: `distill_l06_bf16_gb320_20260514_184612` / step 30000
- GPU: NVIDIA L40S, physical GPU 6 exposed as cuda:0
- num_steps: 10
- warmup: 30
- iterations: 100

| Mode | Noise | Sync wall median | Sync wall p95 | Sync wall p99 |
|---|---|---:|---:|---:|
| model-only | fixed | 21.264 ms | 22.530 ms | 24.125 ms |
| model-only | random | 21.180 ms | 22.383 ms | 23.382 ms |
| policy-level | random | 23.741 ms | 25.037 ms | 26.103 ms |

Interpretation:
Random noise generation is negligible. The policy-level path adds approximately 2.56 ms over model-only inference, but about 89% of end-to-end policy latency still comes from the model sampling path.

</aside>

이번 policy-level 세팅에서도 `cuda_event`와 `sync_wall`이 거의 같았다. 하지만 이것을 곧바로 "policy 전체가 GPU compute-bound이다"라고 해석하면 안된다. 이유는 지금 timing 함수가 `policy.infer()` 전체를 감싼 상태에서 CUDA event를 찍기 때문이다. `policy.infer()` 내부는 다음과 같다:

```text
input transform
numpy → torch tensor 변환 및 device 이동
Observation.from_dict()
sample_actions()
torch tensor → CPU numpy 변환
output transform
```

CUDA event를 함수 바깥에 걸면, CPU preprocessing 때문에 GPU stream이 idle인 시간도 event interval 안에 들어갈 수 있다. 따라서 policy-level에서 event와 wall이 같다는 것은 end-to-end bracket이 일관되게 측정됐다는 뜻이지, preprocessing overhead가 없다는 뜻은 아니다. 따라서 다음으로 stage breakdown이 필요하다.

## **policy stage breakdown**

<details markdown="1">
<summary><code>profile_shallow_pi_policy_stages.py</code></summary>

```python
#!/usr/bin/env python3

import argparse
import json
import pathlib
import statistics
import time
from collections import defaultdict
from typing import Any, Callable

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
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--out-json",
        type=str,
        default="profiles/latency/policy_stage_breakdown.json",
    )

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

    policy = policy_config.create_trained_policy(
        train_config,
        args.ckpt,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_steps},
    )

    return policy


def make_example() -> dict[str, Any]:
    return libero_policy.make_libero_example()


def sync_wall_ms(fn: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = fn()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return out, (t1 - t0) * 1000.0


@torch.inference_mode()
def run_policy_full(policy, example: dict[str, Any]):
    return policy.infer(example)


@torch.inference_mode()
def run_policy_staged(policy, example: dict[str, Any], device: str):
    """
    This manually follows Policy.infer() for PyTorch models and times each stage.

    The per-stage timing uses synchronize-before/after wall timing.
    This is intentionally intrusive, but useful for attribution.
    Do not treat the sum of stages as production latency.
    """

    times: dict[str, float] = {}

    inputs, times["copy_obs"] = sync_wall_ms(
        lambda: jax.tree.map(lambda x: x, example)
    )

    inputs, times["input_transform"] = sync_wall_ms(
        lambda: policy._input_transform(inputs)
    )

    inputs_torch, times["tensorize_h2d"] = sync_wall_ms(
        lambda: jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
            inputs,
        )
    )

    observation, times["observation_from_dict"] = sync_wall_ms(
        lambda: _model.Observation.from_dict(inputs_torch)
    )

    sample_kwargs = dict(policy._sample_kwargs)

    actions, times["sample_actions"] = sync_wall_ms(
        lambda: policy._sample_actions(device, observation, **sample_kwargs)
    )

    outputs = {
        "state": inputs_torch["state"],
        "actions": actions,
    }

    outputs_np, times["to_cpu_numpy"] = sync_wall_ms(
        lambda: jax.tree.map(
            lambda x: np.asarray(x[0, ...].detach().cpu()),
            outputs,
        )
    )

    outputs_final, times["output_transform"] = sync_wall_ms(
        lambda: policy._output_transform(outputs_np)
    )

    times["stage_sum"] = sum(times.values())

    return outputs_final, times


def main() -> None:
    args = parse_args()

    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("[INFO] config:", args.config)
    print("[INFO] ckpt:", args.ckpt)
    print("[INFO] device:", args.device)
    print("[INFO] num_steps:", args.num_steps)
    print("[INFO] warmup:", args.warmup)
    print("[INFO] iters:", args.iters)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)
    print("[INFO] device name:", torch.cuda.get_device_name(0))

    policy = make_policy(args)
    example = make_example()

    print("[INFO] warmup start")
    for _ in range(args.warmup):
        _ = run_policy_full(policy, example)
    torch.cuda.synchronize()
    print("[INFO] warmup done")

    full_policy_total_ms: list[float] = []
    stage_values: dict[str, list[float]] = defaultdict(list)

    print("[INFO] measurement start")

    for _ in range(args.iters):
        _, full_ms = sync_wall_ms(lambda: run_policy_full(policy, example))
        full_policy_total_ms.append(full_ms)

        out, times = run_policy_staged(policy, example, args.device)
        for k, v in times.items():
            stage_values[k].append(v)

    result = {
        "config": args.config,
        "ckpt": args.ckpt,
        "device": args.device,
        "num_steps": args.num_steps,
        "warmup": args.warmup,
        "iters": args.iters,
        "full_policy_total": summarize(full_policy_total_ms),
        "stages": {
            k: summarize(v)
            for k, v in sorted(stage_values.items())
        },
    }

    # Add percentage of full policy total based on medians.
    full_median = result["full_policy_total"]["median_ms"]
    result["stage_median_percent_of_full"] = {
        k: 100.0 * summary["median_ms"] / full_median
        for k, summary in result["stages"].items()
    }

    print(json.dumps(result, indent=2))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
```

</details>

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

uv run python scripts/profiling/profile_shallow_pi_policy_stages.py \
--config pi0_libero_l06 \
--ckpt "${CKPT}" \
--device cuda:0 \
--num-steps 10 \
--warmup 30 \
--iters 100 \
--out-json profiles/latency/policy_stage_breakdown_numsteps10_100iters.json
```
{: style="margin-left: 1rem;" }

```json
{
  "config": "pi0_libero_l06",
  "ckpt": "./checkpoints/pi0_libero_l06/distill_l06_bf16_gb320_20260514_184612/30000",
  "device": "cuda:0",
  "num_steps": 10,
  "warmup": 30,
  "iters": 100,
  "full_policy_total": {
    "count": 100,
    "mean_ms": 23.704008038621396,
    "median_ms": 23.48822948988527,
    "p90_ms": 24.658604990690947,
    "p95_ms": 24.76810000371188,
    "p99_ms": 24.904964957386255,
    "min_ms": 23.32890999969095,
    "max_ms": 24.960848968476057
  },
  "stages": {
    "copy_obs": {
      "count": 100,
      "mean_ms": 0.01343892770819366,
      "median_ms": 0.013194512575864792,
      "p90_ms": 0.014361925423145294,
      "p95_ms": 0.014952966012060642,
      "p99_ms": 0.01714599784463644,
      "min_ms": 0.012468080967664719,
      "max_ms": 0.020900974050164223
    },
    "input_transform": {
      "count": 100,
      "mean_ms": 0.12454838841222227,
      "median_ms": 0.11845299741253257,
      "p90_ms": 0.15158706810325384,
      "p95_ms": 0.15817699022591114,
      "p99_ms": 0.1767550129443407,
      "min_ms": 0.11243904009461403,
      "max_ms": 0.2010620664805174
    },
    "observation_from_dict": {
      "count": 100,
      "mean_ms": 1.9876398053020239,
      "median_ms": 1.921984541695565,
      "p90_ms": 2.212640014477074,
      "p95_ms": 2.2544129751622677,
      "p99_ms": 2.4826559238135815,
      "min_ms": 1.878809998743236,
      "max_ms": 2.6681750314310193
    },
    "output_transform": {
      "count": 100,
      "mean_ms": 0.08812469197437167,
      "median_ms": 0.08372054435312748,
      "p90_ms": 0.11097593232989311,
      "p95_ms": 0.11613406240940094,
      "p99_ms": 0.12070103548467159,
      "min_ms": 0.07583305705338717,
      "max_ms": 0.1208609901368618
    },
    "sample_actions": {
      "count": 100,
      "mean_ms": 21.15997519227676,
      "median_ms": 21.04175853310153,
      "p90_ms": 21.74373308662325,
      "p95_ms": 21.813276107423007,
      "p99_ms": 21.94621495436877,
      "min_ms": 20.935378037393093,
      "max_ms": 21.98795904405415
    },
    "stage_sum": {
      "count": 100,
      "mean_ms": 23.696277211420238,
      "median_ms": 23.501925577875227,
      "p90_ms": 24.637093883939087,
      "p95_ms": 24.740207940340042,
      "p99_ms": 24.89900786895305,
      "min_ms": 23.33356684539467,
      "max_ms": 25.059939012862742
    },
    "tensorize_h2d": {
      "count": 100,
      "mean_ms": 0.22605346865020692,
      "median_ms": 0.22080144844949245,
      "p90_ms": 0.2539310371503234,
      "p95_ms": 0.25926902890205383,
      "p99_ms": 0.26374601293355227,
      "min_ms": 0.21319999359548092,
      "max_ms": 0.26438699569553137
    },
    "to_cpu_numpy": {
      "count": 100,
      "mean_ms": 0.09649673709645867,
      "median_ms": 0.09389093611389399,
      "p90_ms": 0.11240900494158268,
      "p95_ms": 0.11590402573347092,
      "p99_ms": 0.12263504322618246,
      "min_ms": 0.08556910324841738,
      "max_ms": 0.1281519653275609
    }
  },
  "stage_median_percent_of_full": {
    "copy_obs": 0.05617499855213327,
    "input_transform": 0.504307902234785,
    "observation_from_dict": 8.182756143979386,
    "output_transform": 0.3564361647146714,
    "sample_actions": 89.58426833390203,
    "stage_sum": 100.05831043159662,
    "tensorize_h2d": 0.9400514778884297,
    "to_cpu_numpy": 0.39973611529266695
  }
}
```
{: style="margin-left: 1rem;" }

</details>

현재 policy inference는 다음과 같이 분해된다:

```text
full policy median     ≈ 23.49 ms
sample_actions median  ≈ 21.04 ms
policy wrapper overhead ≈  2.45 ms
```

| Stage                   |        Median | Full 대비 비율 | 판단                      |
| ----------------------- | ------------: | ---------: | ----------------------- |
| `sample_actions`        | **21.042 ms** | **89.58%** | 압도적 main bottleneck     |
| `observation_from_dict` |  **1.922 ms** |  **8.18%** | 가장 큰 non-model overhead |
| `tensorize_h2d`         |      0.221 ms |      0.94% | 작음                      |
| `input_transform`       |      0.118 ms |      0.50% | 작음                      |
| `to_cpu_numpy`          |      0.094 ms |      0.40% | 작음                      |
| `output_transform`      |      0.084 ms |      0.36% | 작음                      |
| `copy_obs`              |      0.013 ms |      0.06% | 무시 가능                   |


결론은 다음과 같다:

```text
1. 전체 policy latency의 약 89.6%는 sample_actions() 내부
2. wrapper overhead 중 가장 큰 것은 observation_from_dict ≈ 1.92 ms
3. input_transform, tensorize_h2d, to_cpu_numpy, output_transform은 각각 0.1~0.22 ms 수준
4. 따라서 main optimization target은 여전히 sample_actions()
5. 다만 observation_from_dict는 나중에 별도 최적화 가치 있음
```

<aside class="content-summary" markdown="1">

- Model: `pi0_libero_l06`
- Checkpoint: `distill_l06_bf16_gb320_20260514_184612` / step 30000
- GPU: NVIDIA L40S, physical GPU 6 exposed as cuda:0
- num_steps: 10
- warmup: 30
- iterations: 100

| Mode | Noise | Median | p95 | p99 |
|---|---|---:|---:|---:|
| model-only | fixed | 21.264 ms | 22.530 ms | 24.125 ms |
| model-only | random | 21.180 ms | 22.383 ms | 23.382 ms |
| policy-level | random | 23.741 ms | 25.037 ms | 26.103 ms |
| policy-stage full total | random | 23.488 ms | 24.768 ms | 24.905 ms |
| policy-stage sample_actions | random | 21.042 ms | 21.813 ms | 21.946 ms |
| policy-stage observation_from_dict | - | 1.922 ms | 2.254 ms | 2.483 ms |

Interpretation:
Random noise generation is negligible. About 89–90% of local policy latency is spent inside sample_actions(). The largest non-model overhead is observation_from_dict, contributing about 1.92 ms.

</aside>


## **`num_steps` scaling**

`num_steps` scaling은 다음 수식을 fitting하기 위한 실험이다:

{::nomarkdown}
\[
T_{\text{num_steps}} \approx T_{\text{prefix}} + \text{num_steps} \times T_{\text{denoise_step}}
\]
{:/nomarkdown}

위 수식에서:

```text
T_prefix:
  image/language prefix embedding
  prefix prefill
  KV cache 생성
  one-time setup cost

T_denoise_step:
  embed_suffix
  action expert transformer forward
  attention / MLP / GEMM
  action_out_proj
  Euler update
  mask / position overhead
```

현재 `num_steps=10`에서 `sample_actions ≈ 21.04 ms`인데 아직 이 21ms가 다음 중 무엇인지를 모르는 상황이다:

```text
Case A:
  prefix가 크고 denoise step은 작다

Case B:
  prefix는 작고 denoise step 반복이 대부분이다

Case C:
  둘 다 비슷하게 크다
```

이걸 분리해야 PyTorch Profiler와 Nsight Systems trace를 제대로 해석할 수 있다. random noise overhead가 negligible인 것을 이미 확인했으므로, 구조 분해에는 fixed-noise가 더 깨끗하다. 따라서 이번에는 **model-only fixed-noise**로 돌린다

<details markdown="1">
<summary>shell prompts & result</summary>

```shell
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=0
export UV_LINK_MODE=copy

RUN_NAME=distill_l06_bf16_gb320_20260514_184612
STEP=30000
CKPT=./checkpoints/pi0_libero_l06/${RUN_NAME}/${STEP}

mkdir -p profiles/latency

for N in 1 2 4 6 8 10 12 16; do
  echo "==== num_steps=${N} ===="

  uv run python scripts/profiling/profile_shallow_pi_latency.py \
    --config pi0_libero_l06 \
    --ckpt "${CKPT}" \
    --device cuda:0 \
    --num-steps ${N} \
    --mode model \
    --fixed-noise \
    --warmup 30 \
    --iters 100 \
    --out-json profiles/latency/model_fixed_noise_numsteps${N}_100iters.json
done
```
{: style="margin-left: 1rem;" }

```shell
uv run python - <<'PY'
import json
import pathlib
import numpy as np

base = pathlib.Path("profiles/latency")
steps = [1, 2, 4, 6, 8, 10, 12, 16]

rows = []

for n in steps:
    path = base / f"model_fixed_noise_numsteps{n}_100iters.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append({
        "num_steps": n,
        "cuda_median": data["cuda_event"]["median_ms"],
        "cuda_p95": data["cuda_event"]["p95_ms"],
        "wall_median": data["sync_wall"]["median_ms"],
        "wall_p95": data["sync_wall"]["p95_ms"],
    })

print("num_steps,cuda_median,cuda_p95,wall_median,wall_p95")
for r in rows:
    print(f'{r["num_steps"]},{r["cuda_median"]:.4f},{r["cuda_p95"]:.4f},{r["wall_median"]:.4f},{r["wall_p95"]:.4f}')

x = np.array([r["num_steps"] for r in rows], dtype=np.float64)
y = np.array([r["cuda_median"] for r in rows], dtype=np.float64)

# Fit y = a + b*x
b, a = np.polyfit(x, y, deg=1)
y_hat = a + b * x

ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print()
print("Linear fit using CUDA event median:")
print(f"  T(num_steps) ≈ {a:.4f} ms + {b:.4f} ms * num_steps")
print(f"  intercept / prefix-ish cost: {a:.4f} ms")
print(f"  per denoise step cost:       {b:.4f} ms")
print(f"  R^2:                         {r2:.6f}")

print()
print("Residuals:")
for n, yi, yh in zip(x, y, y_hat):
    print(f"  N={int(n):2d}: observed={yi:.4f} ms, fitted={yh:.4f} ms, residual={yi-yh:+.4f} ms")
PY
```
{: style="margin-left: 1rem;" }

```text
num_steps,cuda_median,cuda_p95,wall_median,wall_p95
1,14.9003,15.5956,14.9625,15.7442
2,15.8370,17.0223,15.9261,16.8606
4,17.3294,18.5309,17.3632,18.5529
6,18.8133,19.8019,18.7842,20.1234
8,19.8144,20.6324,19.8650,20.5646
10,22.2703,24.2536,22.4992,24.3688
12,24.7246,26.7561,24.5631,27.1710
16,26.3230,27.6713,26.3405,28.0805

Linear fit using CUDA event median:
  T(num_steps) ≈ 14.1388 ms + 0.7950 ms * num_steps
  intercept / prefix-ish cost: 14.1388 ms
  per denoise step cost:       0.7950 ms
  R^2:                         0.984047

Residuals:
  N= 1: observed=14.9003 ms, fitted=14.9337 ms, residual=-0.0334 ms
  N= 2: observed=15.8370 ms, fitted=15.7287 ms, residual=+0.1083 ms
  N= 4: observed=17.3294 ms, fitted=17.3186 ms, residual=+0.0108 ms
  N= 6: observed=18.8133 ms, fitted=18.9085 ms, residual=-0.0952 ms
  N= 8: observed=19.8144 ms, fitted=20.4984 ms, residual=-0.6840 ms
  N=10: observed=22.2703 ms, fitted=22.0883 ms, residual=+0.1820 ms
  N=12: observed=24.7246 ms, fitted=23.6782 ms, residual=+1.0464 ms
  N=16: observed=26.3230 ms, fitted=26.8580 ms, residual=-0.5350 ms
```
{: style="margin-left: 1rem;" }

</details>

현재 `sample_actions()`의 bottleneck은 denoising loop이 아니라 prefix / prefill 계열 one-time cost가 훨씬 크다. Linear fit의 결과는 아래와 같다:

```text
T(num_steps) ≈ 14.14 ms + 0.795 ms × num_steps
```

즉 `num_steps=10`에서 대략:

```text
prefix-ish fixed cost     ≈ 14.14 ms  ≈ 63.5%
denoise loop total cost   ≈  7.95 ms  ≈ 35.7%
```

따라서 다음 profiling 목표는 단순하게 "10-step denoise loop가 느리다"가 아니라 아래 3가지를 밝히는 것이다:

```text
1. prefix / prefill 구간이 왜 14 ms나 드는가?
2. denoise step 1회당 약 0.8 ms는 어떤 kernel이 지배하는가?
3. N=10, N=12 주변에서 왜 residual이 커지는가?
```

결과를 더 자세히 해석해보자. 우선 scaling table의 핵심은 아래와 같다:

| `num_steps` | CUDA median |   증가량 from previous |
| ----------: | ----------: | ------------------: |
|           1 |   14.900 ms |                   - |
|           2 |   15.837 ms |           +0.937 ms |
|           4 |   17.329 ms | +1.492 ms / 2 steps |
|           6 |   18.813 ms | +1.484 ms / 2 steps |
|           8 |   19.814 ms | +1.001 ms / 2 steps |
|          10 |   22.270 ms | +2.456 ms / 2 steps |
|          12 |   24.725 ms | +2.454 ms / 2 steps |
|          16 |   26.323 ms | +1.598 ms / 4 steps |

전체 linear fit은 $R^{2} = 0.984$로 좋다. 그러나 완전하게 매끈한 선형은 아니다:

```text
N=8  residual = -0.684 ms
N=12 residual = +1.046 ms
N=16 residual = -0.535 ms
```

위 fluctuation의 원인으로는 아래와 같은 것들이 있을 수 있다:

```text
1. torch.compile graph specialization / recompilation / CUDA graph behavior
2. GPU clock / power / 공유 서버 노이즈
3. num_steps 값에 따른 generated graph 차이
4. denoise loop control-flow 최적화 차이
5. 측정 시점의 thermal / scheduler variation
```

하지만 **"prefix-ish fixed cost가 지배적이고, denoise step cost는 step당 약 0.8 ms 수준이다."**라는 큰 구조는 명확하다.

`num_steps`만 줄여서는 큰 speedup에 한계가 있다. denoising step을 10에서 1로 극단적으로 줄여도 model-only latency lower bound가 약 `14.9ms`이다:

```text
N=10 예상 latency ≈ 14.14 + 0.795 × 10 = 22.09 ms
N=5  예상 latency ≈ 14.14 + 0.795 × 5  = 18.11 ms
N=1  예상 latency ≈ 14.14 + 0.795 × 1  = 14.93 ms

N=10 observed / N=1 observed
= 22.27 / 14.90
≈ 1.49×
```

따라서 num_steps reduction만으로는 2× speedup이 어렵다. 2× 이상을 노리려면 반드시 prefix / prefill 계열 cost를 줄여야 한다. 이 값은 `N=1`에서도 이미 `14.9ms`가 걸린다. 이 시간에는 다음이 포함된다:

```text
preprocess_observation
image embedding
language embedding
prefix attention / prefill
KV cache generation
at least one denoise step
```

의심되는 bottleneck은 아래와 같다:

```text
1. vision encoder / image embedding
2. prefix transformer prefill
3. KV cache construction
4. masked right-wrist image가 실제로도 embedding되는지
5. attention implementation
6. prefix sequence length / image token count
```

## **Conclusion**

```text
policy.infer() full latency ≈ 23.5 ms
├─ sample_actions()       ≈ 21.0~22.3 ms
│  ├─ prefix-ish fixed    ≈ 14.1 ms
│  └─ denoise loop        ≈ 0.8 ms × num_steps
└─ policy wrapper         ≈ 2.4~2.5 ms
   ├─ observation_from_dict ≈ 1.92 ms
   └─ others                ≈ 0.5 ms
```







{% include comments.html %}
