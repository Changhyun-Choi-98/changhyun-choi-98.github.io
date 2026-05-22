---
layout: default
title: "Shallow-π Profiling"
nav_exclude: true
section: project
subcategory: further-optimizing-shallow-pi
date: 2026-05-21
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

위 코드는 [실제 LIBERO observation 대신 shape만 같은 랜덤한 observation과 "do something"이라는 prompt를 줘서](https://github.com/icsl-Jeon/openpi/blob/a5940ac510c7f5d94918f238cfc3722be1a2c5c8/src/openpi/policies/libero_policy.py#L10){:target="_blank" rel="noopener noreferrer"} inference path에만 집중한다. 두 가지 `mode`로 돌아가는데



{% include comments.html %}
