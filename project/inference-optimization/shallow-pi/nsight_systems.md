---
layout: post
title: "3. Nsight Systems profiling"
nav_exclude: true
section: project
subcategory: shallow-pi
date: 2026-06-02
tags:
  - Korean
  - Python
  - Profiling
  - Nsight Systems
  - Writing
language: ko
summary: "Nsight Systems를 이용해서 bottleneck 지점을 더 정확하게 찾기"
math: true
comments: true
comment_id: "project-shallow-pi-nsight-systems"
permalink: /project/inference-optimization/shallow-pi/nsight-systems/
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



[지난 게시물](/project/inference-optimization/shallow-pi/shallow-pi-implementation/)에서 baseline 수치를 확인했다. 결론은 "**prefix-ish fixed cost가 지배적이고, denoise step cost는 step당 약 0.8 ms 수준이다**"였다. 이제 Nsight Systems로 bottleneck의 더 뜯어보자.

PyTorch Profiler로 operator table을 보는 것 대신, Nsight Systems trace를 먼저 찍는 이유는 아래와 같다:

1. prefix cluster와 denoise cluster가 timeline에서 분리되어 보일 가능성이 큼
2. N=1 trace와 N=10 trace를 비교하면 prefix one-time cost와 반복 cost를 시각적으로 분리 가능
3. kernel 사이 gap이 있는지, GPU가 계속 바쁜지, launch-bound인지 바로 확인 가능
4. 이후 PyTorch Profiler / Nsight Compute를 어디에 집중할지 결정 가능

그러므로 현재 목표는 **Nsight Systems로 N=1과 N=10의 compiled production timeline을 비교**하는 것이다.


## **code**

<details markdown="1">
<summary><code>profile_shallow_pi_latency.py</code></summary>

```python
#!/usr/bin/env python3

import argparse
import pathlib
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
    parser.add_argument("--fixed-noise", action="store_true")
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--cuda-profiler-api", action="store_true")

    return parser.parse_args()


def make_policy(args: argparse.Namespace):
    train_config = _config.get_config(args.config)

    return policy_config.create_trained_policy(
        train_config,
        args.ckpt,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": args.num_steps},
    )


def make_example() -> dict[str, Any]:
    return libero_policy.make_libero_example()


def prepare_model_only_observation(policy, example: dict[str, Any], device: str):
    inputs = jax.tree.map(lambda x: x, example)
    inputs = policy._input_transform(inputs)

    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
        inputs,
    )

    observation = _model.Observation.from_dict(inputs)
    return observation


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

    return policy._sample_actions(
        device,
        observation,
        noise=noise,
        num_steps=num_steps,
    )


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("[INFO] config:", args.config)
    print("[INFO] ckpt:", args.ckpt)
    print("[INFO] device:", args.device)
    print("[INFO] num_steps:", args.num_steps)
    print("[INFO] fixed_noise:", args.fixed_noise)
    print("[INFO] warmup:", args.warmup)
    print("[INFO] iters:", args.iters)
    print("[INFO] torch:", torch.__version__)
    print("[INFO] torch cuda:", torch.version.cuda)
    print("[INFO] device name:", torch.cuda.get_device_name(0))

    policy = make_policy(args)
    example = make_example()
    observation = prepare_model_only_observation(policy, example, args.device)

    noise = (
        make_fixed_noise(policy, observation, args.device, args.seed)
        if args.fixed_noise
        else None
    )

    print("[INFO] warmup start")
    for _ in range(args.warmup):
        _ = run_model_only(policy, args.device, observation, noise, args.num_steps)
    torch.cuda.synchronize()
    print("[INFO] warmup done")

    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()

    print("[INFO] trace iterations start")
    for i in range(args.iters):
        torch.cuda.nvtx.range_push(f"sample_actions_iter_{i}_num_steps_{args.num_steps}")
        out = run_model_only(policy, args.device, observation, noise, args.num_steps)
        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()

    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()

    print("[INFO] output shape:", tuple(out.shape))
    print("[INFO] output dtype:", out.dtype)
    print("[INFO] done")


if __name__ == "__main__":
    main()
```

</details>

위 script에서는 [지난 게시물](/project/inference-optimization/shallow-pi/shallow-pi-implementation/)에서 사용한 코드에서 `sample_actions()`만 포함하고 아래 항목들은 제외했다.

```text
input transform 반복
H2D 반복
policy wrapper
D2H
output transform
LIBERO simulation
Docker
websocket
```

## **N=1**

<details markdown="1">
<summary>shell commands & result</summary>

```shell
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export CUDA_LAUNCH_BLOCKING=0
export UV_LINK_MODE=copy

RUN_NAME=distill_l06_bf16_gb320_20260514_184612
STEP=30000
CKPT=./checkpoints/pi0_libero_l06/${RUN_NAME}/${STEP}

mkdir -p profiles/nsys

nsys profile \
  --trace=cuda,nvtx,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite=true \
  --output=profiles/nsys/shallow_pi_model_fixed_noise_N1 \
  uv run python scripts/profiling/trace_shallow_pi_model.py \
    --config pi0_libero_l06 \
    --ckpt "${CKPT}" \
    --device cuda:0 \
    --num-steps 1 \
    --fixed-noise \
    --warmup 40 \
    --iters 5 \
    --cuda-profiler-api
```
{: style="margin-left: 1rem;" }

```shell
nsys stats profiles/nsys/shallow_pi_model_fixed_noise_N1.nsys-rep \
  > profiles/nsys/shallow_pi_model_fixed_noise_N1_stats.txt
```
{: style="margin-left: 1rem;" }

```text
Generating SQLite file profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite from profiles/nsys/shallow_pi_model_fixed_noise_N1.nsys-rep
Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/nvtx_sum.py]...

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)   Style                 Range
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -------  ----------------------------------
     21.3       15,848,896          1  15,848,896.0  15,848,896.0  15,848,896  15,848,896          0.0  PushPop  :sample_actions_iter_0_num_steps_1
     19.9       14,767,837          1  14,767,837.0  14,767,837.0  14,767,837  14,767,837          0.0  PushPop  :sample_actions_iter_4_num_steps_1
     19.7       14,603,742          1  14,603,742.0  14,603,742.0  14,603,742  14,603,742          0.0  PushPop  :sample_actions_iter_1_num_steps_1
     19.6       14,542,300          1  14,542,300.0  14,542,300.0  14,542,300  14,542,300          0.0  PushPop  :sample_actions_iter_2_num_steps_1
     19.5       14,509,340          1  14,509,340.0  14,509,340.0  14,509,340  14,509,340          0.0  PushPop  :sample_actions_iter_3_num_steps_1

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/osrt_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain OS Runtime trace data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_api_sum.py]...

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)              Name
 --------  ---------------  ---------  -----------  -----------  --------  ----------  -----------  ----------------------------
     97.4       66,814,339         10  6,681,433.9  6,646,885.0   587,289  12,885,530  6,413,437.7  cudaStreamSynchronize
      1.7        1,193,016         10    119,301.6     93,560.0    24,487     318,236    102,948.5  cudaGraphLaunch_v10000
      0.8          549,121         60      9,152.0      6,229.5     3,796      54,702      7,845.5  cudaMemcpyAsync
      0.0           25,739          1     25,739.0     25,739.0    25,739      25,739          0.0  cuProfilerStart
      0.0           18,258          1     18,258.0     18,258.0    18,258      18,258          0.0  cudaDeviceSynchronize
      0.0            9,033         10        903.3        791.0       641       1,733        338.8  cudaStreamIsCapturing_v10000

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain CUDA kernel data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_mem_time_sum.py]...

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     87.7           76,321     50   1,526.4   1,440.0     1,120     2,080        352.0  [CUDA memcpy Device-to-Device]
     12.3           10,656     10   1,065.6   1,056.0     1,056     1,120         21.6  [CUDA memcpy Device-to-Host]

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_mem_size_sum.py]...

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      9.067     50     0.181     0.000     0.000     0.602        0.278  [CUDA memcpy Device-to-Device]
      0.000     10     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/openmp_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain OpenMP event data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/opengl_khr_range_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/opengl_khr_gpu_range_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/vulkan_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/vulkan_gpu_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx11_pix_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain DX11 CPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx12_gpu_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain DX12 GPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx12_pix_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain DX12 CPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/wddm_queue_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain WDDM context data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_total_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_cpu_page_faults_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/openacc_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain OpenACC event data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/syscall_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N1.sqlite does not contain syscall data.


```
{: style="margin-left: 1rem;" }

</details>

## **N=10**

<details markdown="1">
<summary>shell commands & result</summary>

```shell
nsys profile \
  --trace=cuda,nvtx,cublas,cudnn \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --force-overwrite=true \
  --output=profiles/nsys/shallow_pi_model_fixed_noise_N10 \
  uv run python scripts/profiling/trace_shallow_pi_model.py \
    --config pi0_libero_l06 \
    --ckpt "${CKPT}" \
    --device cuda:0 \
    --num-steps 10 \
    --fixed-noise \
    --warmup 40 \
    --iters 5 \
    --cuda-profiler-api
```
{: style="margin-left: 1rem;" }

```shell
nsys stats profiles/nsys/shallow_pi_model_fixed_noise_N10.nsys-rep \
  > profiles/nsys/shallow_pi_model_fixed_noise_N10_stats.txt
```
{: style="margin-left: 1rem;" }

```text
Generating SQLite file profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite from profiles/nsys/shallow_pi_model_fixed_noise_N10.nsys-rep
Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/nvtx_sum.py]...

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)   Style                  Range
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -------  -----------------------------------
     21.2       23,518,204          1  23,518,204.0  23,518,204.0  23,518,204  23,518,204          0.0  PushPop  :sample_actions_iter_0_num_steps_10
     19.7       21,848,142          1  21,848,142.0  21,848,142.0  21,848,142  21,848,142          0.0  PushPop  :sample_actions_iter_2_num_steps_10
     19.7       21,825,337          1  21,825,337.0  21,825,337.0  21,825,337  21,825,337          0.0  PushPop  :sample_actions_iter_1_num_steps_10
     19.7       21,824,597          1  21,824,597.0  21,824,597.0  21,824,597  21,824,597          0.0  PushPop  :sample_actions_iter_4_num_steps_10
     19.7       21,805,076          1  21,805,076.0  21,805,076.0  21,805,076  21,805,076          0.0  PushPop  :sample_actions_iter_3_num_steps_10

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/osrt_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain OS Runtime trace data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_api_sum.py]...

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)   Max (ns)   StdDev (ns)              Name
 --------  ---------------  ---------  -----------  ---------  --------  ----------  -----------  ----------------------------
     96.2       93,250,733         55  1,695,467.9  590,375.0   576,464  12,768,939  3,527,388.7  cudaStreamSynchronize
      2.6        2,474,195         55     44,985.4   25,669.0    20,300     315,433     59,054.6  cudaGraphLaunch_v10000
      0.8          757,540        105      7,214.7    6,119.0     3,716      54,652      5,649.4  cudaMemcpyAsync
      0.4          394,422         45      8,764.9    8,372.0     7,021      17,255      1,837.0  cudaLaunchKernel
      0.0           31,751         55        577.3      501.0       421       1,993        241.5  cudaStreamIsCapturing_v10000
      0.0           24,697          1     24,697.0   24,697.0    24,697      24,697          0.0  cuProfilerStart
      0.0           22,474          1     22,474.0   22,474.0    22,474      22,474          0.0  cudaDeviceSynchronize

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]...

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                                                  Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
    100.0           82,112         45   1,824.7   1,824.0     1,792     1,888         29.3  void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::TensorListMetadata<(in…

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_mem_time_sum.py]...

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     56.0           76,064     50   1,521.3   1,440.0     1,088     2,176        359.0  [CUDA memcpy Device-to-Device]
     44.0           59,873     55   1,088.6   1,088.0     1,056     1,216         36.7  [CUDA memcpy Device-to-Host]

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_mem_size_sum.py]...

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      9.067     50     0.181     0.000     0.000     0.602        0.278  [CUDA memcpy Device-to-Device]
      0.000     55     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/openmp_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain OpenMP event data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/opengl_khr_range_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/opengl_khr_gpu_range_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/vulkan_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/vulkan_gpu_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx11_pix_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain DX11 CPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx12_gpu_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain DX12 GPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx12_pix_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain DX12 CPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/wddm_queue_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain WDDM context data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_total_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_cpu_page_faults_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/openacc_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain OpenACC event data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/syscall_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10.sqlite does not contain syscall data.


```
{: style="margin-left: 1rem;" }

</details>



{% include comments.html %}
