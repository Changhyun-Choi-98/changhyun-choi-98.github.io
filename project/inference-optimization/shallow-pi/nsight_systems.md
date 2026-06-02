---
layout: post
title: "3. Nsight Systems profiling & further optimization"
nav_exclude: true
section: project
subcategory: shallow-pi
date: 2026-06-03
tags:
  - Korean
  - Python
  - Profiling
  - Nsight Systems
language: ko
summary: "Nsight Systems를 이용해서 bottleneck 지점을 더 정확하게 찾고 원인 분석 및 최적화"
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



[지난 게시물](/project/inference-optimization/shallow-pi/baseline-latency/)에서 baseline 수치를 확인했다. 결론은 "**prefix-ish fixed cost가 지배적이고, denoise step cost는 step당 약 0.8 ms 수준이다**"였다. 이제 Nsight Systems로 bottleneck을 더 뜯어보자.

## **1st step**

### **code**

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

위 script에서는 [지난 게시물](/project/inference-optimization/shallow-pi/baseline-latency/)에서 사용한 코드에서 `sample_actions()`만 포함하고 아래 항목들은 제외했다.

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

### **N=1 Setting**

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

### **N=10 Setting**

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


### **Result Analysis**

현재 `sample_actions()`는 `num_steps=N`일 때 대략 `N + 1`번의 cudaGraphLaunch + cudaStreamSynchronize + tiny D2H copy를 발생시키고 있다. (CUDA API Summary의 Num Calls 항목을 iteration 횟수인 5로 나눠서 확인)

이 패턴은 `sample_actions()` 내부의 GPU tensor 기반 `while` condition evaluation이 매 denoise step마다 CUDA scalar readback을 유발한다는 가설과 정합적이다. 그리고 이것을 [shallow-$\pi$의 코드](https://github.com/icsl-Jeon/openpi/blob/a5940ac510c7f5d94918f238cfc3722be1a2c5c8/src/openpi/models_pytorch/pi0_pytorch.py#L406){:target="_blank" rel="noopener noreferrer"}나 [original $\pi_0$의 코드](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models_pytorch/pi0_pytorch.py#L407){:target="_blank" rel="noopener noreferrer"}에서 실제로 확인했다. 해당 코드에는 아래의 두 핵심 요소가 있다:

1. `time`과 `dt`가 CUDA tensor
    ```python
    dt = torch.tensor(dt, dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    ```
2. `while` 조건문이 GPU tensor expression
    ```python
    while time >= -dt / 2:
    ```

Python은 `while` loop 탈출 여부를 CPU boolean으로 결정한다. 그러면 PyTorch는 이 scalar CUDA tensor의 값을 CPU 쪽으로 가져와서 조건을 판단해야 하고, 이 과정에서 tiny Device-to-Host copy와 stream synchronization이 발생할 수 있다. 이번 Nsight Systems 결과가 그 가설을 강하게 뒷받침한다:

```text
N=1:
  10 cudaStreamSynchronize / 5 iters = 2 per inference
  10 cudaGraphLaunch       / 5 iters = 2 per inference
  10 D2H memops            / 5 iters = 2 per inference

N=10:
  55 cudaStreamSynchronize / 5 iters = 11 per inference
  55 cudaGraphLaunch       / 5 iters = 11 per inference
  55 D2H memops            / 5 iters = 11 per inference
```

즉 실제 코드와 trace가 같이 가리키는 것은 아래와 같다:

> **while condition check가 denoise step마다 한 번, 그리고 마지막 종료 조건 확인에서 한 번 더 발생한다**

`sample_actions()` 안에서는 prefix prefill로 얻은 `past_key_values`를 `denoise_step()`에 넘긴다. `denoise_step()` 또한 [GitHub에서](https://github.com/icsl-Jeon/openpi/blob/a5940ac510c7f5d94918f238cfc3722be1a2c5c8/src/openpi/models_pytorch/pi0_pytorch.py#L421){:target="_blank" rel="noopener noreferrer"} [확인할 수 있다](https://github.com/Physical-Intelligence/openpi/blob/c23745b5ad24e98f66967ea795a07b2588ed6c79/src/openpi/models_pytorch/pi0_pytorch.py#L422){:target="_blank" rel="noopener noreferrer"}.

따라서 현재 inference는 아래와 같이 정리할 수 있다:

```text
sample_actions()
├─ prefix path: image/language embedding + prefix KV cache 생성
└─ denoise loop
   ├─ while GPU tensor condition check
   ├─ embed_suffix(state, x_t, timestep)
   ├─ suffix attention mask / position_ids 생성
   ├─ Gemma expert forward with cached prefix KV
   ├─ action_out_proj
   └─ Euler update
```

`sample_actions()`는 `PI0Pytorch` class의 `__init__`에서 `torch.compile(..., mode="max-autotune")`으로 compile된다. [PyTorch docs](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html){:target="_blank" rel="noopener noreferrer"}를 보면 `"max-autotune"`이 GPU에서 CUDA Graph를 기본적으로 enable한다고 설명한다. 이 option에는 아래와 같은 장점이 있다:

```text
1. 반복 inference에서 Python overhead 감소
2. 작은 kernel launch overhead 감소 가능
3. pointwise op fusion 가능
4. matmul / convolution config autotuning 가능
5. static shape inference에서 CUDA Graph replay로 overhead 감소 가능
```

하지만 아래와 같은 단점을 수반한다:

```text
1. 첫 실행 compile/autotune overhead가 큼
2. shape나 control flow가 바뀌면 recompile 가능
3. Python control flow와 GPU scalar condition이 있으면 graph가 끊길 수 있음
4. profiler에서 실제 kernel이 CUDA Graph 안에 숨어 보일 수 있음
5. max-autotune이 항상 빠른 것은 아니므로 default/reduce-overhead/no-cudagraphs ablation 필요
```

현재 관찰된 현상이 3, 4번 단점이다:

```text
GPU tensor scalar while condition
→ CPU가 매 step 조건을 확인해야 함
→ graph가 loop 전체를 하나로 깔끔하게 capture하기 어려움
→ Nsight Systems에는 cudaGraphLaunch / sync / tiny D2H가 num_steps+1 패턴으로 보임
```


## **2nd step (`for` loop patch)**

우선 앞에서 발견한 GPU tensor scalar condition을 Python while 조건으로 평가하는 부분을 제거하겠다. 이 부분이 Nsight Systems 결과에서 보인 `num_steps + 1`개의 `cudaGraphLaunch`, `cudaStreamSynchronize`, tiny D2H memop 패턴과 직접 연결되는 강한 후보이기 때문이다.

`sample_actions()`의 `while time >= -dt / 2:`를 `for _ in range(num_steps):`로 바꿨다. latency를 확인해보니 약 `1ms` 줄었다(약 5.1% speedup). Tail latency는 크게 안정화되었다(p95 기준 약 `2ms` 개선). Nsight Systems의 결과는 아래와 같다:

<details markdown="1">
<summary>shell commands & result</summary>

```text
Generating SQLite file profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite from profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.nsys-rep
Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/nvtx_sum.py]...

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style                  Range
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  -----------------------------------
     44.8        2,421,582          1  2,421,582.0  2,421,582.0  2,421,582  2,421,582          0.0  PushPop  :sample_actions_iter_0_num_steps_10
     14.5          782,127          1    782,127.0    782,127.0    782,127    782,127          0.0  PushPop  :sample_actions_iter_1_num_steps_10
     13.7          738,652          1    738,652.0    738,652.0    738,652    738,652          0.0  PushPop  :sample_actions_iter_4_num_steps_10
     13.6          734,826          1    734,826.0    734,826.0    734,826    734,826          0.0  PushPop  :sample_actions_iter_3_num_steps_10
     13.5          731,742          1    731,742.0    731,742.0    731,742    731,742          0.0  PushPop  :sample_actions_iter_2_num_steps_10

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/osrt_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain OS Runtime trace data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_api_sum.py]...

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)              Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------
     97.9       93,249,397          1  93,249,397.0  93,249,397.0  93,249,397  93,249,397          0.0  cudaDeviceSynchronize
      1.7        1,633,115          5     326,623.0     231,805.0     228,521     702,458    210,151.0  cudaGraphLaunch_v10000
      0.4          346,485         50       6,929.7       3,935.5       3,425      64,576      9,658.1  cudaMemcpyAsync
      0.0           31,096          1      31,096.0      31,096.0      31,096      31,096          0.0  cuProfilerStart
      0.0            4,517          5         903.4         641.0         631       1,913        565.1  cudaStreamIsCapturing_v10000

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_kern_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain CUDA kernel data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_mem_time_sum.py]...

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
    100.0           74,656     50   1,493.1   1,312.0     1,088     2,208        409.5  [CUDA memcpy Device-to-Device]

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/cuda_gpu_mem_size_sum.py]...

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 ----------  -----  --------  --------  --------  --------  -----------  ------------------------------
      9.067     50     0.181     0.000     0.000     0.602        0.278  [CUDA memcpy Device-to-Device]

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/openmp_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain OpenMP event data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/opengl_khr_range_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain KHR Extension (KHR_DEBUG) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/opengl_khr_gpu_range_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain GPU KHR Extension (KHR_DEBUG) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/vulkan_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain Vulkan Debug Extension (Vulkan Debug Util) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/vulkan_gpu_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain GPU Vulkan Debug Extension (GPU Vulkan Debug markers) data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx11_pix_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain DX11 CPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx12_gpu_marker_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain DX12 GPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/dx12_pix_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain DX12 CPU debug markers.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/wddm_queue_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain WDDM context data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_total_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/um_cpu_page_faults_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain CUDA Unified Memory CPU page faults data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/openacc_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain OpenACC event data.

Processing [profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite] with [/usr/local/cuda-13.0/nsight-systems-2025.3.2/host-linux-x64/reports/syscall_sum.py]...
SKIPPED: profiles/nsys/shallow_pi_model_fixed_noise_N10_after_forloop.sqlite does not contain syscall data.


```
{: style="margin-left: 1rem;" }

</details>

Sanity check도 완료했다. 변경 전과 후의 action 차이가 허용 범위 이내였다.

주의할 점은 for-loop 변경 후 NVTX range 시간이 곧바로 end-to-end inference latency를 의미하지 않는다는 점이다. 기존 while 버전에서는 매 denoise step마다 GPU scalar condition readback이 발생하면서 `cudaStreamSynchronize`가 loop 내부에 들어갔기 때문에 NVTX range 안에 대기 시간이 포함되었다. 반면 for-loop 변경 후에는 per-step synchronization이 제거되어 `sample_actions()` 호출은 대부분 CUDA Graph launch를 enqueue하고 빠르게 반환한다. 따라서 Nsight Systems의 NVTX range는 주로 CPU-side enqueue time을 보여주고, 실제 GPU work 대기 시간은 trace loop 뒤의 최종 `torch.cuda.synchronize()`에 모인다. 즉 `cudaDeviceSynchronize` 93 ms는 새로운 병목이라기보다, 5회 inference에 대한 queued GPU work를 마지막에 기다린 결과로 해석해야 한다.

| 항목                   |           Original | Minimal for-loop inplace | 판단                               |
| -------------------- | ----------------- | ----------------------- | -------------------------------- |
| CUDA median          |          21.271 ms |            **20.220 ms** | **약 1.05 ms 개선**                 |
| CUDA p95             |          22.733 ms |            **20.383 ms** | **tail latency 크게 개선**           |
| sync wall median     |          21.264 ms |            **20.231 ms** | 일관된 개선                           |
| `cudaGraphLaunch`    | 55 calls / 5 iters |    **5 calls / 5 iters** | `11 → 1` per inference           |
| D2H memops           | 55 calls / 5 iters |                    **0** | GPU scalar condition readback 제거 |




## **3rd step (`num_steps` scaling 재실행)**

[지난 게시물](/project/inference-optimization/shallow-pi/baseline-latency/)에서 구한 linear fit은 아래와 같았다:

```text
T(num_steps) ≈ 14.1388 ms + 0.7950 ms × num_steps
R² ≈ 0.984
```

이제 다음 질문을 해결한다:

```text
while→for patch가 intercept를 줄였는가?
slope를 줄였는가?
N=10 주변 tail jitter만 줄였는가?
```

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

mkdir -p profiles/latency

for N in 1 2 4 6 8 10 12 16; do
  echo "==== after minimal for-loop, num_steps=${N} ===="

  uv run python scripts/profiling/profile_shallow_pi_latency.py \
    --config pi0_libero_l06 \
    --ckpt "${CKPT}" \
    --device cuda:0 \
    --num-steps ${N} \
    --mode model \
    --fixed-noise \
    --warmup 30 \
    --iters 100 \
    --out-json profiles/latency/model_fixed_noise_numsteps${N}_100iters_after_forloop_inplace.json
done
```
{: style="margin-left: 1rem;" }

```python
uv run python - <<'PY'
import json
import pathlib
import numpy as np

base = pathlib.Path("profiles/latency")
steps = [1, 2, 4, 6, 8, 10, 12, 16]

rows = []

for n in steps:
    path = base / f"model_fixed_noise_numsteps{n}_100iters_after_forloop_inplace.json"
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

```
num_steps,cuda_median,cuda_p95,wall_median,wall_p95
1,14.4742,14.6493,14.4983,14.6487
2,15.3308,15.5085,15.3477,15.5184
4,16.5324,16.7055,16.5277,16.7273
6,17.7069,17.9528,17.7119,17.9586
8,18.9512,19.1642,18.9647,19.1770
10,20.0278,20.2585,20.0399,20.2174
12,21.4829,21.5398,21.4986,21.5723
16,24.0840,24.1193,24.0871,24.1350

Linear fit using CUDA event median:
  T(num_steps) ≈ 13.9442 ms + 0.6277 ms * num_steps
  intercept / prefix-ish cost: 13.9442 ms
  per denoise step cost:       0.6277 ms
  R^2:                         0.998912

Residuals:
  N= 1: observed=14.4742 ms, fitted=14.5720 ms, residual=-0.0977 ms
  N= 2: observed=15.3308 ms, fitted=15.1997 ms, residual=+0.1311 ms
  N= 4: observed=16.5324 ms, fitted=16.4552 ms, residual=+0.0772 ms
  N= 6: observed=17.7069 ms, fitted=17.7106 ms, residual=-0.0037 ms
  N= 8: observed=18.9512 ms, fitted=18.9661 ms, residual=-0.0149 ms
  N=10: observed=20.0278 ms, fitted=20.2216 ms, residual=-0.1938 ms
  N=12: observed=21.4829 ms, fitted=21.4770 ms, residual=+0.0058 ms
  N=16: observed=24.0840 ms, fitted=23.9880 ms, residual=+0.0960 ms
```
{: style="margin-left: 1rem;" }


</details>

### **Result analysis**

```text
Before:
T(num_steps) ≈ 14.1388 ms + 0.7950 ms × num_steps
R² = 0.984047

After minimal while→for:
T(num_steps) ≈ 13.9442 ms + 0.6277 ms × num_steps
R² = 0.998912
```

예상할 수 있었듯이, 2nd step에서 진행했던 patch의 주요 효과는 prefix fixed cost 감소가 아니라 denoise step당 반복 overhead 감소이다.

| 항목               |         Before |              After |                            변화 |
| ---------------- | -------------: | -----------------: | ----------------------------: |
| intercept        |     14.1388 ms |         13.9442 ms |                    -0.1946 ms |
| per-step slope   | 0.7950 ms/step | **0.6277 ms/step** | **-0.1673 ms/step, 약 -21.0%** |
| R²               |         0.9840 |         **0.9989** |                scaling 훨씬 선형화 |
| N=10 CUDA median |     22.2703 ms |     **20.0278 ms** |                **-2.2425 ms** |
| N=10 CUDA p95    |     24.2536 ms |     **20.2585 ms** |                **-3.9951 ms** |

추가적으로, scaling trend가 훨씬 깨끗해졌는데 이는 아래와 같이 유추해볼 수 있다:

```text
Before:
  tensor while condition 때문에 graph replay / sync / D2H pattern이 num_steps별로 지저분하게 섞임

After:
  for-loop로 graph structure가 안정화되어 latency model이 거의 선형화됨
```

### **policy-level latency check**

<details markdown="1">
<summary>shell command & result</summary>
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
  --out-json profiles/latency/policy_numsteps10_100iters_after_forloop_inplace.json
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
    "mean_ms": 22.842999000549316,
    "median_ms": 22.828096389770508,
    "p90_ms": 22.956928253173828,
    "p95_ms": 22.97929573059082,
    "p99_ms": 22.9935359954834,
    "min_ms": 22.764095306396484,
    "max_ms": 22.99625587463379
  },
  "sync_wall": {
    "count": 100,
    "mean_ms": 22.830096010075067,
    "median_ms": 22.815933500169194,
    "p90_ms": 22.845122002763674,
    "p95_ms": 22.857551000925014,
    "p99_ms": 22.92256800137693,
    "min_ms": 22.768767001252854,
    "max_ms": 24.132639002345968
  }
}
```
{: style="margin-left: 1rem;" }

</details>

model-only 개선이 policy-level로 잘 전달된 것을 확인해볼 수 있다.

| Metric           | Original policy | After minimal `for` patch |            개선 |
| ---------------- | --------------: | ------------------------: | ------------: |
| CUDA median      |       23.764 ms |             **22.828 ms** | **-0.936 ms** |
| CUDA p95         |       25.102 ms |             **22.979 ms** | **-2.123 ms** |
| Sync wall median |       23.741 ms |             **22.816 ms** | **-0.925 ms** |
| Sync wall p95    |       25.037 ms |             **22.858 ms** | **-2.180 ms** |
| Sync wall p99    |       26.103 ms |             **22.923 ms** | **-3.181 ms** |

```text
1. model-only speedup이 policy-level API latency에도 그대로 전달됨.
2. median은 약 3.9% 개선.
3. p95/p99 tail latency는 훨씬 크게 개선.
4. policy wrapper overhead는 거의 그대로이고, sample_actions 내부 개선이 주효과.
```



{% include comments.html %}
