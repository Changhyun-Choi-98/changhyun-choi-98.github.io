---
layout: post
title: "2. Limited Closed-loop Reproduction, Route-level Profiling, and Wrist-camera Robustness"
nav_exclude: true
section: project
subcategory: realtime-vla-flash
date: 2026-06-16
tags:
  - English
  - Profiling
  - Python
language: en
summary: "A limited closed-loop reproduction and probing project for Realtime-VLA FLASH on Runpod L40S: official checkpoint conversion, LIBERO Goal baseline, synchronized route-level profiling, wrist-camera dropout robustness, and a minimal WristHealthGuard extension"
math: true
comments: true
comment_id: "project-realtime-vla-flash-final"
permalink: /project/paper-to-prototype-lab/realtime-vla-flash/final/
---

## **TL;DR**

I ran **Realtime-VLA FLASH** on a Runpod NVIDIA L40S instance using the official repository and public checkpoints. I resolved and converted the public `pi0_libero` base checkpoint and `draft_libero_goal` draft checkpoint into Triton artifacts, and then executed a limited closed-loop LIBERO Goal baseline. The project culminated in synchronized route-level profiling, a wrist-camera dropout robustness probe, and a minimal local inference-time extension called **WristHealthGuard**.

The most important result is not “paper reproduced.” The correct result is more precise:

> **This is a completed limited closed-loop reproduction + probing + profiling + minimal extension project. It partially supports the Realtime-VLA FLASH latency mechanism in a limited setting, but it is not a full paper reproduction, not a hardware-exact latency reproduction, and not a full robustness evaluation.**

The final measured highlights were:

| Component                            |                                         Result |
| ------------------------------------ | ---------------------------------------------: |
| Stage 4 limited baseline             |         27/30 success on LIBERO Goal tasks 0–2 |
| Stage 4 route mix                    |                         full = 88, draft = 276 |
| Stage 4 draft/full ratio             |                                  0.758 / 0.242 |
| Stage 4 accepted prefix mean         |                                         10.074 |
| Stage 6 synchronized server-side p50 |                                       8.083 ms |
| Stage 6 draft route p50              |                                       8.067 ms |
| Stage 6 full route p50               |                                      33.156 ms |
| Stage 6 client roundtrip p50         |                                      11.957 ms |
| Stage 5 `wrist_zero_every4`          |                                   4/15 success |
| Stage 5 `wrist_zero_all`             |                                   0/15 success |
| Stage 7 WristHealthGuard `every4`    |                                   6/15 success |
| Stage 7 WristHealthGuard `allzero`   |                                    0/9 success |

The paper claims that FLASH reduces LIBERO inference latency by replacing many 58.0 ms full-inference rounds with speculative rounds as fast as 7.8 ms, lowering task-level average latency to 19.1 ms with a 3.04× speedup and only a 0.3-point average success drop. My project supports the **mechanism-level story** in a limited Runpod L40S setting, but it does not reproduce the full paper benchmark or the real-world conveyor experiment. ([arXiv][1]{:target="_blank" rel="noopener noreferrer"})

---

# **1. Why this paper**

Realtime-VLA FLASH is interesting because it attacks a real deployment bottleneck in diffusion-based vision-language-action models, or dVLAs. dVLAs can generate high-quality continuous action chunks, but full-path inference can be too slow for reactive closed-loop robot control. In a robot system, the problem is not just throughput. The policy must repeatedly consume fresh observations, produce action chunks, and avoid executing stale actions when the scene has changed.

The project page describes the central motivation clearly: synchronous full-path inference can make robot commands stale in reactive scenes, and FLASH replaces slow full-path replanning with speculative rounds that execute only the verified action prefix and fall back when the draft is inconsistent. ([Project Page][2]{:target="_blank" rel="noopener noreferrer"})

This makes the paper attractive as a **paper-to-prototype-lab** project for three reasons.

First, the paper’s claim is not purely model-centric. It is a runtime and control-loop claim. The system has a full path, a draft path, parallel verification, accepted-prefix execution, and fallback. If any part of that chain fails in practice, the paper’s real-time control story becomes much weaker.

Second, the official repository and public checkpoint path make the project reproducible enough to audit systematically. The repository provides installation instructions, Triton conversion commands, a policy server command, and a LIBERO client command. It also explicitly supports a workflow where the LIBERO client/evaluation code can run in a separate environment. ([GitHub][3]{:target="_blank" rel="noopener noreferrer"})

Third, the failure modes are valuable even when the project does not fully reproduce the paper. Closed-loop robot model projects often fail because of renderer issues, checkpoint ambiguity, server/client incompatibility, action-shape mismatch, logging gaps, or invalid latency measurement. This project was designed to make those failure modes explicit instead of hiding them.

---

# **2. What this project claims and does not claim**

This project makes a narrow set of claims.

## **This project claims**

1. I successfully ran the official Realtime-VLA FLASH repository on Runpod L40S under a limited evaluation scope.
2. I resolved the public `pi0_libero` base checkpoint and converted both base and draft artifacts into Triton layout.
3. I booted the official `pi0_libero` Triton policy server and connected it to the LIBERO client.
4. I executed a limited closed-loop LIBERO Goal baseline on tasks 0–2 and obtained 27/30 success.
5. I added synchronized server-side timing and observed that the draft route p50 latency was lower than the full route p50 latency in this limited setting.
6. I probed synthetic wrist-camera dropout and found severe degradation.
7. I implemented a minimal local inference-time extension, WristHealthGuard, and observed modest recovery under intermittent dropout.

## **This project does not claim**

1. It does not claim full paper reproduction.
2. It does not claim full LIBERO benchmark coverage.
3. It does not claim hardware-exact paper latency reproduction.
4. It does not claim real-world conveyor reproduction.
5. It does not claim general robustness.
6. It does not claim that WristHealthGuard is an upstream-quality method.

The [Stage 8 claim matrix](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage8_claim_matrix.md){:target="_blank" rel="noopener noreferrer"} explicitly labels the full benchmark and real-world conveyor result as **not claimed**, while the limited baseline, latency mechanism, wrist-dropout probe, and WristHealthGuard extension are marked as limited or partial support only.

---

# **3. Paper claim decomposition**

I did not treat “reproduce Realtime-VLA FLASH” as one monolithic objective. Instead, I decomposed the paper and project into progressively stronger claims.

| Claim ID | Claim                                                                   | My result                     | Status                      |
| -------- | ----------------------------------------------------------------------- | ----------------------------- | --------------------------- |
| C1       | Official repo and model environment can be prepared                     | model env and imports passed  | supported                   |
| C2       | Draft and base artifacts for `pi0_libero` can be resolved and converted | draft/base conversion passed  | supported                   |
| C3       | Policy server can boot with converted artifacts                         | server readiness reached      | supported                   |
| C4       | End-to-end server/client/LIBERO integration works for one task          | 2/2 task 0 episodes succeeded | supported                   |
| C5       | FLASH+Triton can complete a limited LIBERO Goal baseline                | 27/30 success on tasks 0–2    | supported limited           |
| C6       | FLASH latency mechanism can be profiled                                 | draft p50 < full p50          | partially supported limited |
| C7       | Flash route and accepted-prefix behavior can be inspected               | route/prefix logs collected   | supported limited           |
| C8       | Wrist-camera dropout robustness can be probed synthetically             | strong degradation observed   | negative limited            |
| C9       | WristHealthGuard can recover some intermittent dropout                  | 4/15 → 6/15                   | partially supported limited |
| C10      | Full paper benchmark is reproduced                                      | not attempted                 | not claimed                 |
| C11      | Real-world conveyor result is reproduced                                | not attempted                 | not claimed                 |

This decomposition matters. It prevents a common reproduction mistake: using one successful smoke test as evidence for a much larger claim.

The project’s final status is therefore:

> **complete limited closed-loop reproduction + probing + profiling + minimal extension; full paper reproduction: no; paper claim status: partially supported in limited setting.**

---

# **4. Method recap: Realtime-VLA FLASH**

Realtime-VLA FLASH can be understood as a dual-path runtime for diffusion-based VLA policies.

## **4.1 Full path**

The full path is the reliable but expensive route. It runs the full image encoding, VLM prefill, and action denoising process. In a closed-loop robot system, this path refreshes the high-fidelity context and provides the anchor against which speculative outputs can be checked.

## **4.2 Draft path**

The draft path generates a candidate action chunk more cheaply. The idea is not to discard the main model, but to avoid rerunning the full pipeline at every replanning round. The draft path becomes valuable only if it is sufficiently fast and sufficiently aligned with the full model’s action behavior.

The official repository highlights “speculative inference as fast as 7.8 ms” and customized Triton serving as central to the system. ([GitHub][3]{:target="_blank" rel="noopener noreferrer"})

## **4.3 Parallel verification**

The difficult part is that dVLAs output continuous actions through flow matching or denoising, not discrete language tokens. Therefore, token-level speculative decoding cannot be used directly.

FLASH instead verifies draft action chunks using the main Action Expert at selected flow-matching timesteps. The project page summarizes this as a flash path that drafts and verifies a candidate action chunk and returns the longest consistent prefix. ([Project Page][2]{:target="_blank" rel="noopener noreferrer"})

## **4.4 Accepted prefix**

The accepted prefix is the portion of the candidate action chunk that the system decides to execute. This is a control-loop variable, not just an inference artifact. A longer accepted prefix reduces replanning frequency and can improve runtime efficiency, but it also risks executing stale actions if the environment changes or the observation is corrupted.

In my experiments, accepted prefix was useful but not sufficient as an uncertainty signal. In the clean Stage 4 baseline, the accepted prefix mean was 10.074 and the p50/p95/p99 were all 12. Under wrist-camera dropout, the mean decreased, but high percentiles often remained saturated at 12. This means the accepted prefix captured some degradation but did not reliably flag all failure-prone states.

## **4.5 Phase-aware fallback**

FLASH also includes fallback logic. In principle, smooth motion phases tolerate approximate draft behavior, while precision-sensitive phases such as final alignment or gripper switching should fall back to the full path. My project did not isolate phase-aware fallback at the same depth as the paper, but it did track route mix, accepted prefix, success, and failure modes across clean, perturbed, and guarded settings.

---

# **5. Action representation and control-loop meaning**

The policy returned action chunks with shape `(50, 7)`. In the LIBERO setting, this is a continuous robot control representation rather than a discrete token sequence.

The critical control-loop variables in this project were:

| Variable                           | Why it matters                                                        |
| ---------------------------------- | --------------------------------------------------------------------- |
| Action chunk shape                 | Confirms policy-client action interface compatibility                 |
| Route type                         | Indicates whether the runtime used full or draft path                 |
| Accepted prefix length             | Determines how many actions are executed before replanning            |
| Client roundtrip                   | Measures effective closed-loop delay observed by the simulator client |
| Server-side action generation time | Measures model-side computation more directly                         |
| Success/failure at horizon         | Captures whether the closed-loop system actually solved the task      |

This is why I did not stop at server boot. Server boot proves readiness; it does not prove closed-loop behavior. The project only became meaningful after the LIBERO client received valid action chunks, stepped the simulator, produced episode logs, and completed tasks.

---

# **6. Stage 0: scope lock**

I locked the scope before moving into model and checkpoint setup.

The scope lock stated:

* closed-loop simulator scope: allowed
* checkpoint/model setup: allowed
* full benchmark: not allowed yet
* current project type: conditional partial closed-loop reproduction
* paper reproduction claim: not allowed yet

This was not bureaucracy. It prevented a common failure mode: treating setup progress as reproduction evidence. The [Stage 0 scope document](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage0_scope_lock.md){:target="_blank" rel="noopener noreferrer"} explicitly preserved the distinction between infrastructure readiness, model readiness, and paper reproduction.

---

# **7. Stage 1–2B: model, checkpoint, and server readiness**

## **7.1 Model environment**

The top-level model environment was kept separate from the `.venv-libero` client environment.

The final model environment used:

| Component    | Version     |
| ------------ | ----------- |
| Python       | 3.11.13     |
| torch        | 2.7.1+cu126 |
| torch CUDA   | 12.6        |
| Triton       | 3.3.1       |
| JAX          | 0.5.3       |
| transformers | 4.53.2      |
| GPU          | NVIDIA L40S |

This separation was important because the official repository requires Python ≥3.11 and dependencies such as `torch==2.7.1`, `triton==3.3.1`, `jax[cuda12]==0.5.3`, and `transformers==4.53.2`, while LIBERO uses a different evaluation environment. The repository’s `pyproject.toml` confirms the Python and major dependency requirements. ([GitHub][4]{:target="_blank" rel="noopener noreferrer"})

The import smoke tested `torch`, `triton`, `jax`, `transformers`, `openpi`, `openpi.training.config`, `openpi.policies.policy_config`, CUDA availability, JAX devices, and the transformer replacement patch.

## **7.2 Draft checkpoint**

The draft checkpoint `draft_libero_goal.pt` was downloaded and loaded on CPU. The [Stage 2 draft load script](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/scripts/stage2_torch_load_draft.py){:target="_blank" rel="noopener noreferrer"} recorded file existence, file size, top-level object type, keys, tensor count, and tensor shapes/dtypes without printing large tensors.

The draft Triton conversion succeeded and produced a local-only `draft_triton.pkl`.

## **7.3 Base checkpoint blocker and resolution**

The first serious blocker was base checkpoint ambiguity. The repository quick start expects a JAX checkpoint path for base conversion, but the exact public path for `pi0_libero` had to be resolved carefully. The official README shows the expected conversion flow: first convert a pretrained base checkpoint with `--mode base --jax-path /path/to/jax/checkpoint`, then convert the draft checkpoint, then serve with `--config pi0_libero`, `--base-triton-path`, `--draft-triton-path`, and `--backend triton`. ([GitHub][3]{:target="_blank" rel="noopener noreferrer"})

I did not force a blind large download. Instead, I created a [metadata-only checkpoint probe](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/scripts/stage2b_checkpoint_metadata_probe.py){:target="_blank" rel="noopener noreferrer"} and classified candidates. The probe explicitly distinguished:

* `MATCH_FLASH_PI0_LIBERO_JAX`
* `MATCH_FLASH_PI0_LIBERO_TORCH_ONLY`
* `BASE_PI0_PRETRAIN_ONLY`
* `PUBLIC_PI05_LIBERO_MISMATCH`
* `LOCAL_PLACEHOLDER_ONLY`
* `NOT_FOUND`
* `UNKNOWN_ACCESS_ERROR`

The metadata probe classified `gs://openpi-assets/checkpoints/pi0_libero` as the correct JAX/Orbax-style root only if it had `params`, `assets`, and checkpoint metadata.

The [base checkpoint resolver](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/scripts/stage2b_resolve_base_checkpoint.py){:target="_blank" rel="noopener noreferrer"} then downloaded only the single candidate classified as `MATCH_FLASH_PI0_LIBERO_JAX`, validated that it contained `params`, `assets`, checkpoint metadata, and norm stats, and wrote the [final base resolution decision](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/results/stage2b_base_resolution_decision.json){:target="_blank" rel="noopener noreferrer"}.

This resolved the blocker:

* Base checkpoint: `gs://openpi-assets/checkpoints/pi0_libero`
* Local size: approximately 12 GB
* Converted artifact: `converted/base/base_weights.pkl`
* Norm stats: `assets/physical-intelligence/libero/norm_stats.json`

All checkpoint and converted-weight artifacts were kept local-only and excluded from Git. The [Stage 8 artifact index](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage8_artifact_index.md){:target="_blank" rel="noopener noreferrer"} confirms that checkpoints, converted weights, videos, datasets, profiler binaries, private endpoints, and credentials were excluded from Git.

## **7.4 Server boot**

With converted base and draft artifacts present, the [server boot smoke](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/scripts/stage2b_server_boot_smoke.sh){:target="_blank" rel="noopener noreferrer"} used the official serving path:

* `--config pi0_libero`
* `--base-triton-path converted/base`
* `--draft-triton-path converted/draft_goal`
* `--task-suite-name libero_goal`
* `--backend triton`

The boot smoke reached readiness on port 8000.

This still did not prove closed-loop success. It only proved that the server could start with the converted artifacts.

---

# **8. Stage 3: one-task closed-loop smoke**

Stage 3 was the first real end-to-end closed-loop test.

Setup:

| Item           | Value         |
| -------------- | ------------- |
| Suite          | `libero_goal` |
| Task           | 0             |
| Trials         | 2             |
| Seed           | 7             |
| Render backend | EGL           |
| Server backend | Triton        |
| Config         | `pi0_libero`  |

The server was launched with the converted `pi0_libero` base and `draft_libero_goal` draft artifacts. The LIBERO client ran in `.venv-libero` with EGL offscreen rendering.

Results:

| Metric                       |                Value |
| ---------------------------- | -------------------: |
| Episodes requested           |                    2 |
| Episodes completed           |                    2 |
| Success                      |                    2 |
| Success rate                 |                1.000 |
| Infer calls                  |                   21 |
| Route counts                 | full = 2, draft = 19 |
| Accepted prefix mean/p50/p95 |         12 / 12 / 12 |
| Peak VRAM                    |             7390 MiB |

Stage 3 verified that the server, client, policy, action interface, simulator, logs, and videos were connected. It also confirmed action chunks with valid shape. The [Stage 3 report](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage3_closed_loop_smoke_report.md){:target="_blank" rel="noopener noreferrer"} correctly frames this as a tiny smoke test rather than a benchmark result.

---

# **9. Stage 4: limited closed-loop baseline**

Stage 4 was the clean baseline.

Setup:

| Item                    | Value         |
| ----------------------- | ------------- |
| Suite                   | `libero_goal` |
| Tasks                   | 0, 1, 2       |
| Episodes per task       | 10            |
| Total measured episodes | 30            |
| Seed                    | 7             |
| Backend                 | Triton + EGL  |
| Warm-up                 | excluded      |

Results:

|      Task | Episodes | Success | Success rate |
| --------: | -------: | ------: | -----------: |
|         0 |       10 |       9 |        0.900 |
|         1 |       10 |       9 |        0.900 |
|         2 |       10 |       9 |        0.900 |
| Aggregate |       30 |      27 |        0.900 |

Route behavior:

| Metric               |  Value |
| -------------------- | -----: |
| Full route count     |     88 |
| Draft route count    |    276 |
| Draft ratio          |  0.758 |
| Full ratio           |  0.242 |
| Accepted prefix mean | 10.074 |

This is the strongest closed-loop baseline result in the project. It shows that the official setup can run a nontrivial limited LIBERO Goal subset on Runpod L40S.

But it is still not a full paper reproduction. It used only LIBERO Goal tasks 0–2, one seed, and 30 measured episodes. The [Stage 8 final metrics](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/results/stage8_final_metrics.json){:target="_blank" rel="noopener noreferrer"} and [project summary](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/results/stage8_project_summary.json){:target="_blank" rel="noopener noreferrer"} explicitly preserve this limited interpretation.

---

# **10. Stage 6: synchronized latency profiling**

Stage 4 already had latency fields, but they were preliminary wall-clock timings. That was not enough for a latency-focused paper.

Therefore, Stage 6 added synchronized server-side timing. The code change is captured in [scripts/stage6_apply_server_timing_patch.py](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/scripts/stage6_apply_server_timing_patch.py){:target="_blank" rel="noopener noreferrer"} and [patches/stage6_server_timing.patch](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/patches/stage6_server_timing.patch){:target="_blank" rel="noopener noreferrer"}. The key new field was:

* `policy_time_gpu_sync_ms`

The measurement pattern used `torch.cuda.synchronize()` before and after the server-side action-generation path. This does not make the experiment hardware-exact relative to the paper, but it is much better than unsynchronized wall-clock timing.

Setup:

| Item                    | Value                           |
| ----------------------- | ------------------------------- |
| Suite                   | `libero_goal`                   |
| Tasks                   | 0, 1, 2                         |
| Episodes per task       | 3                               |
| Total measured episodes | 9                               |
| Warm-up                 | excluded                        |
| Timing                  | server-side synchronized timing |
| Claim status            | `PARTIALLY_SUPPORTED_LIMITED`   |

Results:

| Metric                        |     Value |
| ----------------------------- | --------: |
| `policy_time_gpu_sync_ms` p50 |  8.083 ms |
| `policy_time_gpu_sync_ms` p95 | 33.501 ms |
| Draft route p50               |  8.067 ms |
| Full route p50                | 33.156 ms |
| Client roundtrip p50          | 11.957 ms |

Interpretation:

The draft route was substantially faster than the full route at p50 in this limited setting. This supports the latency mechanism: speculative/draft rounds can be much cheaper than full-path rounds.

But the result is not a hardware-exact paper latency reproduction. It used Runpod L40S, a small LIBERO Goal subset, and custom synchronized timing fields. The Stage 8 final metrics label the latency claim as `PARTIALLY_SUPPORTED_LIMITED`.

Also, p99 latency remained noisy due to residual first-use and route-specific warm-up tails. This is why the blog should emphasize p50/p95 mechanism-level support and avoid overclaiming p99 stability. The [caveats document](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage8_negative_results_and_caveats.md){:target="_blank" rel="noopener noreferrer"} explicitly says Stage 6 p99 values should be interpreted cautiously.

---

# **11. Stage 5: wrist-camera dropout robustness probe**

Stage 5 added a synthetic robustness probe. The client-side perturbation is captured in [scripts/stage5_apply_client_perturb_patch.py](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/scripts/stage5_apply_client_perturb_patch.py){:target="_blank" rel="noopener noreferrer"} and [patches/stage5_client_camera_perturb.patch](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/patches/stage5_client_camera_perturb.patch){:target="_blank" rel="noopener noreferrer"}.

The perturbation targeted the wrist camera image:

* Source key: `robot0_eye_in_hand_image`
* Sent as: `observation/wrist_image`

Two conditions were tested:

| Condition           | Perturbation                               |
| ------------------- | ------------------------------------------ |
| `wrist_zero_every4` | Zero wrist image every fourth policy query |
| `wrist_zero_all`    | Zero wrist image at every policy query     |

Setup:

| Item                               | Value                  |
| ---------------------------------- | ---------------------- |
| Suite                              | `libero_goal`          |
| Tasks                              | 0, 1, 2                |
| Episodes per task per condition    | 5                      |
| Total measured robustness episodes | 30                     |
| Seed                               | 7                      |
| Baseline reference                 | Stage 4 clean baseline |

Results:

| Condition           | Task 0 | Task 1 | Task 2 | Aggregate |
| ------------------- | -----: | -----: | -----: | --------: |
| Stage 4 clean       |   9/10 |   9/10 |   9/10 |     27/30 |
| `wrist_zero_every4` |    0/5 |    0/5 |    4/5 |      4/15 |
| `wrist_zero_all`    |    0/5 |    0/5 |    0/5 |      0/15 |

Route and prefix behavior:

| Metric               | Stage 4 clean | `wrist_zero_every4` | `wrist_zero_all` |
| -------------------- | ------------: | ------------------: | ---------------: |
| Full ratio           |         0.242 |               0.322 |            0.388 |
| Accepted prefix mean |        10.074 |               7.929 |            6.550 |

The interpretation is clear:

1. Wrist-camera dropout strongly harmed this subset.
2. Full-route ratio increased under perturbation.
3. Accepted-prefix mean decreased under perturbation.
4. However, high-percentile accepted prefix remained saturated at 12.
5. Therefore, accepted prefix is only a partial uncertainty signal.

This is a negative result, and it is valuable. It shows that speculative execution logs can reveal some degradation, but they do not fully diagnose observation corruption.

The Stage 8 caveats explicitly preserve this interpretation: synthetic wrist-camera dropout caused strong degradation, and accepted prefix remained saturated at high percentiles under failure.

---

# **12. Stage 7: WristHealthGuard minimal extension**

Stage 7 implemented a small local inference-time extension called **[WristHealthGuard](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/patches/stage7_wrist_health_guard.patch){:target="_blank" rel="noopener noreferrer"}**.

## **12.1 Motivation**

Stage 5 suggested that accepted-prefix-only adaptation would not be the best first extension. The failure source was not primarily an action-prefix scheduling problem. It was a corrupted wrist observation problem.

Therefore, the extension targeted observation health directly.

## **12.2 Design**

WristHealthGuard uses a simple last-valid wrist-frame cache. The patch application script is [scripts/stage7_apply_wrist_health_guard_patch.py](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/scripts/stage7_apply_wrist_health_guard_patch.py){:target="_blank" rel="noopener noreferrer"}.

The anti-cheating order was:

1. Get simulator observation.
2. Apply Stage 5 dropout perturbation.
3. Compute wrist-image health on the perturbed image.
4. If unhealthy and a valid cached wrist image exists, replace the image with the cached copy.
5. If healthy, update the cache.
6. Send the final observation to the policy.

The guard never sees the clean pre-perturb image after dropout. The cache resets at the start of every episode.

The health metric was:

| Metric | Rule                      |
| ------ | ------------------------- |
| std    | healthy if `std >= 1.0`   |
| range  | healthy if `range >= 5.0` |

This is intentionally simple. The goal was not to invent a robust perception module, but to test whether a minimal observation-health heuristic can recover some synthetic intermittent dropout.

## **12.3 Results**

Setup:

| Condition                 |    Episodes |
| ------------------------- | ----------: |
| `guard_wrist_zero_every4` |          15 |
| `guard_wrist_zero_all`    |           9 |
| Sanity clean guard        | 1, excluded |

Success:

| Condition           | Stage 5 no guard | Stage 7 guard | Recovery |
| ------------------- | ---------------: | ------------: | -------: |
| `wrist_zero_every4` |             4/15 |          6/15 |   +0.133 |
| `wrist_zero_all`    |             0/15 |           0/9 |   +0.000 |

Guard behavior:

| Metric               | Every4 | All-zero |
| -------------------- | -----: | -------: |
| Cache hits           |     96 |        0 |
| Cache misses         |     15 |      433 |
| Cache updates        |    308 |        0 |
| Replacements         |     96 |        0 |
| Cache hit rate       |  0.229 |    0.000 |
| Healthy wrist rate   |  0.735 |    0.000 |
| Unhealthy wrist rate |  0.265 |    1.000 |

Interpretation:

* WristHealthGuard modestly improved intermittent dropout.
* It did not solve persistent all-zero dropout.
* This is expected because the cache resets per episode and all-zero dropout provides no valid frame to cache.
* The all-zero result is a useful sanity check against cheating: the guard did not secretly access clean pre-perturb frames.

The [Stage 8 claim matrix](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage8_claim_matrix.md){:target="_blank" rel="noopener noreferrer"} marks this as `partially_supported_limited`, not as a general robustness solution.

---

# **13. Failure mode taxonomy**

A major purpose of this project was to classify failures rather than simply report success rates.

| Failure mode                             | Stage           | Interpretation                              |
| ---------------------------------------- | --------------- | ------------------------------------------- |
| Checkpoint ambiguity                     | Stage 2         | resolved in Stage 2B                        |
| Server boot uncertainty                  | Stage 2B        | resolved by readiness smoke                 |
| Integration uncertainty                  | Stage 3         | resolved by 2/2 closed-loop smoke           |
| Horizon no-success                       | Stage 4         | policy behavior failure, not infrastructure |
| Wrist-camera perturbation failure        | Stage 5         | perturbation-induced perception failure     |
| Accepted-prefix saturation under failure | Stage 5/7       | accepted prefix is partial signal           |
| Persistent all-zero dropout              | Stage 7         | no valid cache, expected guard failure      |
| p99 latency tail                         | Stage 6/7       | residual first-use / tail behavior          |

The key shift across the project was that failures moved from infrastructure and artifact problems to actual closed-loop behavior problems. That is a good sign. It means the system was running deeply enough for policy-level and observation-level analysis to become possible.

---

# **14. Reproducibility and artifact policy**

The project used a strict public-safe artifact policy.

Tracked artifacts included:

* stage reports
* configs
* scripts
* patches
* lightweight logs
* parsed JSON/CSV summaries
* figures
* final claim matrix
* final project summary
* final blog draft

Excluded artifacts included:

* checkpoints
* converted weights
* videos
* datasets
* profiler binaries
* private endpoints
* credentials

The [Stage 8 artifact index](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage8_artifact_index.md){:target="_blank" rel="noopener noreferrer"} confirms that all required source artifacts were present and that local-only artifacts were excluded from Git.

The fixed reproducibility inputs were:

| Input           | Value                                                 |
| --------------- | ----------------------------------------------------- |
| Official repo   | `dexmal/realtime-vla-flash`                           |
| Official commit | `da6ceccad603695a8a3d6fa14dd410c3aadb536f`            |
| Project repo    | [Changhyun-Choi-98/realtime-vla-flash-runpod-project](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/tree/1b5d58319e7173b91b28a7809a3de12833fc0a04){:target="_blank" rel="noopener noreferrer"} |
| Hardware        | Runpod NVIDIA L40S                                    |
| Simulator       | LIBERO / MuJoCo with EGL                              |
| Main config     | `pi0_libero`                                          |
| Main suite      | `libero_goal`                                         |
| Main tasks      | 0, 1, 2                                               |
| Seed            | 7                                                     |

The reproducibility checklist also notes that re-running from scratch requires enough disk for the public base checkpoint and converted Triton artifacts.

---

# **15. Comparison to the paper**

The paper-level claim should be kept separate from my project-level result.

## **15.1 What the paper reports**

The paper and project page report:

* 58.0 ms full-inference rounds
* speculative rounds as fast as 7.8 ms
* 19.1 ms task-level average inference latency
* 3.04× speedup
* 0.3-point average success drop
* real-world conveyor-belt sorting demonstration ([arXiv][1]{:target="_blank" rel="noopener noreferrer"})

## **15.2 What this project measured**

My project measured:

* 27/30 success on LIBERO Goal tasks 0–2
* draft/full ratio 0.758 / 0.242 in Stage 4
* synchronized server-side `policy_time_gpu_sync_ms` p50 8.083 ms in Stage 6
* draft route p50 8.067 ms
* full route p50 33.156 ms
* client roundtrip p50 11.957 ms
* wrist-camera dropout degradation
* WristHealthGuard minimal extension behavior

## **15.3 Claim status**

The correct claim status is:

| Paper area                            | My status                                                    |
| ------------------------------------- | ------------------------------------------------------------ |
| Full benchmark success preservation   | not reproduced                                               |
| Latency mechanism                     | partially supported in limited setting                       |
| Flash path faster than full path      | supported in limited synchronized profiling                  |
| Accepted-prefix behavior              | supported as logged behavior, but partial uncertainty signal |
| Real-world conveyor result            | not attempted                                                |
| Robustness under wrist-camera dropout | project-added negative probe                                 |
| WristHealthGuard                      | project-added minimal extension                              |

Thus the final phrase should be:

> **Partially supported in a limited setting. Not a full paper reproduction.**

---

# **16. What this project does not claim**

This section is intentionally explicit.

This project does not claim:

1. Full LIBERO benchmark reproduction.
2. Full paper result reproduction.
3. Hardware-exact latency reproduction.
4. Real-world conveyor reproduction.
5. General robustness.
6. General sensor-dropout robustness.
7. That WristHealthGuard is an upstream-quality extension.
8. That accepted prefix is a reliable uncertainty estimator.
9. That the measured latency should be directly compared to the paper without task, hardware, and timing-method caveats.

The safest description is:

> **limited closed-loop reproduction + probing + profiling + minimal extension.**

---

# **17. Final takeaway**

Realtime-VLA FLASH was reproducible enough to become a strong portfolio-quality research-engineering project.

The strongest project-level result is not a single success-rate number. It is the full chain:

1. official repo
2. public checkpoint resolution
3. Triton conversion
4. server readiness
5. closed-loop LIBERO rollout
6. limited baseline
7. synchronized profiling
8. robustness probing
9. minimal extension
10. claim matrix and caveats

The clean limited baseline was strong: 27/30 success on LIBERO Goal tasks 0–2. The synchronized profiling supported the latency mechanism: draft route p50 was lower than full route p50 in this limited setting. The robustness probe revealed a concrete weakness: wrist-camera dropout severely reduced success. The minimal extension gave a modest but interpretable improvement under intermittent dropout and failed, as expected, under persistent all-zero dropout.

This is exactly the kind of result that is useful for robot foundation model research: not overclaimed, not merely a README run, and not just a failed attempt. It turns a paper into a controlled artifact trail.

---

# **18. Career takeaway**

From a Physical AI / Robot AI Model Researcher perspective, this project is valuable because it demonstrates more than implementation ability.

It demonstrates **claim discipline**.

Many reproduction projects fail because they jump from “the repo runs” to “the paper is reproduced.” This project did not do that. It separated:

* model environment readiness
* checkpoint provenance
* Triton conversion
* server readiness
* closed-loop integration
* limited baseline success
* synchronized latency profiling
* robustness probing
* local extension

The Stage 8 career takeaway summarizes the project well: the value is that I did not believe or reject the paper wholesale; I narrowed the problem stage by stage and left a reproducible artifact trail.

This is important for robot AI research because real systems fail at the boundaries: renderer, simulator, checkpoint, client/server protocol, action representation, timing measurement, observation corruption, and control-loop latency. A strong researcher should be able to identify which layer failed and avoid turning infrastructure success into algorithmic evidence.

In this project, I reached the point where the remaining failures were no longer environment failures. They were meaningful behavior failures: horizon no-success, perturbation-induced perception failure, accepted-prefix saturation, and persistent observation loss. That is the point where research begins.

---

# **Appendix A. Final metrics**

| Category           | Metric                        |     Value |
| ------------------ | ----------------------------- | --------: |
| Stage 4 baseline   | Task 0                        |      9/10 |
| Stage 4 baseline   | Task 1                        |      9/10 |
| Stage 4 baseline   | Task 2                        |      9/10 |
| Stage 4 baseline   | Aggregate                     |     27/30 |
| Stage 4 route      | Full                          |        88 |
| Stage 4 route      | Draft                         |       276 |
| Stage 4 route      | Draft ratio                   |     0.758 |
| Stage 4 route      | Full ratio                    |     0.242 |
| Stage 4 prefix     | Accepted prefix mean          |    10.074 |
| Stage 6 latency    | `policy_time_gpu_sync_ms` p50 |  8.083 ms |
| Stage 6 latency    | `policy_time_gpu_sync_ms` p95 | 33.501 ms |
| Stage 6 latency    | Full route p50                | 33.156 ms |
| Stage 6 latency    | Draft route p50               |  8.067 ms |
| Stage 6 latency    | Client roundtrip p50          | 11.957 ms |
| Stage 5 robustness | Clean Stage 4                 |     27/30 |
| Stage 5 robustness | `wrist_zero_every4`           |      4/15 |
| Stage 5 robustness | `wrist_zero_all`              |      0/15 |
| Stage 7 extension  | `guard_wrist_zero_every4`     |      6/15 |
| Stage 7 extension  | `guard_wrist_zero_all`        |       0/9 |
| Stage 7 extension  | Every4 recovery               |    +0.133 |
| Stage 7 extension  | All-zero recovery             |    +0.000 |

These metrics are also encoded in [results/stage8_final_metrics.json](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/results/stage8_final_metrics.json){:target="_blank" rel="noopener noreferrer"}.

---

# **Appendix B. Final claim matrix summary**

| Claim                               | Status                      | Blog wording                            | Caveat                 |
| ----------------------------------- | --------------------------- | --------------------------------------- | ---------------------- |
| C1 official repo/env readiness      | supported                   | model env could be prepared             | not evaluation         |
| C2 checkpoint/draft/base conversion | supported                   | public checkpoint conversion worked     | weights local-only     |
| C3 policy server boot               | supported                   | server readiness confirmed              | not rollout            |
| C4 one-task closed-loop smoke       | supported                   | end-to-end connection confirmed         | tiny smoke             |
| C5 limited LIBERO Goal baseline     | supported limited           | 27/30 success                           | tasks 0–2 only         |
| C6 latency mechanism                | partially supported limited | draft p50 < full p50                    | not hardware-exact     |
| C7 route / accepted prefix          | supported limited           | accepted prefix inspected               | partial signal         |
| C8 wrist dropout robustness         | negative limited            | wrist dropout was damaging              | synthetic only         |
| C9 WristHealthGuard                 | partially supported limited | intermittent dropout modestly recovered | not general robustness |
| C10 full paper benchmark            | not claimed                 | full benchmark not run                  | out of scope           |
| C11 real-world conveyor             | not claimed                 | real-world result outside scope         | no robot setup         |

The full matrix exists in both [Markdown](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/notes/stage8_claim_matrix.md){:target="_blank" rel="noopener noreferrer"} and [JSON](https://github.com/Changhyun-Choi-98/realtime-vla-flash-runpod-project/blob/1b5d58319e7173b91b28a7809a3de12833fc0a04/results/stage8_claim_matrix.json){:target="_blank" rel="noopener noreferrer"} form.



[1]: https://arxiv.org/abs/2605.13778 "[2605.13778] Realtime-VLA FLASH: Speculative Inference Framework for Diffusion-based VLAs"
[2]: https://dexmal.github.io/realtime-vla-flash/ "Realtime-VLA FLASH"
[3]: https://github.com/dexmal/realtime-vla-flash "GitHub - dexmal/realtime-vla-flash · GitHub"
[4]: https://github.com/dexmal/realtime-vla-flash/blob/main/pyproject.toml "realtime-vla-flash/pyproject.toml at main · dexmal/realtime-vla-flash · GitHub"
