---
layout: post
title: "CUDA Execution Flow in PyTorch"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-05-27
tags:
  - Korean
  - Python
  - PyTorch
  - CUDA
language: ko
summary: "PyTorch를 이용한 모델 학습 및 추론에서 CUDA가 어떻게 쓰이는지에 대한 예시"
math: true
comments: true
comment_id: "study-system-optimization-torch-cuda-example"
permalink: /study/system-optimization/torch-cuda-example/
---

[앞선 게시물에서](/study/system-optimization/cuda-triton/) CUDA와 Triton에 대해서 정리했다. 여기서는 실제 PyTorch training, inference 코드에서 CUDA가 어떻게 쓰이는지에 대한 예시를 정리한다.

## **Training**

아래 코드는 PyTorch 예제 코드이다:

```python
import torch
import torch.nn as nn

device = "cuda:0"

model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096),
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

x = torch.randn(32, 4096, device=device, dtype=torch.float16)
target = torch.randn(32, 4096, device=device, dtype=torch.float16)

for _ in range(100):
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = model(x)
        loss = ((y - target) ** 2).mean()

    loss.backward()
    optimizer.step()
```

위 코드는 CUDA execution flow를 설명하기 위한 단순 예제이다. 실제 FP16 mixed precision training에서는 gradient underflow를 줄이기 위해 보통 `torch.amp.GradScaler("cuda")`를 함께 사용한다. 여기서는 CUDA 흐름을 단순화하기 위해 생략했다.

위 코드가 내부에서 CUDA를 쓰는 순서를 살펴보면 다음과 같다.

### **1. Python process 시작**

```python
import torch
```

이 단계에서는 PyTorch Python package가 load된다. CUDA를 실제로 초기화하는 시점은 보통 첫 CUDA operation이 발생할 때이다. 예를 들어 `torch.cuda.is_available()`, `.to("cuda")`, `torch.randn(..., device="cuda")` 같은 호출이 trigger가 될 수 있다. 내부적으로는 다음 계층들이 필요해진다:

```text
Python torch package
  -> libtorch / libtorch_cuda
  -> CUDA runtime libraries
  -> NVIDIA driver library libcuda.so
  -> kernel driver
  -> GPU
```

### **2. Device 선택과 CUDA context 생성**

```python
device = "cuda:0"
model.to(device)
```

처음 CUDA device를 사용할 때 PyTorch는 CUDA runtime/driver를 통해 device를 확인하고, 해당 process에서 GPU를 사용할 때 CUDA context를 준비한다. CUDA context는 GPU 쪽의 "process-like execution state"와 같은 개념이다. Context에는 대략 다음 상태들이 묶인다:

* GPU virtual address space
* loaded modules/kernels
* memory allocations
* stream/event state
* library handles
* device-side execution state

### **3. Model parameter를 GPU memory로 이동**

```python
model.to(device)
```

위 call은 단순하게 "모델이 GPU로 간다"가 아니라, parameter와 buffer마다 아래 일이 일어난다:

```text
CPU tensor storage
  -> GPU memory allocation
  -> host-to-device copy
  -> PyTorch tensor metadata가 CUDA storage를 가리키도록 변경
```

다음 CUDA 기능들이 사용된다:

| 단계                 | CUDA 구성요소                                              |
| ------------------ | ------------------------------------------------------ |
| GPU memory 확보      | CUDA allocator → CUDA Runtime/Driver memory allocation |
| CPU→GPU 복사         | `cudaMemcpyAsync` 또는 이에 대응되는 copy enqueue              |
| stream dependency  | CUDA stream                                            |
| 실제 복사              | GPU copy engine 또는 DMA path                            |
| tensor metadata 갱신 | PyTorch C++ tensor/runtime layer                       |

단, PyTorch의 `.to("cuda")`가 항상 CPU 관점에서 완전히 non-blocking이라는 뜻은 아니다. DataLoader batch 전송처럼 H2D copy와 compute overlap을 노릴 때는 pinned memory와 `non_blocking=True`, stream dependency를 함께 고려해야 한다.

PyTorch는 매번 raw `cudaMalloc`/`cudaFree`를 호출하면 overhead가 크기 때문에 보통 caching allocator를 사용한다. 따라서 user 관점에서는 “tensor allocation”이지만 내부에서는 PyTorch allocator가 GPU memory block을 재사용한다.

### **4. Input tensor 생성**

```python
x = torch.randn(32, 4096, device=device, dtype=torch.float16)
```

위 한 줄에는 아래 CUDA operation들이 들어가있다:

1. CUDA device memory allocation
2. random number generation kernel 또는 random library path
3. dtype이 FP16이면 half precision storage
4. 생성 kernel launch가 CUDA stream에 enqueue
5. CPU는 기본적으로 kernel 완료를 기다리지 않고 다음 Python line으로 진행 가능

### **5. Forward pass: Linear layer**

```python
y = model(x)
```

첫 번째로 `nn.Linear(4096, 4096)`은 본질적으로 `Y = X W^T + b` GEMM(General Matrix-Matrix Multiplication) 또는 GEMM-like `addmm` 연산이다. (`[32 x 4096] @ [4096 x 4096] -> [32 x 4096]`)

이때 PyTorch는 보통 직접 작성한 단순 kernel을 쓰기보다, shape/dtype/backend 조건에 따라 cuBLAS 또는 cuBLASLt 기반 GEMM path를 사용한다. cuBLAS는 CUDA runtime 위에서 NVIDIA GPU의 computational resources를 사용하게 해주는 BLAS(Basic Linear Algebra Subprograms) 구현이고, cuBLASLt는 GEMM 중심의 더 유연한 API와 algorithm/heuristics 선택을 제공한다. 실제 흐름은 대략 아래와 같다:

```text
Python nn.Linear.forward
  -> PyTorch dispatcher
  -> ATen CUDA implementation
  -> cuBLAS/cuBLASLt GEMM or addmm path
  -> library heuristic/algorithm selection if applicable
  -> one or more CUDA kernel launches
  -> Tensor Core / CUDA core execution on SMs
  -> output tensor in GPU HBM
```

### **6. Forward pass: GELU and other memory-bound ops**

```python
nn.GELU()
```

GELU같은 elementwise 연산은 보통 memory-bound이다. 단순 GELU는 각 element들에 대해서 HBM에서 load하고, GELU 연산을 수행하고, 다시 HBM에 store한다. 따라서 연산량보다 HBM read/write가 중요하다. 그래서 performance optimization에서는 다음을 노린다:

```text
Linear output
  -> bias add
  -> activation
  -> dropout
  -> residual add
  -> layernorm
```

위 연산들을 각각 kernel로 launch하면 HBM을 여러 번 왕복한다:

```text
bad:
  kernel1: bias add      read/write HBM
  kernel2: GELU          read/write HBM
  kernel3: dropout       read/write HBM
  kernel4: residual add  read/write HBM

better:
  one fused kernel:
    load once
    do several operations in register
    store once
```

여기서 custom CUDA kernel이나 Triton kernel이 매우 중요해진다. PyTorch eager mode가 각각의 op를 따로 launch한다면 launch overhead와 memory traffic이 커질 수 있고, `torch.compile`/`TorchInductor`나 hand-written Triton/CUDA kernel이 이를 fuse할 수 있다.

이 예제에는 GELU만 등장하지만, 실제 Transformer/VLA 모델에서는 bias add, activation, dropout, residual add, layernorm/RMSNorm 같은 memory-bound 연산들이 연속해서 등장한다.

### **7. Forward pass: convolution이면 cuDNN**

CNN, Vision backbone, 일부 VLA perception module에서는 convolution이 나온다. 이때는 보통 cuDNN이 관여한다. 최신 cuDNN은 단순히 convolution kernel만 제공하는 수준을 넘어, operation graph, engine configuration, heuristic selection, execution plan, fusion 등을 지원한다. 따라서 framework와 연산 조건에 따라 cuDNN 내부에서 적절한 implementation이 선택될 수 있다. 흐름은 대략 다음과 같다:

```text
PyTorch Conv2d
  -> ATen CUDA convolution path
  -> cuDNN convolution / graph-capable DNN primitive path if selected
  -> heuristic or algorithm selection
  -> workspace allocation if needed
  -> cuDNN launches CUDA kernels
  -> output tensor
```

### **8. Loss 계산**

```python
loss = ((y - target) ** 2).mean()
```

위 계산에는 elementwise subtraction, square, reduction mean으로 이루어져 있기 때문에 내부적으로는 다음 과정들이 발생한다:

```text
sub kernel
square kernel
reduction kernel
possibly fused kernel if compiler/fusion path enabled
```

Reduction은 단순 elementwise보다 어렵다. 여러 thread가 부분합을 만들고, block-level reduction을 하고, 필요하면 multi-stage reduction을 수행한다. CUDA 관점에서는 다음이 중요하다:

CUDA 관점에서는 다음이 중요하다:

* coalesced global load
* shared memory reduction
* warp-level reduction
* atomic operation 여부
* block size
* occupancy
* memory bandwidth
* temporary tensor allocation

### **9. Backward pass**

```python
loss.backward()
```

Backward에서는 autograd graph를 역순으로 순회하면서 gradient kernel을 launch한다. Linear layer의 backward는 본질적으로 다시 GEMM이다. 따라서 backward에서도 cuBLAS/cuBLASLt GEMM이 매우 중요하다:

```text
loss.backward()
  -> PyTorch autograd engine
  -> for each op in reverse:
       call CUDA backward implementation
       call cuBLAS/cuDNN/custom kernels as needed
  -> gradients accumulated in parameter.grad tensors
```

딥러닝 training에서 forward/backward 시간의 큰 비율이 GEMM이면 Tensor Core utilization이 중요하고, normalization/activation/optimizer가 크면 memory bandwidth와 fusion이 중요하다.


### **10. Optimizer step**

```python
optimizer.step()
```

AdamW를 예로 들자면, parameter마다 다음과 같은 연산이 있다:

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad * grad
param = param - lr * m / (sqrt(v) + eps) - weight_decay * param
```

이건 대부분 elementwise 연산이다. parameter tensor가 많으면 작은 kernel launch가 많아질 수 있다. 그래서 고성능 training에서는 다음을 쓴다:

* fused AdamW kernel
* multi-tensor apply
* optimizer state sharding
* mixed precision optimizer
* CUDA Graph capture
* distributed optimizer

CUDA 관점에서는 optimizer가 compute-bound라기보다 memory bandwidth-bound인 경우가 많다. parameter, gradient, first moment, second moment를 모두 HBM에서 읽고 쓰기 때문이다.

### **11. Multi-GPU training이면 NCCL**

DDP(Distributed Data Parallel)를 쓰면 각 GPU가 local batch로 gradient를 계산한 뒤, GPU 간 gradient synchronization을 해야 한다.

```text
GPU0 grad
GPU1 grad
GPU2 grad
GPU3 grad
  -> all-reduce
  -> every GPU receives averaged/summed grad
```

이때 PyTorch DDP는 일반적으로 NCCL(NVIDIA Collective Communication Library)을 사용한다. [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html){:target="_blank" rel="noopener noreferrer"}은 topology-aware inter-GPU communication primitives를 제공하는 NVIDIA library이고, collective communication과 point-to-point send/receive primitives를 구현한다. 흐름은 다음과 같다:

```text
loss.backward()
  -> gradient bucket ready
  -> NCCL all-reduce launched on CUDA stream
  -> communication over NVLink / PCIe / InfiniBand
  -> overlap with remaining backward compute if possible
```

## **Inference**

Inference는 training과 다르게 다음이 없다:

* backward
* gradient accumulation
* optimizer update
* backward를 위한 activation 저장
* DDP gradient all-reduce

(inference에서도 다음 layer 계산을 위한 intermediate tensor와 workspace는 여전히 생성될 수 있다) 대신 다음의 항목들이 중요해진다:

* latency
* throughput
* batch size
* kernel launch overhead
* memory bandwidth
* KV cache management
* Tensor Core utilization
* static shape optimization
* CUDA Graph replay
* TensorRT engine optimization
* custom fused kernels

### **1. PyTorch eager inference**

```python
model.eval()

with torch.inference_mode():
    y = model(x)
```

위 흐름은 training forward와 거의 같다:

```text
Python model(x)
  -> PyTorch dispatcher
  -> ATen CUDA ops
  -> cuBLAS/cuDNN/custom kernels
  -> CUDA Runtime/Driver
  -> GPU execution
```

하지만 autograd graph를 만들지 않고 gradient tracking 관련 bookkeeping을 줄이기 때문에 memory usage와 CPU overhead가 줄어든다.

### **2. `torch.compile` inference**

```python
compiled_model = torch.compile(model)
y = compiled_model(x)
```
위 경우 PyTorch는 TorchDynamo를 통해 Python-level execution에서 graph를 추출하고, TorchInductor 같은 backend compiler를 통해 lower/codegen을 수행한다. 대략적인 흐름은 아래와 같다:

```text
Python model
  -> TorchDynamo graph capture
  -> AOTAutograd if training/backward involved
  -> TorchInductor lowering
  -> code generation
       ├── Triton kernels for many fused CUDA ops
       ├── calls to cuBLAS/cuDNN/external kernels for library-friendly ops
       └── C++/other backend paths depending target
  -> compiled callable
  -> CUDA kernel/library execution
```

`torch.compile`이 모든 것을 Triton으로 바꾸는 것은 아니다. 큰 GEMM은 여전히 cuBLAS/cuBLASLt가 더 좋을 수 있고, convolution은 cuDNN이 더 좋을 수 있다. Triton은 주로 fusion, custom tiling, pointwise/reduction, non-standard layout, attention-like kernels에서 강력하다.

### **3. CUDA Graph inference**

고정 shape inference에서는 Python/CUDA launch overhead도 무시할 수 없다. 특히 robotics real-time inference처럼 batch가 작고 반복 주기가 짧으면 GPU compute보다 CPU launch overhead가 튀어나올 수 있다.

CUDA Graph는 반복되는 CUDA work sequence를 capture한 뒤에 replay하는 방식이다:

```text
warmup
  -> capture: H2D copy + kernels + D2H copy sequence
  -> instantiate graph
  -> replay graph each control cycle
```

PyTorch는 `torch.cuda.CUDAGraph`, `torch.cuda.graph` API를 제공한다. 다만 CUDA Graph를 안정적으로 쓰려면 보통 shape, control flow, memory address가 반복 실행 사이에서 고정되어야 한다. PyTorch에서는 static input/output buffer를 미리 할당해두고, 매 iteration마다 그 buffer에 새 데이터를 copy한 뒤 graph를 replay하는 패턴을 자주 사용한다.

### **4. TensorRT inference**

TensorRT는 CUDA Toolkit 자체는 아니지만, CUDA 위에서 돌아가는 NVIDIA inference optimization/runtime stack이다. 대략적인 흐름은 보통 다음과 같다:

```text
PyTorch model
  -> export to ONNX or other representation
  -> TensorRT builder
       ├── layer fusion
       ├── precision / quantization selection: FP32/FP16/BF16/INT8/FP8 depending on hardware and TensorRT version
       ├── tactic selection
       ├── memory planning
       └── engine build
  -> TensorRT runtime
  -> CUDA kernels / cuBLAS/cuDNN/custom TensorRT kernels
  -> GPU execution
```

참고로 OpenAI Triton과 NVIDIA Triton Inference Server는 다른 개념이다:

| 이름                                 | 정체                                                 |
| ---------------------------------- | -------------------------------------------------- |
| **OpenAI Triton / triton-lang**    | Python 기반 GPU kernel programming language/compiler |
| **NVIDIA Triton Inference Server** | 모델 serving server/runtime platform                 |

NVIDIA Triton Inference Server는 TensorRT, PyTorch, ONNX, OpenVINO, Python backend 등 여러 framework의 모델을 deploy/serve하기 위한 inference serving software이다.



{% include comments.html %}
