---
layout: post
title: "NVIDIA cuPyNumeric vs CuPy"
nav_exclude: true
section: study
subcategory: system-optimization
date: 2026-05-30
tags:
  - Korean
  - Python
  - CUDA
language: ko
summary: "NumPy 코드를 Legate 기반으로 CPU/GPU/멀티노드에 확장 실행하는 cuPyNumeric과, GPU 메모리 위에서 직접 NumPy/SciPy-style 연산을 수행하는 CuPy의 차이 정리"
math: true
comments: true
comment_id: "study-system-optimization-cupynumeric"
permalink: /study/system-optimization/cupynumeric/
---

<aside class="series-preface" markdown="1">

NVIDIA cuPyNumeric을 먼저 소개하고, 그에 대한 비교로 CuPy를 소개하는 방식으로 내용을 전개한다. 우선 이름이 비슷한 **CuPy**와 **cuPyNumeric**을 구분하면 다음과 같다.

| 라이브러리 | 핵심 의미 |
| --- | --- |
| **NumPy** | CPU memory 위에서 동작하는 표준 Python array computing library |
| **CuPy** | GPU device memory 위에서 직접 동작하는 NumPy/SciPy-like GPU array library |
| **cuPyNumeric** | NumPy API를 Legate runtime 위에 구현해 CPU/GPU/멀티노드로 확장 실행하려는 distributed NumPy replacement |

즉, **CuPy는 사용자가 GPU array를 직접 다루는 방식**에 가깝고, **cuPyNumeric은 기존 NumPy-style 코드를 runtime이 병렬/분산 실행하도록 만드는 방식**에 가깝다.

</aside>

[NVIDIA cuPyNumeric](https://docs.nvidia.com/cupynumeric/latest/){:target="_blank" rel="noopener noreferrer"}은 느린 Python/NumPy 기반 데이터 전처리 코드를 C++로 완전히 다시 쓰기 전에, `import` 교체만으로 NumPy array 연산을 CPU/GPU/멀티노드로 병렬 실행해볼 수 있는 NVIDIA 라이브러리이다. 대규모 numerical computation, matrix operation, data analysis 같은 array-heavy 작업을 Legate runtime을 통해 CPU/GPU 자원에 나누어 실행할 수 있고, workload가 충분히 크면 single GPU를 넘어 multi-node multi-GPU 환경까지 확장할 수 있다.

## **기본 개념**

기본적으로 NumPy API를 [Legate framework](https://docs.nvidia.com/legate/latest/){:target="_blank" rel="noopener noreferrer"}(여러 대의 CPU와 GPU 클러스터에서 복잡한 코드를 분산 실행하고 가속화할 수 있도록 지원하는 엔비디아의 프로그래밍 프레임워크) 위에 구현한 라이브러리이며, single CPU → single GPU → multi-node multi-GPU 시스템까지 투명하게 확장하는 것을 목표로 한다(NumPy 코드를 거의 그대로 두고 가속/분산 실행). 즉, 기존 NumPy 코드가 아래와 같았다면:

```python
import numpy as np
```

cuPyNumeric에서는 보통 아래와 같이 바꾼다:

```python
import cupynumeric as np
```

## **효과: training pipeline의 bottleneck 해결**

GPU training pipeline에서 GPU가 느린 Python 전처리를 기다린다고 가정해보자. 예를 들어 training loop가 아래처럼 돌아간다고 하자:

```python
for batch in dataloader:
    x = load_raw_data(batch)          # CPU
    x = preprocess_with_numpy(x)      # NumPy/Python
    x = torch.tensor(x).cuda()        # GPU로 복사
    loss = model(x)                   # GPU training
```

여기서 `preprocess_with_numpy()`가 batch당 40ms 걸리고, GPU forward/backward가 15ms만 걸린다면 비싼 GPU는 계속 기다린다. 이 경우 GPU utilization은 낮고, end-to-end goodput도 낮다. cuPyNumeric은 이런 전처리가 Python object loop가 아니라 NumPy array 연산으로 표현되어 있을 때 유리하다. 예를 들면:

```python
x = (x - mean) / std
x = np.clip(x, -3, 3)
x = np.matmul(x, W)
x = np.fft.fft(x)
x = np.histogram(x)
```
위와 같은 array operation은 CPU 한 코어에서 순차적으로 처리하는 대신, Legate runtime을 통해 CPU/GPU에 task로 나뉘어 실행될 수 있다. Legate는 CPU, OpenMP, GPU task variant를 제공하며, task가 어떤 processor에서 실행될 수 있는지를 runtime이 판단해 배치한다.

## **drop-in NumPy replacement**

위에서 언급한 것과 같이, 코드 구조를 거의 유지하면서 기존 NumPy import만 바꾸면 된다. 다만 “drop-in”은 100% 완전 동일이라는 뜻은 아니다. [cuPyNumeric 문서](https://docs.nvidia.com/cupynumeric/latest/user/differences.html){:target="_blank" rel="noopener noreferrer"}는 일부 dtype, shape, view/copy behavior, reduction 결과의 floating-point order 차이, scalar 반환 방식, indexing behavior 차이 등을 명시한다.

## **내부적인 작동 방식**

cuPyNumeric을 이해하려면 NumPy API layer와 Legate runtime layer를 나눠 보면 좋다.

### **1. 사용자는 NumPy처럼 코드를 쓴다**

사용자 입장에서는 NumPy와 거의 같은 array API를 사용한다.

```python
import cupynumeric as np

x = np.random.randn(100_000_000)
y = np.sin(x) + np.cos(x)
z = y.sum()
```

### **2. cuPyNumeric이 NumPy 연산을 Legate task로 바꾼다**

`np.sin`, `np.cos`, `+`, `sum` 같은 연산은 내부적으로 cuPyNumeric operation이 되고, Legate runtime이 이를 task graph처럼 관리한다. Legate program의 주요 실행 단위는 task이며, task는 runtime에 제출되어 실행된다. 또한 leaf task는 top-level program과 비동기적으로 실행될 수 있고, prerequisite이 만족되어야 실행된다.

### **3. Runtime이 CPU/GPU에 작업을 배치한다**

Legate task는 CPU variant, OpenMP variant, GPU variant를 가질 수 있다. GPU variant가 있으면 runtime은 GPU kernel launch나 GPU-accelerated library 호출을 의도한 task로 판단할 수 있고, 가능하면 더 accelerated된 variant를 선호한다.

### **4. 큰 array는 partitioning되어 병렬 처리된다**

대규모 array가 있으면 runtime은 이를 여러 chunk로 나누고, GPU 여러 개 또는 노드 여러 개에 나눠 처리할 수 있다. `legate --nodes 2 script.py` 형태로 multi-node execution을 수행할 수 있고, 2개 이상 노드에서는 `srun`, `mpirun`, `jsrun` 같은 launcher를 지정해야 한다.

## **C++과의 비교**

Python 전처리가 병목이면 C++로 다시 구현하거나 cuPyNumeric을 쓸 수 있다. 둘의 차이는 다음과 같다:

| 접근                 | 장점                                                   | 단점                                                          |
| ------------------ | ---------------------------------------------------- | ----------------------------------------------------------- |
| **C++/CUDA로 재작성**  | 최고 수준의 control, 커스텀 최적화 가능                           | 개발 비용 큼, 유지보수 어려움, CUDA/메모리/스레딩 지식 필요                       |
| **cuPyNumeric 적용** | import 교체 수준으로 빠르게 시도 가능, multi-GPU/multi-node 확장 가능 | NumPy API coverage와 workload 특성에 의존, 모든 Python 병목을 해결하지는 않음 |

Performance engineering 관점에서는 보통 다음과 같이 접근한다:

1. **Profiler로 병목 확인**
  * PyTorch Profiler, Nsight Systems, `cProfile`, `line_profiler` 등으로 GPU가 CPU preprocessing을 기다리는지 확인한다.
2. **병목 코드가 NumPy array 연산인지 확인**
  * `np.matmul`, `np.sum`, `np.fft`, `np.linalg`, broadcasting, elementwise op 중심이면 cuPyNumeric 후보가 된다.
3. **`import cupynumeric as np`로 빠르게 A/B test**
  * 큰 코드 변경 없이 latency, throughput, GPU utilization을 비교한다.
4. **성능이 충분하면 유지**
  * C++/CUDA rewrite를 피하고 생산성을 확보한다.
5. **성능이 부족하거나 unsupported API가 많으면 C++/CUDA/Triton/DALI/PyTorch op로 이동**
  * cuPyNumeric은 강력하지만 모든 병목에 대한 silver bullet은 아니다.


## **실행 예시**

### **단일 머신에서 기본 실행**

```shell
python main.py
```

이 방식은 기본적으로 현재 머신의 available hardware resource를 사용한다.

### **GPU 수를 명시해서 실행**

```shell
legate --gpus 2 main.py
```

### **멀티노드 실행**

```shell
legate --launcher srun --nodes 2 main.py
```

## **cuPyNumeric이 잘 맞는 경우**

| 잘 맞는 경우                                                  | 이유                                  |
| -------------------------------------------------------- | ----------------------------------- |
| 큰 NumPy array를 반복적으로 처리                                  | GPU/CPU 병렬화 overhead를 amortize하기 좋음 |
| elementwise, reduction, matmul, linalg, FFT, histogram 류 | array operation으로 잘 표현됨             |
| 단일 GPU 메모리보다 큰 데이터 처리                                    | Legate 기반 분산 실행 가능성                 |
| 기존 NumPy 코드가 많음                                          | migration cost가 낮음                  |
| C++/CUDA rewrite 전에 빠르게 실험하고 싶음                          | import 교체만으로 A/B test 가능            |


[공식 best practice 문서](https://docs.nvidia.com/cupynumeric/latest/user/practices.html){:target="_blank" rel="noopener noreferrer"}도 cuPyNumeric에서는 array-based computation을 권장하고, 각 API가 single CPU, multi-GPU/multi-CPU 등 어디에서 사용 가능한지 docstring에 표시된다고 설명한다. 또한 `CUPYNUMERIC_DOCTOR=1`로 common usage issue를 진단하는 doctor mode도 제공한다.

## **cuPyNumeric이 잘 안 맞는 경우**

| 잘 안 맞는 경우                                       | 이유                                                              |
| ----------------------------------------------- | --------------------------------------------------------------- |
| 작은 array를 아주 조금만 처리                             | GPU 초기화, scheduling overhead가 이득보다 클 수 있음                       |
| Python list/object/string 위주 전처리                | NumPy array 연산이 아니므로 cuPyNumeric의 장점이 작음                        |
| 이미지 decode, video decode, augmentation pipeline | NVIDIA DALI, nvJPEG, torchvision/PyTorch pipeline이 더 적합할 수 있음   |
| PyTorch tensor 연산이 이미 대부분인 경우                   | `torch.compile`, CUDA Graphs, Triton, fused kernel이 더 직접적일 수 있음 |
| unsupported NumPy API가 많음                       | base NumPy fallback으로 오히려 느려질 수 있음                              |

특히 GPU 초기화, memory allocation, kernel compilation 같은 one-time setup cost가 작은 array benchmark를 왜곡할 수 있으므로, 큰 input과 warm-up pass를 사용해야 한다.

## **CuPy와 cuPyNumeric 구분**

이름이 비슷해서 헷갈리기 쉽지만, **CuPy**와 **cuPyNumeric**은 같은 라이브러리가 아니다. 둘 다 NumPy와 비슷한 API를 제공하지만, 목표와 실행 모델이 다르다.

한 문장으로 구분하면:

> **CuPy는 GPU 메모리 위에서 직접 동작하는 NumPy/SciPy-like array library이고, cuPyNumeric은 NumPy 코드를 Legate runtime 위에서 CPU/GPU/멀티노드로 확장하려는 distributed NumPy replacement이다.**

### **CuPy란?**

[CuPy](https://docs.cupy.dev/en/latest/){:target="_blank" rel="noopener noreferrer"}는 Python에서 GPU array computation을 하기 위한 NumPy/SciPy-compatible 라이브러리이다. NumPy가 `numpy.ndarray`를 CPU 메모리에서 다룬다면, CuPy는 `cupy.ndarray`를 GPU device memory에서 다룬다. 주의할 점은 **CuPy가 cuPyNumeric의 하위 구성요소가 아니라는 것**이다. 이름은 비슷하지만, CuPy와 cuPyNumeric은 독립적인 프로젝트이며 사용 목적도 다르다.

기본 사용 방식은 NumPy와 매우 비슷하다:

```python
import numpy as np

x = np.arange(10)
y = x * 2
```

CuPy에서는 보통 아래처럼 쓴다:

```python
import cupy as cp

x = cp.arange(10)
y = x * 2
```

이때 `x`와 `y`는 CPU memory에 있는 NumPy array가 아니라, GPU memory에 있는 CuPy array이다. 따라서 `x * 2`, `cp.sin(x)`, `cp.matmul(a, b)` 같은 연산은 가능하면 GPU에서 실행된다.

즉, CuPy의 핵심은 다음과 같다:

| 항목 | 설명 |
| --- | --- |
| **핵심 객체** | `cupy.ndarray` |
| **실행 위치** | 주로 GPU |
| **API 스타일** | NumPy/SciPy와 유사 |
| **주요 backend** | CUDA, 일부 환경에서는 ROCm |
| **주요 용도** | GPU에서 array 연산, FFT, linear algebra, sparse matrix, custom CUDA kernel 실행 |
| **대표 import** | `import cupy as cp` |

CuPy는 내부적으로 CUDA Toolkit ecosystem을 적극적으로 사용한다. 예를 들어 matrix multiplication은 cuBLAS, FFT는 cuFFT, sparse 연산은 cuSPARSE, random number generation은 cuRAND 같은 CUDA library를 활용할 수 있다.

### **CuPy가 특히 유용한 경우**

CuPy는 다음과 같은 경우에 잘 맞는다.

| 잘 맞는 경우 | 이유 |
| --- | --- |
| 큰 array에 같은 연산을 반복 적용 | GPU 병렬성이 잘 살아남 |
| NumPy/SciPy 기반 수치 계산 코드를 GPU로 옮기고 싶음 | API가 NumPy/SciPy와 유사함 |
| FFT, linear algebra, sparse operation이 많음 | CUDA optimized library를 활용 가능 |
| Python에서 간단한 custom CUDA kernel을 쓰고 싶음 | `RawKernel`, `ElementwiseKernel`, `ReductionKernel` 등을 제공 |
| PyTorch/JAX와 GPU tensor를 주고받고 싶음 | DLPack 기반 zero-copy 교환이 가능 |

예를 들어 PyTorch tensor를 CuPy array로 바꿔 custom kernel을 적용하고 다시 PyTorch tensor로 넘기는 식의 workflow도 가능하다:

```python
import torch
import cupy as cp

x_torch = torch.randn(1024, device="cuda")

# PyTorch tensor -> CuPy array
x_cupy = cp.from_dlpack(x_torch)

# CuPy operation
y_cupy = cp.sin(x_cupy)

# CuPy array -> PyTorch tensor
y_torch = torch.from_dlpack(y_cupy)
```

이런 방식은 PyTorch 모델 중간에 직접 CUDA kernel을 넣고 싶을 때 유용하다. 다만 DLPack zero-copy 교환은 source와 destination이 같은 device memory를 참조할 수 있을 때 의미가 있다. 중간에 CPU NumPy array로 변환하거나 device가 달라지면 copy 비용이 다시 발생할 수 있으므로, 실제 training loop에서는 Nsight Systems나 PyTorch Profiler로 boundary cost를 확인해야 한다.

### **CuPy의 custom kernel 기능**

CuPy는 단순히 NumPy-like operation만 제공하는 것이 아니다. Python 코드 안에서 CUDA kernel을 정의하고 실행할 수도 있다.

대표적인 방식은 다음과 같다:

| 방식 | 설명 |
| --- | --- |
| `ElementwiseKernel` | elementwise CUDA kernel을 간단히 정의 |
| `ReductionKernel` | reduction 연산을 위한 kernel 정의 |
| `RawKernel` | CUDA C/C++ kernel source를 문자열로 작성해 직접 실행 |
| kernel fusion | 여러 CuPy operation을 하나의 kernel로 합쳐 launch overhead와 memory traffic 감소 |

예를 들어 `RawKernel`을 사용하면 Python 파일 안에 CUDA C kernel을 직접 작성할 수 있다:

```python
import cupy as cp

kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void double_kernel(const float* x, float* y, int n) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < n) {
            y[idx] = 2.0f * x[idx];
        }
    }
    ''',
    'double_kernel'
)

n = 1024
x = cp.ones(n, dtype=cp.float32)
y = cp.empty_like(x)

threads = 256
blocks = (n + threads - 1) // threads

kernel((blocks,), (threads,), (x, y, cp.int32(n)))
```

이런 점에서 CuPy는 단순한 “NumPy replacement”라기보다, Python에서 CUDA programming에 접근하는 비교적 쉬운 gateway라고 볼 수 있다.

### **CuPy가 잘 안 맞는 경우**

CuPy도 silver bullet은 아니다.

| 잘 안 맞는 경우 | 이유 |
| --- | --- |
| array 크기가 작음 | GPU kernel launch overhead가 더 클 수 있음 |
| CPU와 GPU 사이를 자주 왕복함 | `cp.asnumpy`, `cp.asarray`에서 copy overhead 발생 |
| Python object/list/string 처리 위주 | GPU array operation으로 표현되지 않음 |
| PyTorch tensor operation이 이미 대부분임 | `torch.compile`, Triton, PyTorch custom op가 더 자연스러울 수 있음 |
| multi-node distributed NumPy scaling이 목표임 | 이 경우는 cuPyNumeric/Legate 쪽이 더 직접적인 후보 |

특히 AI training pipeline에서는 CuPy를 넣는 위치가 중요하다. 예를 들어 preprocessing은 CuPy로 했는데, 매 batch마다 다시 CPU NumPy array로 변환한 뒤 PyTorch tensor로 바꾸면 성능 이득이 사라질 수 있다. 따라서 CuPy를 사용할 때는 가능한 한 GPU memory 위에서 preprocessing → tensor conversion → model input까지 이어지도록 설계해야 한다.

### **cuPyNumeric과 CuPy의 핵심 차이**

두 라이브러리는 모두 NumPy와 비슷한 API를 제공하지만, 지향점이 다르다.

| 항목 | CuPy | cuPyNumeric |
| --- | --- | --- |
| **대표 import** | `import cupy as cp` | `import cupynumeric as np` |
| **핵심 목표** | GPU에서 NumPy/SciPy-like array 연산 수행 | 기존 NumPy 코드를 CPU/GPU/멀티노드로 투명하게 확장 |
| **실행 모델** | 사용자가 GPU array를 직접 다룸 | Legate runtime이 task와 data partitioning을 관리 |
| **주요 객체** | `cupy.ndarray` | `cupynumeric.ndarray` |
| **주요 대상** | single-node/single-GPU 또는 명시적 GPU array workflow | large array computation, multi-GPU, multi-node scaling |
| **분산 실행** | `cupyx.distributed` 등 일부 기능 존재 | Legate 기반 분산 실행이 핵심 설계 목표 |
| **custom CUDA kernel** | `RawKernel`, `ElementwiseKernel`, `ReductionKernel` 등 제공 | 주 목적은 NumPy API의 분산/가속 실행 |
| **PyTorch 연동** | DLPack을 통한 GPU tensor 교환이 자연스러움 | PyTorch training pipeline과 연결할 때는 boundary cost를 별도로 확인해야 함 |
| **사용 감각** | “NumPy를 GPU array library처럼 사용” | “NumPy 코드를 runtime이 알아서 병렬/분산 실행” |

CuPy는 사용자가 명시적으로 GPU array를 다룬다. `cp.asarray()`로 GPU에 올리고, `cp.asnumpy()`로 CPU에 내린다. 반면 cuPyNumeric은 기존 NumPy-style 코드를 유지하면서 Legate runtime이 array operation을 task 단위로 나누고, CPU/GPU/multi-node resource에 배치하는 방향에 가깝다.

### **언제 CuPy를 쓰고, 언제 cuPyNumeric을 쓸까?**

간단한 선택 기준은 다음과 같다.

| 상황 | 더 먼저 볼 후보 |
| --- | --- |
| 기존 NumPy 코드를 단일 GPU에서 빠르게 돌리고 싶다 | CuPy |
| PyTorch tensor와 GPU array를 섞어 쓰고 싶다 | CuPy |
| Python에서 간단한 CUDA kernel을 직접 쓰고 싶다 | CuPy |
| NumPy 기반 대규모 simulation/data processing 코드를 여러 GPU/노드로 확장하고 싶다 | cuPyNumeric |
| 기존 NumPy 코드를 최대한 덜 고치고 CPU/GPU/multi-node scaling을 실험하고 싶다 | cuPyNumeric |
| 이미지/video decode/augmentation pipeline이 병목이다 | NVIDIA DALI, torchvision, custom CUDA/Triton 등을 먼저 검토 |
| PyTorch model 내부 tensor operation이 병목이다 | `torch.compile`, Triton, custom CUDA op 등을 먼저 검토 |

### **결론**

CuPy와 cuPyNumeric은 둘 다 “NumPy-like”라는 공통점이 있지만, 실제 사용 감각은 꽤 다르다.

**CuPy**는 GPU memory 위의 array를 직접 다루는 라이브러리이다. `numpy` 대신 `cupy`를 쓰고, `np.ndarray` 대신 `cp.ndarray`를 사용한다. GPU에서 연산이 수행되므로 큰 array operation, FFT, linear algebra, custom CUDA kernel에 적합하다.

반면 **cuPyNumeric**은 NumPy API를 Legate runtime 위에 구현한 라이브러리이다. 목표는 기존 NumPy 코드를 크게 바꾸지 않고 single CPU, single GPU, multi-GPU, multi-node 환경으로 확장하는 것이다. 따라서 단순히 “GPU용 NumPy”라기보다는 “distributed NumPy runtime”에 가깝다.

AI Systems Performance Engineering 관점에서 중요한 것은 **무엇이 병목인지 먼저 profile로 확인하는 것**이다. 병목이 큰 NumPy-style array 연산이면 CuPy나 cuPyNumeric이 후보가 될 수 있다. 병목이 PyTorch graph 내부라면 `torch.compile`이나 Triton이 더 직접적일 수 있고, 병목이 image/video decoding이면 DALI가 더 적합할 수 있다. 결국 도구 선택은 “어떤 layer에서 병목이 발생하는가”에 따라 달라진다.









{% include comments.html %}
