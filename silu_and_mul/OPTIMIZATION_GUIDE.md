# CUDA → SYCL Kernel Optimization Guide
## Leveraging PyTorch ATen XPU Element-wise Patterns

**Context**: 从 NVIDIA CUDA kernel 转换到 Intel SYCL 时，第一步是实现语义等价（semantic equivalence），第二步是性能优化。PyTorch ATen XPU backend 提供了成熟的 element-wise kernel 优化模式可以复用。

**Source Analysis**: 
- PyTorch ATen XPU: `/home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/`
- 核心文件: `Loops.h`, `ActivationSiluKernels.cpp`, `ActivationGluKernels.cpp`

---

## Stage 1: CUDA → SYCL 语义等价转换 ✅ 已完成

**我们的 silu_and_mul kernel**:
```cpp
template <int VEC_SIZE>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>))
void silu_and_mul_kernel_vec(float* __restrict__ out, const float* __restrict__ input, int d) {
  auto item = syclwi::get_nd_item<1>();
  const int token_idx = item.get_group(0);         // CUDA: blockIdx.x
  const int tid = item.get_local_id(0);            // CUDA: threadIdx.x
  
  // Vectorized load/compute/store (VEC_SIZE=4, 128-bit)
  // ... 1-block-per-token parallelism
}
```

**Current Status**:
- ✅ Accuracy: 28/28 tests passed (max_diff < 1.91e-06)
- ✅ Semantic equivalence: 1:1 mapping with vLLM CUDA kernel
- ⚠️ Performance: 67.29 GB/s (12.7% HBM peak) vs vLLM CUDA 75% peak

**Gap**: 性能差距主要来自 kernel 架构（1-block-per-token），而不是代码翻译错误。

---

## Stage 2: PyTorch ATen 优化模式借鉴

### 核心发现：PyTorch ATen 的 Element-wise 优化框架

PyTorch ATen XPU 使用统一的 `gpu_kernel()` 抽象，内部自动应用以下优化：

#### 2.1 动态向量化选择 (`launch_vectorized_kernel`)

**ATen Pattern** ([Loops.h:393-421](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L393-421)):
```cpp
template <typename func_t, typename array_t, typename in_calc_t>
static inline void launch_vectorized_kernel(
    int64_t N, const func_t& f, array_t data, in_calc_t input_calc, int vec_size) {
  
  auto wg_sz = syclMaxWorkItemsPerSubSlice();  // ← Intel Xe 优化的 work-group size
  
  #define VEC_KER(vec_size) { \
    auto ker = VectorizedElementwiseKernel<vec_size, func_t, array_t, in_calc_t>(N, f, data, input_calc); \
    int64_t num_wg = ceil_div<int64_t>(N, wg_sz * vec_size); \
    sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker); \
  }
  
  switch (vec_size) {
    case 16: VEC_KER(16); break;  // ← 动态选择向量化大小
    case 8:  VEC_KER(8);  break;
    case 4:  VEC_KER(4);  break;
    case 2:  VEC_KER(2);  break;
    case 1:  /* fallback to unrolled */ break;
  }
}
```

**关键优化技术**:
1. **动态 vec_size 选择**: 根据数据类型和大小自动选择 1/2/4/8/16
2. **Intel Xe 特定 work-group size**: `syclMaxWorkItemsPerSubSlice()` 而不是固定值
3. **向量化约束检查**: `max_scalar_bytes * vec_size <= 16`（保证向量化有效）

**对比我们的实现**:
```cpp
// 我们当前的做法
constexpr int VEC_SIZE = 4;  // ← 固定值
int block_size = 512;        // ← 固定值

// ATen 的做法
auto wg_sz = syclMaxWorkItemsPerSubSlice();  // ← 动态查询硬件
int vec_size = compute_vec_size(...);         // ← 动态选择
```

#### 2.2 两路径设计：Vectorized Path + Tail Path

**ATen Pattern** ([Loops.h:104-132](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L104-132)):
```cpp
template <int vec_size, typename func_t, typename array_t, typename in_calc_t>
struct VectorizedElementwiseKernel {
  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid = item.get_local_id(0);
    int group_work_size = vec_size * grpsz;
    int remaining = numel_ - grpid * group_work_size;
    
    if (remaining < group_work_size) {
      // ← Tail path: unrolled scalar processing
      auto policy = at::native::memory::policies::unroll<vec_size, ...>(...);
      elementwise_kernel_helper<vec_size>(f_, policy);
    } else {
      // ← Fast path: vectorized processing
      auto policy = at::native::memory::policies::vectorized<vec_size, ...>(...);
      elementwise_kernel_helper<vec_size>(f_, policy);
    }
  }
};
```

**关键优化技术**:
1. **Fast path 判断**: `remaining < group_work_size` 区分完整向量块和尾部
2. **Policy-based 抽象**: 使用不同的 memory policy (vectorized vs unrolled)
3. **统一的 compute kernel**: `elementwise_kernel_helper<vec_size>` 处理两种情况

**对比我们的实现**:
```cpp
// 我们当前的做法
for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
  // ← vectorized path
}
for (int i = num_vec * VEC_SIZE + tid; i < d; i += block_size) {
  // ← scalar tail
}

// ATen 的做法
// ← 在 work-group 级别判断是否为尾部，使用不同的 policy
```

#### 2.3 GLU Kernel 作为参考案例

**ATen GLU Implementation** ([ActivationGluKernels.cpp:23-37](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/ActivationGluKernels.cpp#L23-37)):
```cpp
template <typename scalar_t>
struct GluFunctor {
  using opmath_t = at::opmath_type<scalar_t>;  // ← 使用 opmath_t 提高精度
  scalar_t operator()(scalar_t a_, scalar_t b_) const {
    const opmath_t a = a_;
    const opmath_t b = b_;
    const opmath_t one = opmath_t(1);
    const opmath_t sigmoid = one / (one + sycl::exp(-b));  // ← SiLU 的一半
    return a * sigmoid;  // ← GLU = a * sigmoid(b)
  }
};

void glu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half,
      iter.dtype(), "glu_xpu",
      [&] { gpu_kernel(iter, GluFunctor<scalar_t>()); });  // ← 使用 gpu_kernel 抽象
}
```

**SiLU-and-Mul 是 GLU 的变体**:
```
GLU:          out = a * sigmoid(b)
SiLU-and-Mul: out = silu(gate) * up = gate * sigmoid(gate) * up
                                    = gate * up * sigmoid(gate)
```

**关键优化技术**:
1. **opmath_t**: 使用更高精度的中间类型（FP16 → FP32, FP32 → FP32）
2. **TensorIterator**: 自动处理 broadcasting, striding, alignment
3. **gpu_kernel 抽象**: 隐藏了 vectorization, coalescing, grid launch 的复杂性

#### 2.4 Work-group Size 动态查询

**ATen Pattern**:
```cpp
auto wg_sz = syclMaxWorkItemsPerSubSlice();  // ← Intel Xe 硬件查询

// 而不是固定值
constexpr int BLOCK_SIZE = 512;  // ← 我们当前的做法
```

**Intel Xe 硬件参数** (BMG-G31):
- `syclMaxWorkItemsPerSubSlice()`: 返回硬件最优 work-group size
- `syclMaxWorkItemsPerTile()`: 返回整个 tile 的最大线程数
- Sub-group size: 16 (SIMD width)

---

## Stage 2 优化建议：逐步应用 ATen 模式

### 优化 1: 动态 Work-group Size（低成本，高收益）

**Current**:
```cpp
int block_size = 512;  // Fixed
```

**Optimized** (参考 ATen):
```cpp
int block_size = at::xpu::syclMaxWorkItemsPerSubSlice();
// 或者使用 SYCL API 直接查询:
int max_wg_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
```

**Expected Gain**: 1-3%（根据硬件自适应，避免 occupancy 问题）

### 优化 2: 增强的向量化策略（中等成本）

**Current**:
```cpp
constexpr int VEC_SIZE = 4;  // Fixed float4
```

**Optimized** (参考 ATen):
```cpp
// 根据 d 的大小和对齐动态选择
int vec_size = 4;  // default
if (d % 8 == 0 && sizeof(float) * 8 <= 16) {
  vec_size = 8;  // float8 for better bandwidth
} else if (d % 4 == 0) {
  vec_size = 4;
} else if (d % 2 == 0) {
  vec_size = 2;
} else {
  vec_size = 1;  // scalar fallback
}

// 使用 template dispatch
switch (vec_size) {
  case 8: launch_kernel<8>(...); break;
  case 4: launch_kernel<4>(...); break;
  case 2: launch_kernel<2>(...); break;
  case 1: launch_kernel<1>(...); break;
}
```

**Expected Gain**: 0-5%（取决于 d 的对齐，d=4096 时可能用 vec=8）

### 优化 3: Grid Launch 策略优化（高成本，潜在高收益）

**Current Grid Strategy**:
```
1 work-group per token (grid = num_tokens)
```

**ATen Grid Strategy**:
```cpp
int64_t total_work = num_tokens * d;  // ← 总工作量
int64_t wg_sz = syclMaxWorkItemsPerSubSlice();
int64_t num_wg = ceil_div(total_work, wg_sz * vec_size);  // ← 动态计算 work-groups
```

**Key Difference**:
- **Current**: 每个 token 一个 work-group → 小 batch 时 GPU 利用率低
- **ATen**: 根据总工作量动态分配 work-groups → 更好的负载均衡

**Tradeoff**:
- ✅ Pro: 更好的 GPU 利用率（特别是小 batch）
- ⚠️ Con: 需要重新设计 data access pattern（不再是简单的 1-block-per-token）

**Expected Gain**: 5-15%（主要改善小 batch 性能，大 batch 可能无差异）

### 优化 4: opmath_t 精度提升（零成本）

**Current**:
```cpp
inline float silu_op(float x) {
  return x / (1.0f + sycl::exp(-x));
}
```

**Optimized** (参考 ATen GLU):
```cpp
template <typename scalar_t>
struct SiluAndMulFunctor {
  using opmath_t = at::opmath_type<scalar_t>;  // FP16→FP32, BF16→FP32, FP32→FP32
  
  scalar_t operator()(scalar_t gate_, scalar_t up_) const {
    const opmath_t gate = gate_;
    const opmath_t up = up_;
    const opmath_t one = opmath_t(1);
    const opmath_t silu_gate = gate / (one + sycl::exp(-gate));
    return scalar_t(silu_gate * up);
  }
};
```

**Expected Gain**: 
- FP32: 无影响（opmath_t = float32）
- FP16/BF16: 显著提升精度，性能基本不变

### 优化 5: TensorIterator 集成（高成本，生产级）

**最终形态**: 完全集成到 PyTorch ATen XPU backend

```cpp
void silu_and_mul_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half,
      iter.dtype(), "silu_and_mul_xpu",
      [&] { 
        gpu_kernel(iter, SiluAndMulFunctor<scalar_t>());  // ← 自动优化
      });
}
```

**Benefits**:
- ✅ 自动 vectorization (vec_size 1/2/4/8/16)
- ✅ 自动 alignment 检查
- ✅ 自动 coalescing
- ✅ 支持 broadcasting, striding, non-contiguous tensors
- ✅ 支持 FP16/BF16/FP32/FP64

**Expected Gain**: 10-30%（依赖 gpu_kernel 的优化质量）

---

## 优化路线图

### Phase 1: 低成本优化（1-2 天）

1. ✅ **动态 work-group size** → ~2% gain
2. ✅ **opmath_t 模板化** → 支持 FP16/BF16
3. ✅ **向量化 dispatch** (vec_size 2/4/8) → ~3% gain

**Target**: 67 → 70 GB/s (13.2% peak)

### Phase 2: 中等成本优化（3-5 天）

4. ⚙️ **Grid launch 重设计** → ~10% gain (small batch)
5. ⚙️ **Memory coalescing 分析** → Intel VTune profiling
6. ⚙️ **SLM (Shared Local Memory) 使用** → reduce global mem access

**Target**: 70 → 85 GB/s (16% peak)

### Phase 3: 高成本优化（1-2 周）

7. 🔨 **TensorIterator 集成** → 复用 ATen 优化框架
8. 🔨 **Multi-token per work-group** → 减少 kernel launch overhead
9. 🔨 **Intel intrinsics / block2d** → 使用 Xe 特定指令

**Target**: 85 → 120+ GB/s (20-25% peak)

### Phase 4: 架构级优化（长期）

10. 🚀 **Operator fusion**: GEMM + SiLU-and-Mul → 消除中间 memory 访问
11. 🚀 **oneDNN integration**: 使用 Intel 优化的 primitives
12. 🚀 **Persistent scheduling**: 减少 kernel dispatch overhead

**Target**: 达到 vLLM CUDA 的 75% peak

---

## 立即可行的代码改进

### 改进 1: 动态 work-group size

```cpp
// File: torch_ext/silu_and_mul_xpu.cpp
torch::Tensor silu_and_mul_xpu(torch::Tensor input) {
  // ... validation ...
  
  c10::xpu::XPUStream xpu_stream = c10::xpu::getCurrentXPUStream(input.device().index());
  sycl::queue& q = xpu_stream.queue();
  
  // ← OLD: constexpr int block_size = 512;
  // ← NEW: Query hardware
  int block_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  // 或者使用 ATen helper: block_size = at::xpu::syclMaxWorkItemsPerSubSlice();
  
  // 限制在合理范围（避免寄存器压力）
  block_size = std::min(block_size, 512);
  
  constexpr int VEC_SIZE = 4;
  sycl::nd_range<1> ndr{num_tokens * block_size, block_size};
  // ... launch kernel ...
}
```

### 改进 2: 向量化 dispatch

```cpp
// 根据 d 的对齐选择最优向量化大小
int vec_size = 4;  // default
if (d % 8 == 0) {
  vec_size = 8;
} else if (d % 4 == 0) {
  vec_size = 4;
} else if (d % 2 == 0) {
  vec_size = 2;
} else {
  vec_size = 1;
}

// Template dispatch
switch (vec_size) {
  case 8:
    syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_and_mul_kernel_vec<8>>, out_ptr, in_ptr, d);
    break;
  case 4:
    syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_and_mul_kernel_vec<4>>, out_ptr, in_ptr, d);
    break;
  // ... case 2, 1
}
```

### 改进 3: opmath_t 模板化

```cpp
template <typename scalar_t>
inline auto silu_op(scalar_t x) {
  using opmath_t = typename std::conditional<
      sizeof(scalar_t) < sizeof(float), float, scalar_t>::type;
  
  opmath_t x_acc = static_cast<opmath_t>(x);
  return x_acc / (opmath_t(1) + sycl::exp(-x_acc));
}

// 在 kernel 中使用
template <int VEC_SIZE, typename scalar_t>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>))
void silu_and_mul_kernel_vec(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    int d) {
  // ... 使用 silu_op<scalar_t>
}
```

---

## 性能预期

| Optimization | Estimated Gain | Cumulative BW | % of Peak |
|--------------|----------------|---------------|-----------|
| **Baseline** | - | 67.29 GB/s | 12.7% |
| + Dynamic wg_size | +1-2% | 68.5 GB/s | 12.9% |
| + Vec dispatch | +2-3% | 70.5 GB/s | 13.3% |
| + opmath_t | 0% (FP32) | 70.5 GB/s | 13.3% |
| **Phase 1 Total** | **+3-5%** | **70.5 GB/s** | **13.3%** |
| + Grid redesign | +5-10% | 77 GB/s | 14.5% |
| + SLM usage | +5-8% | 84 GB/s | 15.8% |
| **Phase 2 Total** | **+15-25%** | **84 GB/s** | **15.8%** |
| + TensorIterator | +10-15% | 97 GB/s | 18.3% |
| + Intel intrinsics | +5-10% | 106 GB/s | 20% |
| **Phase 3 Total** | **+35-50%** | **~106 GB/s** | **~20%** |

**Note**: 突破 20% peak 需要架构级优化（operator fusion, persistent scheduling 等）。

---

## 参考文件

| File | Purpose |
|------|---------|
| [Loops.h](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h) | ATen XPU element-wise kernel 框架 |
| [ActivationSiluKernels.cpp](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/ActivationSiluKernels.cpp) | SiLU kernel 参考实现 |
| [ActivationGluKernels.cpp](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/ActivationGluKernels.cpp) | GLU kernel (与 SiLU-and-Mul 类似) |
| [MemoryAccess.h](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/MemoryAccess.h) | Memory policy 抽象 |
| [OffsetCalculator.h](file:///home/liangan1/torch-xpu-ops/src/ATen/native/xpu/sycl/OffsetCalculator.h) | Strided tensor offset 计算 |

---

## 总结

**Stage 1 (已完成)**: CUDA → SYCL 语义等价转换
- ✅ 1:1 语义映射
- ✅ 精度验证通过
- ⚠️ 性能只有 12.7% peak

**Stage 2 (推荐路径)**: 借鉴 PyTorch ATen XPU 优化模式
1. **短期 (1-2 天)**: 动态 wg_size + vec_size dispatch → +3-5% gain
2. **中期 (1 周)**: Grid redesign + SLM → +15-25% gain  
3. **长期 (2-4 周)**: TensorIterator 集成 + Intel intrinsics → +35-50% gain

**关键洞察**:
- PyTorch ATen 已经为 element-wise kernels 提供了成熟的优化框架
- 不需要从零开始，可以复用 `gpu_kernel()`, `TensorIterator`, memory policies
- GLU kernel 是 SiLU-and-Mul 的近亲，可以直接参考其实现

**下一步行动**: 从 Phase 1 的低成本优化开始，逐步迭代。
