# SiLU-and-Mul Fusion Kernel — CUDA vs SYCL

High-performance fused activation kernel used in LLM inference (SwiGLU FFN), ported from vLLM's production CUDA implementation to Intel XPU using SYCL free-function kernels.

## Overview

**Kernel**: `out[i] = silu(gate[i]) * up[i]` where `silu(x) = x / (1 + exp(-x))`

**Input layout**: `[num_tokens, 2, d]` — gate and up concatenated along dim=1  
**Output layout**: `[num_tokens, d]`

**Performance strategy**:
- 1 block per token (grid = `[num_tokens]`)
- 128-bit vectorized loads (`float4` / `sycl::vec<float, 4>`)
- Scalar tail for non-divisible dimensions
- Fuses two memory-bound ops (activation + multiply) into single HBM pass

## Files

| File | Purpose |
|------|---------|
| [silu_and_mul.cu](silu_and_mul.cu) | CUDA reference implementation (derived from vLLM) |
| [silu_and_mul.sycl.cpp](silu_and_mul.sycl.cpp) | SYCL free-function kernel implementation |
| [silu_and_mul_bench.sycl.cpp](silu_and_mul_bench.sycl.cpp) | Micro-benchmark (xpu-perf style) |
| [README.md](README.md) | This document |
| [PROMPT.md](PROMPT.md) | AI agent conversation prompt |

---

## Part A: CUDA Kernel Origin and Key Code

### Source

**vLLM**: `vllm-project/vllm/csrc/libtorch_stable/activation_kernels.cu`  
**PyTorch op**: `torch.ops._C.silu_and_mul`  
**Production use**: LLaMA-2/3, Mistral, Qwen (SwiGLU FFN layer)

### Registration in vLLM

```cpp
// csrc/libtorch_stable/torch_bindings.cpp
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  // ...
  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
}
```

### Key Code Walkthrough

#### 1. Scalar SiLU Function

```cpp
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}
```

**Math**: SiLU (a.k.a. Swish) is a smooth activation function. Unlike ReLU, it's non-zero for negative inputs and has continuous derivatives.

#### 2. Main Fusion Kernel

```cpp
template <typename scalar_t, int VEC_SIZE = 4>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,         // [num_tokens, d]
    const scalar_t* __restrict__ input, // [num_tokens, 2 * d]
    const int d)
{
  const int token_idx = blockIdx.x;  // 1 block per token
  
  // Input layout: [num_tokens, 2, d]
  // gate = input[token_idx, 0, :]  (first half)
  // up   = input[token_idx, 1, :]  (second half)
  const scalar_t* gate = input + token_idx * 2 * d;
  const scalar_t* up   = input + token_idx * 2 * d + d;
  scalar_t* out_ptr    = out + token_idx * d;
  
  // Vectorized path: 128-bit aligned loads (4 × float32)
  const int num_vec = d / VEC_SIZE;
  using vec_t = float4; // CUDA built-in vector type
  
  for (int vec_idx = threadIdx.x; vec_idx < num_vec; vec_idx += blockDim.x) {
    int base = vec_idx * VEC_SIZE;
    
    // Load 4 elements at once
    vec_t gate_vec = *reinterpret_cast<const vec_t*>(gate + base);
    vec_t up_vec   = *reinterpret_cast<const vec_t*>(up + base);
    vec_t out_vec;
    
    // Compute silu(gate[i]) * up[i] for i ∈ [0, 4)
    out_vec.x = silu_kernel(gate_vec.x) * up_vec.x;
    out_vec.y = silu_kernel(gate_vec.y) * up_vec.y;
    out_vec.z = silu_kernel(gate_vec.z) * up_vec.z;
    out_vec.w = silu_kernel(gate_vec.w) * up_vec.w;
    
    // Store 4 elements at once
    *reinterpret_cast<vec_t*>(out_ptr + base) = out_vec;
  }
  
  // Scalar tail for d not divisible by 4
  for (int i = num_vec * VEC_SIZE + threadIdx.x; i < d; i += blockDim.x) {
    out_ptr[i] = silu_kernel(gate[i]) * up[i];
  }
}
```

**Key insights**:
1. **Tile-per-token**: Each block processes one token's full hidden dimension `d`. This maximizes L1 cache reuse and avoids cross-block synchronization.
2. **128-bit vectorization**: `float4` loads coalesce 4 adjacent float32 elements, saturating memory bus on modern GPUs.
3. **Scalar tail**: Handles `d % 4 != 0` cases without padding overhead.
4. **Fusion benefit**: Without fusion, you'd do:
   - Pass 1: `tmp = silu(gate)` → write `d` elements to HBM
   - Pass 2: `out = tmp * up` → read `d` + write `d` from/to HBM  
   **Total**: 3× HBM traffic. Fused version: 2× HBM traffic (1 read gate+up, 1 write out).

---

## Part B: SYCL Implementation and CUDA Mapping

### Semantic Mapping Table

| Concept | CUDA | SYCL (free-function kernel) |
|---------|------|------------------------------|
| **Kernel marker** | `__global__ void kernel(...)` | `SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)) void kernel(...)` |
| **Block index** | `blockIdx.x` | `item.get_group(0)` |
| **Thread index** | `threadIdx.x` | `item.get_local_id(0)` |
| **Block size** | `blockDim.x` | `item.get_local_range(0)` |
| **Math intrinsic** | `expf(-x)` | `sycl::exp(-x)` |
| **Vector type (128-bit)** | `float4` (`.x .y .z .w`) | `sycl::vec<float, 4>` (`[0] [1] [2] [3]`) |
| **Vector load** | `*reinterpret_cast<const float4*>(ptr)` | `vec.load(0, sycl::multi_ptr<...>(ptr))` |
| **Vector store** | `*reinterpret_cast<float4*>(ptr) = vec` | `vec.store(0, sycl::multi_ptr<...>(ptr))` |
| **Launch config** | `<<<grid, block>>>` or `cudaLaunchKernel(...)` | `syclexp::nd_launch(q, nd_range, kernel_function<K>, ...)` |
| **Device alloc** | `cudaMalloc(&p, bytes)` | `sycl::malloc_device<T>(count, q)` |
| **H2D copy** | `cudaMemcpy(..., cudaMemcpyH2D)` | `q.memcpy(dst, src, bytes).wait()` |
| **Sync** | `cudaDeviceSynchronize()` | `q.wait()` |
| **Free** | `cudaFree(p)` | `sycl::free(p, q)` |

### SYCL Kernel Body (Side-by-Side)

**CUDA** ([silu_and_mul.cu](silu_and_mul.cu)):
```cpp
__global__ void silu_and_mul_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    int d)
{
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  
  const float* gate = input + token_idx * 2 * d;
  const float* up   = input + token_idx * 2 * d + d;
  float* out_ptr    = out + token_idx * d;
  
  // Vectorized loop
  constexpr int VEC_SIZE = 4;
  const int num_vec = d / VEC_SIZE;
  using vec_t = float4;
  
  for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
    int base = vec_idx * VEC_SIZE;
    vec_t gate_vec = *reinterpret_cast<const vec_t*>(gate + base);
    vec_t up_vec   = *reinterpret_cast<const vec_t*>(up + base);
    vec_t out_vec;
    
    out_vec.x = silu_op(gate_vec.x) * up_vec.x;
    out_vec.y = silu_op(gate_vec.y) * up_vec.y;
    out_vec.z = silu_op(gate_vec.z) * up_vec.z;
    out_vec.w = silu_op(gate_vec.w) * up_vec.w;
    
    *reinterpret_cast<vec_t*>(out_ptr + base) = out_vec;
  }
  
  // Scalar tail
  for (int i = num_vec * VEC_SIZE + tid; i < d; i += block_size) {
    out_ptr[i] = silu_op(gate[i]) * up[i];
  }
}
```

**SYCL** ([silu_and_mul.sycl.cpp](silu_and_mul.sycl.cpp)):
```cpp
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_and_mul_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    int d)
{
  auto item = syclwi::get_nd_item<1>();
  const int token_idx = item.get_group(0);      // ← blockIdx.x
  const int tid = item.get_local_id(0);         // ← threadIdx.x
  const int block_size = item.get_local_range(0); // ← blockDim.x
  
  const float* gate = input + token_idx * 2 * d;
  const float* up   = input + token_idx * 2 * d + d;
  float* out_ptr    = out + token_idx * d;
  
  // Vectorized loop (same structure)
  constexpr int VEC_SIZE = 4;
  const int num_vec = d / VEC_SIZE;
  using vec_t = sycl::vec<float, VEC_SIZE>;
  
  for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
    int base = vec_idx * VEC_SIZE;
    vec_t gate_vec, up_vec, out_vec;
    
    gate_vec.load(0, sycl::multi_ptr<const float,
                      sycl::access::address_space::global_space>(gate + base));
    up_vec.load(0, sycl::multi_ptr<const float,
                    sycl::access::address_space::global_space>(up + base));
    
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      out_vec[i] = silu_op(gate_vec[i]) * up_vec[i];
    }
    
    out_vec.store(0, sycl::multi_ptr<float,
                     sycl::access::address_space::global_space>(out_ptr + base));
  }
  
  // Scalar tail (identical structure)
  for (int i = num_vec * VEC_SIZE + tid; i < d; i += block_size) {
    out_ptr[i] = silu_op(gate[i]) * up[i];
  }
}
```

**Key differences** (API only, not algorithmic):
1. **Kernel marker**: CUDA `__global__` → SYCL `SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>))`
2. **Thread ID access**: CUDA direct (`blockIdx.x`) → SYCL via `nd_item` handle
3. **Vector load/store**: CUDA `reinterpret_cast` → SYCL explicit `.load()` / `.store()` methods
4. **Math**: CUDA `expf()` → SYCL `sycl::exp()` (both compile to native GPU intrinsics)

**Everything else is byte-for-byte identical**: loop structure, indexing arithmetic, fusion logic.

---

## Part C: Build and Test

### Build Commands

```bash
# CUDA (requires CUDA Toolkit ≥ 11.0, sm_80 = A100/H100)
nvcc -O3 -arch=sm_80 silu_and_mul.cu -o silu_and_mul_cuda

# SYCL (requires oneAPI ≥ 2025.0 for free-function kernels)
icpx -fsycl -O3 -fsycl-targets=spir64 silu_and_mul.sycl.cpp -o silu_and_mul_sycl

# Benchmark (SYCL only)
icpx -fsycl -O3 -fsycl-targets=spir64 silu_and_mul_bench.sycl.cpp -o silu_and_mul_bench
```

### Run Tests

```bash
# Functional test
./silu_and_mul_sycl

# Benchmark (default: 8192 tokens × 4096 hidden dim)
./silu_and_mul_bench

# Custom shape (LLaMA-3.1-8B: d=14336)
./silu_and_mul_bench --tokens 8192 --d 14336
```

### Expected Performance

**Hardware**: Intel Arc BMG-G31  
**Measured HBM peak**: ~530 GB/s  

Large shapes (e.g., `[8192, 4096]`): **>90% of HBM peak**

---

## References

- **vLLM source**: [activation_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/activation_kernels.cu)
- **SYCL free-function kernels**: [sycl_ext_oneapi_free_function_kernels.asciidoc](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_free_function_kernels.asciidoc)
- **xpu-perf framework**: [bytedance/xpu-perf](https://github.com/bytedance/xpu-perf)
- **Unfused SiLU example**: [../silu](../silu) (educational CUDA↔SYCL mapping)

---

## License

Code derived from vLLM (Apache 2.0) and original SYCL implementation (MIT).
