# SiLU — CUDA vs SYCL, three side-by-side demos

Goal: show a customer that porting a CUDA kernel to SYCL is a near-mechanical
translation. SiLU (a.k.a. Swish): `y = x * sigmoid(x) = x / (1 + exp(-x))`.

Four layers, ordered to match the three-step narrative requested by customers:

| # | Folder / file | Step | Audience | What it demonstrates |
| - | ------------- | ---- | -------- | -------------------- |
| 1 | `silu.cu` + `silu.sycl.cpp` (this folder) | **Step 1 — concept mapping** | Engineer skimming for 30 s | A bare free-function kernel + host driver. Kernel body is byte-for-byte identical; only `__global__` ↔ `SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(nd_range_kernel<1>)` and the launch glue change. |
| 2 | [torch_ext/](torch_ext/) | (extension example) | Engineer who writes custom ops | The same kernel wrapped as a PyTorch C++ extension — `AT_DISPATCH_*`, `data_ptr<T>()`, stream acquisition. `at::cuda::*` ↔ `c10::xpu::*` is the only diff. |
| 3 | [upstream/](upstream/) | **Step 2 — analyze upstream** | Architect / decision-maker | The actual upstream code that ships in PyTorch + torch-xpu-ops, placed side-by-side. Same `silu_stub`, same `gpu_kernel(iter, op)`, same `opmath_t` promotion. Plus a deeper concept mapping inside `gpu_kernel` (CUDA `vectorized_elementwise_kernel` ↔ SYCL `VectorizedElementwiseKernel`). |
| 4 | [vectorized/](vectorized/) | **Step 3 — standalone vectorized** | CUDA engineer porting a real kernel | A self-contained vectorized SYCL kernel, structurally derived from PyTorch CUDA's `vectorized_elementwise_kernel` but with no PyTorch dependency. `float4` ↔ `sycl::vec<float, 4>`, same tile-per-block + scalar-tail shape. Compiles in one line. |

The three-step pitch:

1. **Step 1 (this folder).** Show the concept mapping with the simplest
   possible CUDA and SYCL kernel side by side.
2. **Step 2 (upstream/).** Show that the same mapping holds for the real
   production code in PyTorch + torch-xpu-ops.
3. **Step 3 (vectorized/).** Show that a CUDA engineer can take PyTorch's
   `vectorized_elementwise_kernel` pattern and rewrite it in SYCL in one
   file, no PyTorch needed.

## 1. The kernel — concept mapping

| Concept                  | CUDA                                             | SYCL (free-function kernel)                                                  |
| ------------------------ | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| Kernel marker            | `__global__ void silu_kernel(...)`               | `SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)) void silu_kernel(...)` |
| Global thread index      | `blockIdx.x * blockDim.x + threadIdx.x`          | `this_work_item::get_nd_item<1>().get_global_id(0)`                          |
| Block / work-group size  | `blockDim.x`                                     | `it.get_local_range(0)`                                                      |
| Grid / num work-groups   | `gridDim.x`                                      | `it.get_group_range(0)`                                                      |
| Math intrinsic           | `expf(-v)`                                       | `sycl::exp(-v)`                                                              |
| `__restrict__` / fast    | `__restrict__`, `__ldg`                          | `__restrict__` allowed; `sycl::ext::oneapi::experimental::cache_control`     |

## 2. The kernel body — literally identical

CUDA ([silu.cu](silu.cu)):
```cpp
__global__ void silu_kernel(const float* __restrict__ x,
                            float*       __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}
```

SYCL ([silu.sycl.cpp](silu.sycl.cpp)):
```cpp
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void silu_kernel(const float* x, float* y, int n) {
    auto it = syclex::this_work_item::get_nd_item<1>();
    int i = it.get_global_id(0);
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + sycl::exp(-v));
    }
}
```

The three lines inside the `if` are byte-for-byte the same.

## 3. Host-side launch — concept mapping

| Concept              | CUDA                                  | SYCL                                                     |
| -------------------- | ------------------------------------- | -------------------------------------------------------- |
| Device alloc         | `cudaMalloc(&p, bytes)`               | `sycl::malloc_device<T>(N, q)`                           |
| H2D / D2H copy       | `cudaMemcpy(...)`                     | `q.memcpy(...).wait()`                                   |
| Launch config        | `<<<grid, block>>>`                   | `sycl::nd_range<1>{grid*block, block}`                   |
| Launch                | `silu_kernel<<<grid, BLOCK>>>(d_x, d_y, N);` &nbsp;or&nbsp; `cudaLaunchKernel((const void*)silu_kernel, dim3(grid), dim3(BLOCK), args, 0, 0)` | `syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_kernel>, d_x, d_y, N)` |
| Sync                 | `cudaDeviceSynchronize()`             | `q.wait()`                                               |
| Free                 | `cudaFree(p)`                         | `sycl::free(p, q)`                                       |

## 4. Build & run

```bash
# CUDA
nvcc -O3 -arch=sm_80 silu.cu -o silu_cuda && ./silu_cuda

# SYCL (Intel GPU, oneAPI ≥ 2025.0 for free-function kernels)
icpx -fsycl -O3 -fsycl-targets=spir64 silu.sycl.cpp -o silu_sycl && ./silu_sycl
```

Expected output (both):
```
SiLU[12345] = 0.xxxxxx   (ref 0.xxxxxx)
```

## 5. What's the same, what's different

**Same** (the message to customers):
- Programming model — SPMD over an nd-range / grid of blocks.
- Index arithmetic, control flow, math intrinsics.
- Memory model — explicit device alloc + H2D/D2H copy + sync.
- The kernel itself is a plain free function.

**Different** (and intentional):
- CUDA exposes *two* spellings for kernel launch: the language-extension
  sugar `kernel<<<grid, block>>>(args)` and the explicit C runtime API
  [`cudaLaunchKernel(funcAddr, gridDim, blockDim, args, smem, stream)`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c) —
  the former is sugar that nvcc lowers to the latter. SYCL's
  `syclexp::nd_launch` is the semantically equivalent API on the SYCL side:
  a plain C++ function call taking `(queue, nd_range, kernel address, arg
  pack)`. The three forms differ only in syntactic surface; SYCL stays a
  pure C++ library by virtue of *not* needing a language extension.
- µarch tuning differs: Intel Xe2 has DPAS systolic arrays, 2D block loads,
  XeSS, and an LSC/L1/L2 hierarchy distinct from NVIDIA SM/SMEM. For SiLU
  (memory-bound, elementwise), the optimal block size and vectorization width
  differ — but the **code shape doesn't**.

For BF16 elementwise on Xe2 specifically, vectorize with `sycl::vec<sycl::ext::oneapi::bfloat16, 8>`
and let the JIT lower to block-load SEND messages — same idea as CUDA's
`__half2` / 128-bit ld.global.
