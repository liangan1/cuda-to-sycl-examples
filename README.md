# cuda-to-sycl-examples

Side-by-side **CUDA ↔ SYCL** kernel examples for engineers porting NVIDIA code
to Intel XPU. The thesis: porting is structurally mechanical — the programming
model, math, indexing, and memory model all map one-to-one; only µarch tuning
constants change. **All contents in this repo are gernated by AI**. 

Worked example: **SiLU** (Swish), `y = x * sigmoid(x) = x / (1 + exp(-x))`,
walked in **three steps** of increasing realism.

---

## The 3-step story

### Step 1 — Concept mapping with a bare free-function kernel

**Detail:** [silu/README.md](silu/README.md) · **Source:**
[silu/silu.cu](silu/silu.cu) ↔ [silu/silu.sycl.cpp](silu/silu.sycl.cpp)

A minimal CUDA `__global__` kernel and its SYCL free-function-kernel twin.
Kernel body is byte-for-byte identical; only the launch glue differs.

| Concept                  | CUDA                                        | SYCL (free-function kernel)                                                                  |
| ------------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Kernel marker            | `__global__ void silu_kernel(...)`          | `SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)) void silu_kernel(...)`|
| Global thread index      | `blockIdx.x * blockDim.x + threadIdx.x`     | `this_work_item::get_nd_item<1>().get_global_id(0)`                                          |
| Math intrinsic           | `expf(-v)`                                  | `sycl::exp(-v)`                                                                              |
| Launch                   | `silu_kernel<<<grid, block>>>(d_x,d_y,N)`   | `syclexp::nd_launch(q, ndr, kernel_function<silu_kernel>, d_x, d_y, N)`                      |
| Sync / alloc / free      | `cudaDeviceSynchronize/Malloc/Free`         | `q.wait()` / `sycl::malloc_device` / `sycl::free`                                            |

**CUDA** ([silu/silu.cu](silu/silu.cu)):
```cpp
__global__ void silu_kernel(const float* __restrict__ x,
                            float*       __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}
// launch
silu_kernel<<<grid, BLOCK>>>(d_x, d_y, N);
cudaDeviceSynchronize();
```

**SYCL** ([silu/silu.sycl.cpp](silu/silu.sycl.cpp)):
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
// launch (nd_launch from sycl_ext_oneapi_free_function_kernels —
// the closest SYCL equivalent of CUDA <<<grid, block>>>)
syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_kernel>, d_x, d_y, N);
q.wait();
```

The three lines inside the `if` are the same.

---

### Step 2 — Analyze the real PyTorch upstream implementations

**Detail:** [silu/upstream/README.md](silu/upstream/README.md) · **Source:**
[silu/upstream/silu_upstream_cuda.cu](silu/upstream/silu_upstream_cuda.cu) ↔
[silu/upstream/silu_upstream_sycl.cpp](silu/upstream/silu_upstream_sycl.cpp)

The actual code shipping in PyTorch (`aten/src/ATen/native/cuda/ActivationSiluKernel.cu`)
vs torch-xpu-ops (`src/ATen/native/xpu/sycl/ActivationSiluKernels.cpp`). Both
plug into the same `silu_stub`; the math is identical.

| Concept                       | CUDA upstream                                  | SYCL/XPU upstream                                       |
| ----------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| Iterator                      | `TensorIteratorBase&`                          | `TensorIteratorBase&` (same)                            |
| Dispatch macro                | `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2`  | same macro, same args                                   |
| Element op                    | `[] GPU_LAMBDA(scalar_t x) { ... }`            | `struct SiluFunctor { operator()(...) }`                |
| Element-op driver             | `gpu_kernel(iter, op)`                         | `gpu_kernel(iter, op)` (same name)                      |
| `opmath_t` promotion          | `at::opmath_type<scalar_t>`                    | same                                                    |
| `exp`                         | `::exp` / `c10::cuda::compat::exp`             | `std::exp` (resolves to `sycl::exp`)                    |
| Stub registration             | `REGISTER_DISPATCH(silu_stub, &silu_kernel)`   | `REGISTER_XPU_DISPATCH(silu_stub, &xpu::silu_kernel)`   |

**CUDA upstream** ([silu/upstream/silu_upstream_cuda.cu](silu/upstream/silu_upstream_cuda.cu)):
```cpp
void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "silu_cuda", [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
          using opmath_t = at::opmath_type<scalar_t>;
          const opmath_t x_acc = static_cast<opmath_t>(x);
          return x_acc / (opmath_t(1) + ::exp(-x_acc));
        });
      });
}
REGISTER_DISPATCH(silu_stub, &silu_kernel)
```

**SYCL/XPU upstream** ([silu/upstream/silu_upstream_sycl.cpp](silu/upstream/silu_upstream_sycl.cpp)):
```cpp
template <typename scalar_t>
struct SiluFunctor {
  scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t x_acc = static_cast<opmath_t>(x);
    return x_acc / (opmath_t(1) + std::exp(-x_acc));        // <-- same line
  }
};
void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.dtype(), "silu_xpu", [&]() {
        gpu_kernel(iter, SiluFunctor<scalar_t>());
      });
}
REGISTER_XPU_DISPATCH(silu_stub, &xpu::silu_kernel);
```

The only structural diff: CUDA's `GPU_LAMBDA` becomes a named `struct` in SYCL
(SYCL kernels need named types for kernel-bundle lookup). The body —
`x_acc / (opmath_t(1) + exp(-x_acc))` — is literally identical.

[silu/upstream/README.md](silu/upstream/README.md) drills one layer deeper,
mapping `gpu_kernel` itself: CUDA's `vectorized_elementwise_kernel` ↔ SYCL's
`VectorizedElementwiseKernel` — same two-branch shape (fast vectorized path +
scalar tail), same policy abstraction, same helper names.

---

### Step 3 — Standalone vectorized SYCL kernel (no PyTorch dependency)

**Detail:** [silu/vectorized/README.md](silu/vectorized/README.md) · **Source:**
[silu/vectorized/silu_vectorized.cu](silu/vectorized/silu_vectorized.cu) ↔
[silu/vectorized/silu_vectorized.sycl.cpp](silu/vectorized/silu_vectorized.sycl.cpp)

The structure of PyTorch CUDA's `vectorized_elementwise_kernel` extracted into
a self-contained single file (no `TensorIterator`, no dtype dispatch). Same
two-branch shape; ports to SYCL line-for-line.

| Concept                | CUDA                                              | SYCL (free-function kernel)                                                   |
| ---------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------- |
| Kernel marker          | `__global__`                                      | `SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)) void k(...)` (top-level function, same as step 1) |
| Block / thread index   | `blockIdx.x` / `threadIdx.x` / `blockDim.x`       | `auto it = syclwi::get_nd_item<1>(); it.get_group(0) / get_local_id(0) / get_local_range(0)` |
| 128-bit aligned load   | `float4 in = *reinterpret_cast<const float4*>(p)` | `sycl::vec<float,4> v; v.load(0, multi_ptr<…global_space>(p))`                |
| 128-bit aligned store  | `*reinterpret_cast<float4*>(p) = out`             | `v.store(0, multi_ptr<…global_space>(p))`                                     |
| Launch                 | `kernel<<<grid, BLOCK>>>(args)`                   | `syclexp::nd_launch(q, ndr, kernel_function<kernel>, args...)`                |
| Math intrinsic         | `__expf(-v)`                                      | `sycl::exp(-v)`                                                               |

**CUDA** ([silu/vectorized/silu_vectorized.cu](silu/vectorized/silu_vectorized.cu)):
```cpp
template <int VEC>
__global__ void silu_vectorized_kernel(const float* __restrict__ x,
                                       float*       __restrict__ y, int N) {
    int tile_base = blockIdx.x * kBlockWork;
    int remaining = N - tile_base;
    if (remaining >= kBlockWork) {
        int base = tile_base + threadIdx.x * VEC;
        const float4* xv = reinterpret_cast<const float4*>(x + base);
        float4* yv       = reinterpret_cast<float4*>(y + base);
        float4 in = *xv;  float4 out;
        out.x = silu_op(in.x);  out.y = silu_op(in.y);
        out.z = silu_op(in.z);  out.w = silu_op(in.w);
        *yv = out;
    } else {
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int i = tile_base + threadIdx.x + j * blockDim.x;
            if (i < N) y[i] = silu_op(x[i]);
        }
    }
}
```

**SYCL** ([silu/vectorized/silu_vectorized.sycl.cpp](silu/vectorized/silu_vectorized.sycl.cpp)) — **free-function kernel**:
```cpp
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void silu_vectorized_kernel(const float* x, float* y, int N) {
    auto item = syclwi::get_nd_item<1>();
    int grpid = item.get_group(0);
    int lid   = item.get_local_id(0);
    int wgsz  = item.get_local_range(0);
    int tile_base = grpid * kBlockWork;
    int remaining = N - tile_base;
    if (remaining >= kBlockWork) {
        int base = tile_base + lid * kVecSize;
        using vec_t = sycl::vec<float, kVecSize>;
        vec_t in;
        in.load(0, sycl::multi_ptr<const float,
                    sycl::access::address_space::global_space>(x + base));
        vec_t out;
        #pragma unroll
        for (int j = 0; j < kVecSize; ++j) out[j] = silu_op(in[j]);
        out.store(0, sycl::multi_ptr<float,
                     sycl::access::address_space::global_space>(y + base));
    } else {
        #pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
            int i = tile_base + lid + j * wgsz;
            if (i < N) y[i] = silu_op(x[i]);
        }
    }
}
```

Same `if (remaining >= kBlockWork)` fast-path branch + scalar-tail `for` loop
with bounds check. Same algorithm, just spelled as a SYCL free function
(same kernel form as step 1) instead of a CUDA `__global__`.

---

## Benchmark — naive vs vectorized on Intel XPU

A single-file SYCL micro-bench (methodology inspired by
[bytedance/xpu-perf](https://github.com/bytedance/xpu-perf)) — warmup +
N timed iters via SYCL event profiling, effective bandwidth = bytes / time.
Runs both the naive (1-elem-per-WI) and the vectorized
(`sycl::vec<float,4>` + tile-per-WG + scalar-tail) kernel across sizes
spanning L1/L2 → LLC → HBM.

**Source:** [silu/vectorized/silu_bench.sycl.cpp](silu/vectorized/silu_bench.sycl.cpp)

### Build & run

Requires Intel oneAPI (`icpx`, tested with 2025.3) and an Intel GPU.

```bash
cd silu/vectorized

# build
icpx -fsycl -O3 -fsycl-targets=spir64 silu_bench.sycl.cpp -o silu_bench

# default sweep (warmup=10, iters=50, 6 sizes from 64 KiB to 256 MiB)
./silu_bench

# add a %peak column (e.g. BMG-G31 measured triad ~610 GB/s)
./silu_bench --peak-gbps 610

# tighter timing on a single size
./silu_bench --size 1048576 --iters 200 --warmup 50
```

CLI: `--warmup N` · `--iters N` · `--peak-gbps F` · `--size N` (repeatable).

### Measured on Intel BMG-G31 (Arc Battlemage, Driver 1.14.36300+8)

| N (elems)  | working set | naive min ms | naive GB/s | vec min ms | vec GB/s | **speedup** |
| ---------: | :---------- | -----------: | ---------: | ---------: | -------: | ----------: |
|     65,536 | 0.5 MiB     |        0.001 |        630 |      0.001 |      560 |   **0.89×** |
|    262,144 | 2 MiB (LLC) |        0.002 |      1,061 |      0.001 |    1,440 |   **1.36×** |
|  1,048,576 | 8 MiB (LLC) |        0.006 |      1,467 |      0.004 |    2,305 |   **1.57×** |
|  4,194,304 | 32 MiB (HBM)|        0.053 |        633 |      0.054 |      616 |       0.97× |
| 16,777,216 | 128 MiB (HBM)|       0.252 |        533 |      0.247 |      544 |       1.02× |
| 67,108,864 | 512 MiB (HBM)|       1.015 |        529 |      0.984 |      546 |       1.03× |

### How to read it

- **HBM-resident (≥ 16 MiB)** — both saturate at ~530 GB/s. SiLU is
  memory-bound (AI ≈ 0.25 FLOP/B); the bus is already wire-limited, so
  vectorization gives **no speedup**. Same roofline behavior as on NVIDIA.
- **LLC-resident (256 KiB – 4 MiB)** — vectorization wins **1.36× – 1.57×**.
  Fewer, wider LSC `load.ugm.d32x4` messages reduce request-rate pressure on
  L1/L2 vs the naive per-element `d32x1` traffic.
- **Tiny (≤ 64 KiB)** — kernel-launch overhead (~1 µs) dominates; don't read
  signal into this row.

See [silu/vectorized/README.md](silu/vectorized/README.md) §7 for a detailed
roofline interpretation and the three µarch knobs (`kWgSizeVec`, `kVecSize`,
fast-math `exp`) at the top of the file you can tune.

---

## License

Apache-2.0
