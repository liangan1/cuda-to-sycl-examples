# SiLU — standalone vectorized kernel, CUDA vs SYCL

This is layer **3** of the three-layer demo, mirroring layer **3** of the
in-tree PyTorch story (`../upstream/`) but **without** any PyTorch dependency:

- [silu_vectorized.cu](silu_vectorized.cu) — a self-contained extract of
  PyTorch upstream's `vectorized_elementwise_kernel`
  ([CUDALoops.cuh#L167](../../../pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh#L167)),
  trimmed down to one elementwise op (SiLU).
- [silu_vectorized.sycl.cpp](silu_vectorized.sycl.cpp) — the same structure in
  SYCL: named functor + `nd_item<1>` + `sycl::vec<float, 4>` aligned load/store.

Both files compile and run **standalone**. No `TensorIterator`, no
`gpu_kernel`, no `AT_DISPATCH_*`. The point: once the PyTorch glue is
stripped, the CUDA and SYCL kernels are the same two-branch SPMD body.

## 1. Shape of the kernel — what we kept from upstream

Both files keep exactly the four ideas that make
`vectorized_elementwise_kernel` fast:

1. **Tile-per-block.** Each block / work-group owns `kBlockWork = VEC * BLOCK`
   contiguous output elements.
2. **Aligned wide load/store on the fast path.** CUDA: `float4` reinterpret;
   SYCL: `sycl::vec<float, 4>::{load,store}`. Both lower to a single 128-bit
   memory transaction per work-item.
3. **Branch on `remaining < kBlockWork`** to fall back to a scalar unroll loop
   when the trailing tile is short. No out-of-bounds access on the fast path,
   no warp divergence on the slow path beyond the tail block.
4. **VEC as a template parameter** so the compiler can fully unroll the
   per-element op (SiLU here).

What we dropped, intentionally: `TensorIterator`, `OffsetCalculator`,
`LoadWithoutCast`, the `policies::vectorized` / `policies::unroll` indirection,
dtype dispatch. Those exist in upstream because PyTorch supports arbitrary
shapes, strides, broadcasts, and 14 dtypes — none of which a customer needs to
see to evaluate "is the port mechanical?".

## 2. Side-by-side — the kernel body

CUDA ([silu_vectorized.cu](silu_vectorized.cu)):
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

SYCL ([silu_vectorized.sycl.cpp](silu_vectorized.sycl.cpp)) — **free-function kernel**:
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
        in.load (0, sycl::multi_ptr<const float, sycl::access::address_space::global_space>(x + base));
        vec_t out;
        #pragma unroll
        for (int j = 0; j < kVecSize; ++j) out[j] = silu_op(in[j]);
        out.store(0, sycl::multi_ptr<float,       sycl::access::address_space::global_space>(y + base));
    } else {
        #pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
            int i = tile_base + lid + j * wgsz;
            if (i < N) y[i] = silu_op(x[i]);
        }
    }
}
```

Launch glue (same `nd_launch` pattern as step 1):
```cpp
syclexp::nd_launch(q, ndr,
                   syclexp::kernel_function<silu_vectorized_kernel>,
                   d_x, d_y, N);
q.wait();
```

For reference, the explicit-API form on the CUDA side
(what `kernel<<<...>>>(args)` lowers to) is the direct counterpart:
```cpp
void* args[] = { (void*)&d_x, (void*)&d_y, (void*)&N };
cudaLaunchKernel((const void*)&silu_vectorized_kernel<kVecSize>,
                 dim3(grid), dim3(kBlockSize),
                 args, /*sharedMem=*/0, /*stream=*/0);
cudaDeviceSynchronize();
```
See the [CUDA Runtime API reference](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c).
Same three pieces of information — kernel address, launch geometry, packed
args — just in different syntactic clothing.

The two branches are line-for-line the same shape; only the **vector type
spelling** and the **launch wrapper** change. The SYCL kernel is a plain
top-level free function (sycl_ext_oneapi_free_function_kernels), the direct
structural counterpart of CUDA's `__global__` — same approach as step 1.

## 3. Concept mapping — vectorized layer

| Concept                         | CUDA                                              | SYCL (free-function kernel)                                                    |
| ------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------ |
| Kernel marker                   | `__global__`                                      | `SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>)) void k(...)` |
| Captures / args                 | function parameters                               | function parameters (same)                                                     |
| Work-item coordinates           | `blockIdx.x / threadIdx.x / blockDim.x`           | `auto it = syclwi::get_nd_item<1>(); it.get_group(0) / get_local_id(0) / get_local_range(0)` |
| 128-bit aligned load            | `float4 in = *reinterpret_cast<const float4*>(p)` | `sycl::vec<float,4> v; v.load(0, multi_ptr<…global_space>(p))`                |
| 128-bit aligned store           | `*reinterpret_cast<float4*>(p) = out`             | `v.store(0, multi_ptr<…global_space>(p))`                                     |
| Element access in vec           | `in.x / in.y / in.z / in.w`                       | `in[0] / in[1] / in[2] / in[3]`                                               |
| Math intrinsic                  | `__expf(-v)`                                      | `sycl::exp(-v)`                                                               |
| Launch                          | `kernel<<<grid, BLOCK>>>(args)` &nbsp;or&nbsp; `cudaLaunchKernel(funcAddr, dim3(grid), dim3(BLOCK), args, 0, 0)` | `syclexp::nd_launch(q, ndr, syclexp::kernel_function<kernel>, args...)` |
| Sync                            | `cudaDeviceSynchronize()`                         | `q.wait()`                                                                    |
| Stream / queue                  | implicit / `cudaStream_t`                         | `sycl::queue`                                                                 |

## 4. What is _not_ in this file that is in upstream

If a customer asks "what did you simplify away?" — here is the honest list,
with one-line justifications:

| Upstream feature dropped here              | Why it's not needed for a kernel demo |
| ------------------------------------------ | -------------------------------------- |
| `TensorIterator` (shape/stride/broadcast)  | Demo uses a flat 1-D array; PyTorch needs N-D + strided + broadcast |
| `OffsetCalculator`                         | Trivial 1-D offset = `tile_base + lid*VEC + j` |
| `LoadWithoutCast` / `StoreWithoutCast`     | Single dtype (`float`); no fp16↔fp32 promotion path |
| `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2` | Single dtype; no runtime type switch |
| `policies::vectorized` / `policies::unroll`| Inlined directly — two `if/else` branches |
| `vec_size = can_vectorize_up_to(...)`      | Compile-time `VEC = 4` for the demo; production picks per-tensor |
| Multi-output / multi-input arities         | Single in, single out |
| `at::opmath_type` promotion                | `float` everywhere |

Add any of these back and you get the upstream file in
[../upstream/silu_upstream_cuda.cu](../upstream/silu_upstream_cuda.cu) /
[../upstream/silu_upstream_sycl.cpp](../upstream/silu_upstream_sycl.cpp) — no
new programming-model concepts, just more bookkeeping.

## 5. Where the two files diverge — and why

| Knob in this folder | CUDA value | SYCL value | Why differ |
| ------------------- | ---------- | ---------- | ---------- |
| `kBlockSize` / `kWgSize` | 128 | 256 | NVIDIA SM occupancy peaks at ~128 threads/block for memory-bound elementwise; Xe2 sub-slice fits 1024 work-items, 256 is a balanced default |
| `kVecSize`          | 4 (float4) | 4 (`sycl::vec<float,4>`) | Same 16-B wide access on both — CUDA `ld.global.v4.f32`, Xe2 LSC `load.ugm.d32x4` |
| Math intrinsic      | `__expf`   | `sycl::exp`   | Both map to the device fast-math `exp` |

Everything else — tile size logic, branch condition, scalar tail, kernel
structure — is identical.

## 6. Build & run

```bash
# CUDA
nvcc -O3 -arch=sm_80 silu_vectorized.cu -o silu_vec_cuda && ./silu_vec_cuda

# SYCL (Intel GPU)
icpx -fsycl -O3 -fsycl-targets=spir64 silu_vectorized.sycl.cpp -o silu_vec_sycl
./silu_vec_sycl
```

Expected output (both):
```
... SiLU[12345] = 0.xxxxxx   (ref 0.xxxxxx)
```

## 7. Benchmark — naive vs vectorized on Intel XPU

[silu_bench.sycl.cpp](silu_bench.sycl.cpp) is a single-file SYCL micro-bench
(methodology inspired by [bytedance/xpu-perf](https://github.com/bytedance/xpu-perf):
warmup + N timed iters, SYCL event profiling, effective bandwidth = bytes / time).
It runs both the naive (one-elem-per-work-item) and the vectorized
(`sycl::vec<float,4>` + tile-per-WG + scalar-tail) kernels at several problem
sizes spanning L1/L2 → LLC → HBM.

```bash
icpx -fsycl -O3 -fsycl-targets=spir64 silu_bench.sycl.cpp -o silu_bench
./silu_bench                       # default: warmup=10, iters=50
./silu_bench --peak-gbps 456       # add a %peak column (BMG-G31 ~456 GB/s)
./silu_bench --size 1048576 --iters 100
```

### Measured on Intel BMG-G31 (Arc Battlemage, Driver 1.14.36300+8)

| N (elems) | working set | naive min ms | naive GB/s | vec min ms | vec GB/s | **speedup** |
| --------- | ----------- | ------------ | ---------- | ---------- | -------- | ----------- |
|     65,536 | 0.5 MiB (L2)   | 0.001 |   630 | 0.001 |   560 | **0.89×** |
|    262,144 | 2 MiB (LLC)    | 0.002 | 1,061 | 0.001 | 1,440 | **1.36×** |
|  1,048,576 | 8 MiB (LLC)    | 0.006 | 1,467 | 0.004 | 2,305 | **1.57×** |
|  4,194,304 | 32 MiB (HBM)   | 0.053 |   633 | 0.054 |   616 | 0.97× |
| 16,777,216 | 128 MiB (HBM)  | 0.252 |   533 | 0.247 |   544 | 1.02× |
| 67,108,864 | 512 MiB (HBM)  | 1.015 |   529 | 0.984 |   546 | 1.03× |

### How to read it

1. **HBM-resident (≥ 16 MiB working set)** — both kernels saturate at
   ~530 GB/s (≈ 87 % of BMG-G31's ~610 GB/s measured peak triad). SiLU is
   memory-bound: 2 ops/elem (`exp` + a divide on opmath_t) against 8 B/elem of
   traffic → AI ≈ 0.25 FLOP/B. Vectorization gives essentially **no speedup**
   here — the GPU is already wire-limited, the only difference between the two
   kernels is how the compiler issues the SEND messages, not how many bytes
   move.

2. **LLC-resident (256 KiB – 4 MiB working set)** — vectorization gives
   **1.36×–1.57×** speedup. Here the kernel is no longer DRAM-bound, so fewer,
   wider LSC `load.ugm.d32x4` messages (one per 4 elems) win against more,
   narrower `d32x1` messages (one per elem) — same bytes, fewer message-issue
   cycles, lower L1 request-rate pressure, more thread-level parallelism.
   Effective bandwidth climbs above the HBM peak because the data is hitting
   L2/LLC, not DRAM.

3. **Tiny (64 KiB working set)** — both kernels are dominated by kernel-launch
   overhead (~1 µs), and the naive kernel happens to be slightly faster because
   its grid arithmetic is simpler. Don't read anything into this row.

### What this tells the customer

The vectorization pattern lifted from PyTorch CUDA's
`vectorized_elementwise_kernel` ports to Xe2 cleanly and gives the same shape
of win it gives on NVIDIA: **large speedup when the kernel is cache-bound,
zero speedup when it's DRAM-bound** (because both versions already saturate
the bus). That is the exact same roofline behavior CUDA engineers expect — the
mental model carries over.

Where Xe2 differs from NVIDIA:
- Optimal work-group size (256 here vs ~128 on NVIDIA) — picked once per
  device class, not per kernel.
- LSC message granularity — `d32x4` (16 B SEND) is the sweet spot;
  `d32x8` (32 B) hits register-pressure on small kernels.
- Fast-math `exp` — `sycl::exp` lowers to the same hardware op as CUDA's
  `__expf` (single-cycle EM lookup, ~22-bit accuracy).

None of these change the **shape** of the kernel — they're just constants in
the `kWgSizeVec` / `kVecSize` knobs at the top of the file.

## 7. The three-layer story, recap

| Layer | Folder | Programming-model concepts | Optimization concepts |
| ----- | ------ | -------------------------- | --------------------- |
| 1 | `..` (silu.cu / silu.sycl.cpp) | SPMD over grid/nd-range; `__global__` ↔ free-function kernel | none — scalar, one-elem-per-thread |
| 2 | `../torch_ext/` | + PyTorch C++ extension glue (`data_ptr<T>`, stream) | none |
| **3a** | **this folder** | + named functor, aligned wide load/store, tile-per-block | vectorized fast path + scalar tail |
| 3b | `../upstream/` | + `TensorIterator`, dtype dispatch, policy abstraction | what ships in PyTorch today |

Layer 3a is the one to put in front of a CUDA engineer asking
"how would I write a SiLU SYCL kernel today, by hand, with the same
performance pattern I'd use in CUDA?" — the answer is *this file*.
