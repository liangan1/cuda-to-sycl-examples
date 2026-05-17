# SiLU in upstream PyTorch — CUDA vs SYCL, the real code

This is the strongest version of the demo: not a hand-written port, but the
**actual code shipping in upstream PyTorch today**, placed side-by-side.

| Side  | File in this folder | Upstream source |
| ----- | ------------------- | --------------- |
| CUDA  | [silu_upstream_cuda.cu](silu_upstream_cuda.cu)  | [pytorch/aten/src/ATen/native/cuda/ActivationSiluKernel.cu](../../../pytorch/aten/src/ATen/native/cuda/ActivationSiluKernel.cu) |
| SYCL  | [silu_upstream_sycl.cpp](silu_upstream_sycl.cpp) | [torch-xpu-ops/src/ATen/native/xpu/sycl/ActivationSiluKernels.cpp](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/ActivationSiluKernels.cpp) |

## 1. The forward kernel — side by side

CUDA (lambda form):
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
```

SYCL (functor form, semantically identical):
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
```

The compute line — `x_acc / (opmath_t(1) + exp(-x_acc))` — is **literally
identical**. Everything else is glue.

## 2. Concept mapping — what changes, what doesn't

| Concept                  | CUDA upstream                            | SYCL/XPU upstream                          |
| ------------------------ | ---------------------------------------- | ------------------------------------------ |
| Iterator type            | `TensorIteratorBase&`                    | `TensorIteratorBase&` (same)               |
| Dispatch macro           | `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2` | same macro, same args                 |
| Element op               | `[] GPU_LAMBDA(scalar_t x) { ... }`      | `struct SiluFunctor { operator()(...) }`   |
| Element-op invoker       | `gpu_kernel(iter, op)`                   | `gpu_kernel(iter, op)` (same name)         |
| `opmath_t`               | `at::opmath_type<scalar_t>`              | same                                       |
| `exp`                    | `::exp` / `c10::cuda::compat::exp`       | `std::exp` (resolves to `sycl::exp`)       |
| Dispatch registration    | `REGISTER_DISPATCH(silu_stub, &silu_kernel)` | `REGISTER_XPU_DISPATCH(silu_stub, &xpu::silu_kernel)` |
| Stub                     | `silu_stub` (defined in ATen)            | `silu_stub` — same stub, different impl    |

Why lambda → functor: SYCL kernels need a *named type* so the runtime can look
up the compiled kernel by symbol. CUDA's `GPU_LAMBDA` macro emits a unique
unnamed type behind the scenes; SYCL makes you spell it out. The body is
identical.

## 3. Why this is the right pitch for customers

1. **Same architectural pattern.** Both backends plug into the same
   `silu_stub` dispatch hook. PyTorch's device-dispatch machinery decides at
   runtime which `silu_kernel` to call based on the tensor's device — the user
   API (`torch.nn.functional.silu`) doesn't change.
2. **Same numerics.** Both promote `Half`/`BFloat16` to `opmath_t` (float),
   compute in float, cast back. Identical accuracy.
3. **Same elementwise helper.** Both call `gpu_kernel(iter, op)`. CUDA's
   `gpu_kernel` lives in `ATen/native/cuda/Loops.cuh`; SYCL's lives in
   `ATen/native/xpu/sycl/Loops.h`. Same API, different .cu/.cpp underneath.
4. **Only "porting cost" is lambda → functor.** That's a 5-line mechanical
   change per kernel.

## 4. The full dispatch path — Python to silicon

```
  torch.nn.functional.silu(x)
        │ (torch/nn/functional.py L2398)
        ▼
  torch._C._nn.silu(x)               ─► aten::silu schema
        │
        ▼  TORCH_META_FUNC(silu)        (Activation.cpp)
  TORCH_IMPL_FUNC(silu_out) → silu_stub(device_type(), iter)
        │
        ├── device_type() == kCUDA ──► REGISTER_DISPATCH (ActivationSiluKernel.cu)
        │                              └► silu_kernel(iter)
        │                                  └► gpu_kernel(iter, GPU_LAMBDA …)
        │
        └── device_type() == kXPU  ──► REGISTER_XPU_DISPATCH (ActivationSiluKernels.cpp)
                                       └► xpu::silu_kernel(iter)
                                           └► gpu_kernel(iter, SiluFunctor<T>())
```

Same flow, same stub, swappable backends. That is the message.

## 5. Build context

These files are **not** standalone — they are read in the context of their
respective trees:

- The CUDA file is built as part of `libtorch_cuda` via `setup.py develop`
  in the PyTorch repo.
- The SYCL file is built as part of `libtorch_xpu_ops` (the `torch-xpu-ops`
  submodule), linked by `libtorch_xpu`.

For a **self-contained** demo (a free-function kernel + custom-op extension a
customer can compile in 30 seconds), see the sibling folders:

- [../silu.cu](../silu.cu) + [../silu.sycl.cpp](../silu.sycl.cpp) — raw kernels, no PyTorch.
- [../torch_ext/](../torch_ext/) — PyTorch C++ extension (JIT-built).

## 6. One layer deeper — concept mapping inside `gpu_kernel`

The `silu_kernel` file ends at `gpu_kernel(iter, op)`. Both backends implement
that helper themselves. The next layer of mapping shows the machinery is
**structurally identical** — same five stages, same names — just one is built
on CUDA `<<<>>>` and the other on SYCL `parallel_for`.

| Stage                                  | CUDA — [aten/src/ATen/native/cuda/](../../../pytorch/aten/src/ATen/native/cuda/) | SYCL — [torch-xpu-ops/src/ATen/native/xpu/sycl/](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/) |
| -------------------------------------- | ----------- | ---------- |
| Entry helper                           | `gpu_kernel(iter, op)` ([Loops.cuh](../../../pytorch/aten/src/ATen/native/cuda/Loops.cuh#L115)) | `gpu_kernel(iter, op)` ([Loops.h](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L634)) |
| Strip cast / pick path                 | `gpu_kernel_impl` ([CUDALoops.cuh](../../../pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh)) | `gpu_kernel_impl` ([Loops.h](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L557)) |
| Vectorization width pick               | `memory::can_vectorize_up_to<func_t>(data)` → 16 / sizeof(out) | `at::native::memory::can_vectorize_up_to(...)` → vec ∈ {1,2,4,8,16} |
| Vectorized launcher                    | `launch_vectorized_kernel` ([CUDALoops.cuh#L298](../../../pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh#L298)) | `launch_vectorized_kernel` ([Loops.h#L393](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L393)) |
| SPMD kernel body                       | `__global__ vectorized_elementwise_kernel<vec_size>` ([CUDALoops.cuh#L167](../../../pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh#L167)) | `struct VectorizedElementwiseKernel<vec_size>` w/ `operator()(nd_item<1>)` ([Loops.h#L102](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L102)) |
| Per-thread vectorized policy           | `memory::policies::vectorized<vec_size, …>` | `at::native::memory::policies::vectorized<vec_size, …>` (same name) |
| Remainder path (tail < block_work_size)| `memory::policies::unroll<…>` | `at::native::memory::policies::unroll<…>` (same name) |
| Thread / work-item driver              | `elementwise_kernel_helper(f, policy)` | `elementwise_kernel_helper<vec_size>(f, policy)` (same name) |
| Launch primitive                       | `kernel<<<grid, num_threads(), 0, stream>>>(N, f, data)` | `sycl_kernel_submit(wg_sz*num_wg, wg_sz, queue, ker)` → `q.parallel_for(nd_range, ker)` |
| Block / work-group size                | `num_threads()` (typically 128) | `syclMaxWorkItemsPerSubSlice()` (Xe2: 1024 per SS, picked per device) |
| Grid / num work-groups                 | `grid = ceil(N / (tws*num_threads()))` | `num_wg = ceil(N / (wg_sz*vec_size))` |
| Stream / queue                         | `at::cuda::getCurrentCUDAStream()` | `getCurrentSYCLQueue()` |

### 6.1 SPMD body — side by side

CUDA (`vectorized_elementwise_kernel`, `__global__`, [CUDALoops.cuh#L167](../../../pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh#L167)):
```cpp
template <int vec_size, typename func_t, typename array_t>
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
  using traits  = function_traits<func_t>;
  int remaining = N - io_block_work_size<io_size>() * blockIdx.x;
  if (remaining >= io_block_work_size<io_size>()) {
    elementwise_kernel_helper(
        f, memory::policies::vectorized<vec_size, array_t, …>(data));
  } else {
    /* tail: unroll policy with bounds check */
    auto policy = memory::policies::unroll<…>(data, remaining, …);
    elementwise_kernel_helper(f, policy);
  }
}
```

SYCL (`VectorizedElementwiseKernel`, named functor, [Loops.h#L102](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L102)):
```cpp
template <int vec_size, typename func_t, typename array_t, typename in_calc_t>
struct VectorizedElementwiseKernel {
  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid   = item.get_local_id(0);
    int group_work_size = vec_size * grpsz;
    int remaining       = numel_ - grpid * group_work_size;
    if (remaining < group_work_size) {
      auto policy = at::native::memory::policies::unroll<vec_size, …>(
          data_, remaining, ic_, …, lid, grpid, grpsz);
      elementwise_kernel_helper<vec_size>(f_, policy);
    } else {
      auto policy = at::native::memory::policies::
          vectorized<vec_size, array_t, in_calc_t>(data_, ic_, lid, grpid, grpsz);
      elementwise_kernel_helper<vec_size>(f_, policy);
    }
  }
  /* fields: numel_, f_, data_, ic_ */
};
```

The **two-branch logic is the same**: pick the vectorized policy on the fast
path, fall back to a scalar-unroll policy for the partial-tile tail. The
namespaces (`memory::policies::vectorized` / `…::unroll`) and the helper
(`elementwise_kernel_helper`) are spelled the same in both trees. What changes
is mechanical:

| CUDA idiom                       | SYCL equivalent                                |
| -------------------------------- | ---------------------------------------------- |
| `__global__` free function       | `struct K { void operator()(nd_item<1>) }` (named type, captures stored as fields) |
| `blockIdx.x`                     | `item.get_group(0)`                            |
| `threadIdx.x`                    | `item.get_local_id(0)`                         |
| `blockDim.x`                     | `item.get_local_range(0)`                      |
| `gridDim.x`                      | `item.get_group_range(0)`                      |
| `kernel<<<grid,block,0,stream>>>(args)` | `q.parallel_for(nd_range<1>{grid*block, block}, K{args…})` |
| `C10_LAUNCH_BOUNDS_1(num_threads())` | (none — wg size is chosen at submit time)  |

### 6.2 Launcher — side by side

CUDA ([CUDALoops.cuh#L298](../../../pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh#L298), excerpt):
```cpp
static inline void launch_vectorized_kernel(int64_t N, const func_t& f, array_t data) {
  auto stream  = at::cuda::getCurrentCUDAStream();
  int  vec_size = memory::can_vectorize_up_to<func_t>(data);
  int  tws      = elems_per_thread<io_size>();
  int  bws      = tws * num_threads();
  int64_t grid  = (N + bws - 1) / bws;
  switch (vec_size) {
    case 8: vectorized_elementwise_kernel<8, func_t, array_t>
              <<<grid, num_threads(), 0, stream>>>(N, f, data); break;
    case 4: vectorized_elementwise_kernel<4, func_t, array_t>
              <<<grid, num_threads(), 0, stream>>>(N, f, data); break;
    /* … 2, 1 … */
  }
}
```

SYCL ([Loops.h#L393](../../../torch-xpu-ops/src/ATen/native/xpu/sycl/Loops.h#L393), excerpt):
```cpp
static inline void launch_vectorized_kernel(int64_t N, const func_t& f,
                                            array_t data, in_calc_t input_calc, int vec_size) {
  auto wg_sz = syclMaxWorkItemsPerSubSlice();
#define VEC_KER(vec_size) {                                                   \
    auto ker = VectorizedElementwiseKernel<vec_size, func_t, array_t, in_calc_t>( \
                   N, f, data, input_calc);                                   \
    int64_t num_wg = ceil_div<int64_t>(N, wg_sz * vec_size);                  \
    sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker);    \
  }
  switch (vec_size) {
    case 8: VEC_KER(8); break;
    case 4: VEC_KER(4); break;
    /* … 2, 1 … */
  }
}
```

Same `switch`, same per-case template instantiation, same `(grid, block)` /
`(num_wg, wg_sz)` arithmetic — just `<<<…>>>` ↔ `sycl_kernel_submit` /
`q.parallel_for`.

### 6.3 Where the µarch tuning hides

Customers will (correctly) ask "if the code is the same, where does
device-specific tuning live?" Answer — these three knobs:

| Knob                          | CUDA value                              | SYCL/Xe2 value                          | Why it differs |
| ----------------------------- | --------------------------------------- | --------------------------------------- | -------------- |
| Block / work-group size       | `num_threads()` ≈ 128                   | `syclMaxWorkItemsPerSubSlice()`         | NVIDIA SM vs Intel Xe-core have different occupancy curves |
| `thread_work_size` (tws)      | `elems_per_thread<io_size>()` (4 typ.)  | folded into `vec_size` (1/2/4/8/16)     | Xe2 LSC prefers fewer, wider SEND messages; CUDA prefers more, narrower ld.global |
| Max vec width                 | 16 / sizeof(out), capped at 4 on pre-sm_90 | up to 16, gated by `max_scalar_bytes*vec ≤ 16` | DPAS / 2D-block-load granularity vs ld.global.v4 |

Everything else above — the iterator, the dispatch macro, the functor body,
the policy structs, the helper, the launcher skeleton — is the same in both
trees. **That is the porting story.**
