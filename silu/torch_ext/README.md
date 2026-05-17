# SiLU as a PyTorch extension — CUDA vs SYCL, side by side

Same demo as the standalone version, but wired into PyTorch as a custom op so
customers see the **full path**: Python `tensor.silu()` → C++ binding →
free-function device kernel.

Files:

| File                                     | Role                                  |
| ---------------------------------------- | ------------------------------------- |
| [silu_cuda.cu](silu_cuda.cu)             | CUDA `__global__` kernel + binding    |
| [silu_sycl.cpp](silu_sycl.cpp)           | SYCL free-function kernel + binding   |
| [silu_torch.py](silu_torch.py)           | JIT build + numerical check vs `F.silu` |

## 1. Kernel body — identical

CUDA:
```cpp
template <typename scalar_t>
__global__ void silu_kernel(const scalar_t* __restrict__ x,
                            scalar_t*       __restrict__ y, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        scalar_t v = x[i];
        y[i] = v / (scalar_t(1) + ::exp(-v));
    }
}
```

SYCL:
```cpp
template <typename scalar_t>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void silu_kernel(const scalar_t* x, scalar_t* y, int64_t n) {
    auto it = syclex::this_work_item::get_nd_item<1>();
    int64_t i = it.get_global_id(0);
    if (i < n) {
        scalar_t v = x[i];
        y[i] = v / (scalar_t(1) + sycl::exp(-v));
    }
}
```

Same template, same indexing pattern, same arithmetic. Only the kernel-marker
attribute and the index API call differ.

## 2. PyTorch-side mapping

| Concept                | CUDA extension                                  | SYCL / XPU extension                                       |
| ---------------------- | ----------------------------------------------- | ---------------------------------------------------------- |
| Tensor device check    | `x.is_cuda()`                                   | `x.is_xpu()`                                               |
| Dispatch macro         | `AT_DISPATCH_FLOATING_TYPES_AND_HALF`           | `AT_DISPATCH_FLOATING_TYPES_AND_HALF` (same)               |
| Current stream         | `at::cuda::getCurrentCUDAStream()`              | `c10::xpu::getCurrentXPUStream().queue()`                  |
| Launch                 | `kernel<<<grid,block,0,stream>>>(...)`          | `q.submit([&](sycl::handler& h){ h.set_args(...); h.parallel_for(ndr, k); })` |
| Build helper           | `torch.utils.cpp_extension.load(..., .cu)`      | `torch.utils.cpp_extension.load(..., .cpp, extra_sycl_cflags=...)` |
| Header                 | `<torch/extension.h>` + `<cuda_runtime.h>`      | `<torch/extension.h>` + `<sycl/sycl.hpp>` + `<c10/xpu/XPUStream.h>` |

The Python-facing API is **identical**:
```python
y = ext.silu_forward(x)          # works the same on cuda or xpu tensor
```

## 3. Run

```bash
cd torch_ext
python silu_torch.py
```

Expected:
```
=== CUDA ===          (if a CUDA GPU is present)
[cuda] max|y - ref| = 0.000e+00
=== XPU (SYCL) ===    (if an Intel GPU + oneAPI + PyTorch-XPU are present)
[xpu]  max|y - ref| = 0.000e+00
```

## 4. Takeaways for the customer

1. The **device-side code** ports almost mechanically — same template, same
   index math, same intrinsic name (`exp`).
2. The **PyTorch glue** ports one-to-one — `AT_DISPATCH_*` macros, `data_ptr<T>()`,
   `empty_like()`, stream acquisition, all behave identically; only the
   namespace prefix changes (`at::cuda` → `c10::xpu`).
3. The **build system** uses the same `torch.utils.cpp_extension.load` entry
   point; you swap `.cu` for `.cpp` + `extra_sycl_cflags`.
4. µarch tuning (block size, vector width, use of DPAS / block2d on Xe2) is a
   second, independent step — separate from the porting work.
