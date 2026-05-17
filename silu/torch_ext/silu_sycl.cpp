// silu_sycl.cpp
// SiLU as a PyTorch XPU extension (SYCL free-function kernel).
//
//   y = x * sigmoid(x) = x / (1 + exp(-x))
//
// Build via torch.utils.cpp_extension.SyclExtension — see silu_torch.py.
//
// Concept mapping vs the CUDA version (silu_cuda.cu):
//   __global__              ->  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(nd_range_kernel<1>)
//   blockIdx/threadIdx      ->  this_work_item::get_nd_item<1>().get_global_id(0)
//   <<<grid,block,0,strm>>> ->  q.parallel_for(nd_range, kernel)  on c10::xpu stream
//   at::cuda::getCurrentCUDAStream() -> c10::xpu::getCurrentXPUStream()
//   AT_DISPATCH_FLOATING_TYPES_AND_HALF  -> same macro, works on XPU tensors

#include <torch/extension.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

// ---- Device kernel: free function (same shape as CUDA __global__) ----------
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

// ---- C++ wrapper called from Python ----------------------------------------
torch::Tensor silu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_xpu(),        "x must be an XPU tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    const int64_t n = x.numel();

    constexpr int BLOCK = 256;
    const int64_t grid = (n + BLOCK - 1) / BLOCK;
    sycl::nd_range<1> ndr{sycl::range<1>(grid * BLOCK), sycl::range<1>(BLOCK)};

    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "silu_forward", [&] {
        auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            q.get_context());
        sycl::kernel k = syclex::get_kernel<silu_kernel<scalar_t>>(bundle);

        auto* xp = x.data_ptr<scalar_t>();
        auto* yp = y.data_ptr<scalar_t>();
        q.submit([&](sycl::handler& h) {
            h.set_args(xp, yp, n);
            h.parallel_for(ndr, k);
        });
    });
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_forward", &silu_forward, "SiLU forward (SYCL/XPU)");
}
