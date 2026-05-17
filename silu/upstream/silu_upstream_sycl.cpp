// silu_upstream_sycl.cpp
//
// Verbatim copy of upstream torch-xpu-ops' SiLU SYCL kernel, with annotations.
// Source: torch-xpu-ops/src/ATen/native/xpu/sycl/ActivationSiluKernels.cpp
//
//   y = x * sigmoid(x) = x / (1 + exp(-x))
//
// Side-by-side with silu_upstream_cuda.cu: this file is what the PyTorch XPU
// backend ships today. Compare line-by-line — the math is identical, only the
// kernel-launch plumbing (gpu_kernel) differs internally.

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>          // <-- gpu_kernel (SYCL variant)
#include <comm/XPUMathCompat.h>

#include <ATen/native/xpu/sycl/ActivationSiluKernels.h>

namespace at::native::xpu {

// ---------------------------------------------------------------------------
// SiLU forward:  y = x / (1 + exp(-x))
//
// Note: upstream uses a functor instead of a lambda, because SYCL kernels must
// be named types for offline compilation / kernel-bundle lookup. Semantically
// it is the same as the CUDA GPU_LAMBDA.
// ---------------------------------------------------------------------------
template <typename scalar_t>
struct SiluFunctor {
  scalar_t operator()(scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t x_acc = static_cast<opmath_t>(x);
    return x_acc / (opmath_t(1) + std::exp(-x_acc));
  }
};

// ---------------------------------------------------------------------------
// SiLU backward:  dx = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// ---------------------------------------------------------------------------
template <typename scalar_t>
struct SiluBackwardFunctor {
  scalar_t operator()(scalar_t dy, scalar_t x) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t dy_acc = static_cast<opmath_t>(dy);
    const opmath_t x_acc  = static_cast<opmath_t>(x);
    const opmath_t s_acc  = opmath_t(1) / (opmath_t(1) + std::exp(-x_acc));
    return dy_acc * s_acc * (opmath_t(1) + x_acc * (opmath_t(1) - s_acc));
  }
};

void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_xpu",
      [&]() { gpu_kernel(iter, SiluFunctor<scalar_t>()); });
}

void silu_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_backward_xpu",
      [&]() { gpu_kernel(iter, SiluBackwardFunctor<scalar_t>()); });
}

} // namespace at::native::xpu

// Registration lives in torch-xpu-ops/src/ATen/native/xpu/Activation.cpp:
//
//   REGISTER_XPU_DISPATCH(silu_stub,          &xpu::silu_kernel);
//   REGISTER_XPU_DISPATCH(silu_backward_stub, &xpu::silu_backward_kernel);
//
// — same `silu_stub` symbol as CUDA's REGISTER_DISPATCH; the device dispatch
// table picks the right one at runtime based on the tensor's device key.
