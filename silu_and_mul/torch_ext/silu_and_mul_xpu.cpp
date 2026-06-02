// silu_and_mul_xpu.cpp
//
// PyTorch C++ extension for SiLU-and-Mul fusion kernel on Intel XPU
// Registers as torch.ops.silu_and_mul_xpu.silu_and_mul
//
// Build:
//   python setup.py install
//
// Usage:
//   import torch
//   import silu_and_mul_xpu
//   out = torch.ops.silu_and_mul_xpu.silu_and_mul(input)

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUFunctions.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

// ---------------------------------------------------------------------------
// Kernel (free-function style) - same as standalone version
// ---------------------------------------------------------------------------

inline float silu_op(float x) {
  return x / (1.0f + sycl::exp(-x));
}

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_and_mul_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    int d)
{
  auto item = syclwi::get_nd_item<1>();
  const int token_idx = item.get_group(0);
  const int tid = item.get_local_id(0);
  const int block_size = item.get_local_range(0);
  
  const float* gate = input + token_idx * 2 * d;
  const float* up   = input + token_idx * 2 * d + d;
  float* out_ptr    = out + token_idx * d;
  
  // Vectorized path
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
  
  // Scalar tail
  for (int i = num_vec * VEC_SIZE + tid; i < d; i += block_size) {
    out_ptr[i] = silu_op(gate[i]) * up[i];
  }
}

// ---------------------------------------------------------------------------
// PyTorch binding function using PyTorch XPU queue
// ---------------------------------------------------------------------------

torch::Tensor silu_and_mul_xpu(torch::Tensor input) {
  // Input shape: [num_tokens, 2 * d]
  // Output shape: [num_tokens, d]
  
  TORCH_CHECK(input.is_xpu(), "Input must be on XPU device");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
  TORCH_CHECK(input.dim() == 2, "Input must be 2D [num_tokens, 2*d]");
  TORCH_CHECK(input.size(1) % 2 == 0, "Input dim 1 must be even (2*d)");
  
  int num_tokens = input.size(0);
  int d = input.size(1) / 2;
  
  // Allocate output
  auto output = torch::empty({num_tokens, d}, 
                             torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(input.device()));
  
  // Get PyTorch XPU queue
  sycl::queue q = c10::xpu::get_queue_from_stream(
      c10::xpu::getCurrentXPUStream(input.device().index()));
  
  // Launch kernel
  constexpr int BLOCK_SIZE = 256;
  sycl::nd_range<1> ndr{num_tokens * BLOCK_SIZE, BLOCK_SIZE};
  
  syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_and_mul_kernel>,
                     output.data_ptr<float>(),
                     input.data_ptr<float>(),
                     d);
  
  return output;
}

// ---------------------------------------------------------------------------
// PyTorch C++ extension registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &silu_and_mul_xpu, "SiLU-and-Mul fusion kernel for XPU");
}

// Register as torch.ops namespace
TORCH_LIBRARY(silu_and_mul_xpu, m) {
  m.def("silu_and_mul(Tensor input) -> Tensor", &silu_and_mul_xpu);
}
