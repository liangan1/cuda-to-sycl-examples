// silu_and_mul_xpu_optimized_v2.cpp
//
// Simplified optimized version: Focus on proven optimizations only
// - Larger work-group size (1024 vs 512)
// - Vec_size=8 for better vectorization
// - Keep 1-block-per-token for coalescing
// - No complex SLM logic (add later if simple version works)

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <c10/xpu/XPUFunctions.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

// Simple vectorized kernel - 1 block per token (preserve coalescing)
template <int VEC_SIZE>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_and_mul_kernel_vec_optimized(
    float* __restrict__ out,
    const float* __restrict__ input,
    int d)
{
  auto item = syclwi::get_nd_item<1>();
  
  // 1-block-per-token (coalesced access)
  const int token_idx = item.get_group(0);
  const int tid = item.get_local_id(0);
  const int block_size = item.get_local_range(0);
  
  const float* gate = input + token_idx * 2 * d;
  const float* up = gate + d;
  float* out_row = out + token_idx * d;
  
  using vec_t = sycl::vec<float, VEC_SIZE>;
  const int num_vec = d / VEC_SIZE;
  
  // Vectorized loop - each thread processes multiple vectors
  for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
    const int base = vec_idx * VEC_SIZE;
    
    vec_t gate_vec, up_vec, out_vec;
    
    // Coalesced loads
    gate_vec.load(0, sycl::multi_ptr<const float,
                      sycl::access::address_space::global_space>(gate + base));
    up_vec.load(0, sycl::multi_ptr<const float,
                    sycl::access::address_space::global_space>(up + base));
    
    // Compute
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float g = gate_vec[i];
      float u = up_vec[i];
      out_vec[i] = (g / (1.0f + sycl::exp(-g))) * u;
    }
    
    // Coalesced store
    out_vec.store(0, sycl::multi_ptr<float,
                     sycl::access::address_space::global_space>(out_row + base));
  }
  
  // Scalar tail
  const int vec_covered = num_vec * VEC_SIZE;
  for (int i = vec_covered + tid; i < d; i += block_size) {
    float g = gate[i];
    float u = up[i];
    out_row[i] = (g / (1.0f + sycl::exp(-g))) * u;
  }
}

torch::Tensor silu_and_mul_xpu_optimized(torch::Tensor input) {
  TORCH_CHECK(input.is_xpu(), "Input must be on XPU device");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "Input must be 2D [num_tokens, 2*d]");
  TORCH_CHECK(input.size(1) % 2 == 0, "Input dim 1 must be even (2*d)");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "Only FP32 supported");
  
  const int num_tokens = input.size(0);
  const int d = input.size(1) / 2;
  
  auto output = torch::empty({num_tokens, d}, input.options());
  
  c10::xpu::XPUStream xpu_stream = c10::xpu::getCurrentXPUStream(input.device().index());
  sycl::queue& q = xpu_stream.queue();
  
  auto* out_ptr = output.data_ptr<float>();
  auto* in_ptr = input.data_ptr<float>();
  
  // Optimization 1: Larger work-group (1024 vs baseline 512)
  const int wg_size = 1024;
  
  // Optimization 2: vec_size=8 if aligned, else 4
  const int vec_size = (d % 8 == 0) ? 8 : 4;
  
  sycl::range<1> global_range(num_tokens * wg_size);
  sycl::range<1> local_range(wg_size);
  sycl::nd_range<1> ndr(global_range, local_range);
  
  if (vec_size == 8) {
    syclexp::nd_launch(q, ndr,
                       syclexp::kernel_function<silu_and_mul_kernel_vec_optimized<8>>,
                       out_ptr, in_ptr, d);
  } else {
    syclexp::nd_launch(q, ndr,
                       syclexp::kernel_function<silu_and_mul_kernel_vec_optimized<4>>,
                       out_ptr, in_ptr, d);
  }
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &silu_and_mul_xpu_optimized, 
        "Optimized SiLU-and-Mul (wg_size=1024, vec=8)");
}

TORCH_LIBRARY(silu_and_mul_xpu_opt, m) {
  m.def("silu_and_mul(Tensor input) -> Tensor", &silu_and_mul_xpu_optimized);
}
