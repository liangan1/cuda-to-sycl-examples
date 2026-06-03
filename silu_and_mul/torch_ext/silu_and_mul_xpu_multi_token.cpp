// silu_and_mul_xpu_multi_token.cpp
//
// Optimization: Multi-tokens per work-group
// - Each work-group processes TOKENS_PER_WG tokens (e.g. 4)
// - Increase work per WG to hide memory latency
// - Better occupancy and resource utilization
//
// Config: WG_SIZE=512, VEC_SIZE=4, TOKENS_PER_WG=4

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

template <int VEC_SIZE, int TOKENS_PER_WG>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_and_mul_kernel_multi_token(
    float* __restrict__ out,
    const float* __restrict__ input,
    int num_tokens,
    int d)
{
  auto item = syclwi::get_nd_item<1>();
  
  const int wg_id = item.get_group(0);
  const int tid = item.get_local_id(0);
  const int block_size = item.get_local_range(0);
  
  // Each work-group processes TOKENS_PER_WG tokens
  const int base_token_idx = wg_id * TOKENS_PER_WG;
  
  using vec_t = sycl::vec<float, VEC_SIZE>;
  const int num_vec = d / VEC_SIZE;
  
  // Process multiple tokens
  #pragma unroll
  for (int t = 0; t < TOKENS_PER_WG; ++t) {
    const int token_idx = base_token_idx + t;
    if (token_idx >= num_tokens) break;
    
    const float* gate = input + token_idx * 2 * d;
    const float* up = gate + d;
    float* out_row = out + token_idx * d;
    
    // Vectorized loop for this token
    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
      const int base = vec_idx * VEC_SIZE;
      
      vec_t gate_vec, up_vec, out_vec;
      
      gate_vec.load(0, sycl::multi_ptr<const float,
                        sycl::access::address_space::global_space>(gate + base));
      up_vec.load(0, sycl::multi_ptr<const float,
                      sycl::access::address_space::global_space>(up + base));
      
      #pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float g = gate_vec[i];
        float u = up_vec[i];
        out_vec[i] = (g / (1.0f + sycl::exp(-g))) * u;
      }
      
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
}

torch::Tensor silu_and_mul_xpu_optimized(torch::Tensor input) {
  TORCH_CHECK(input.is_xpu(), "Input must be on XPU device");
  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "Input must be 2D");
  TORCH_CHECK(input.size(1) % 2 == 0, "Input dim 1 must be even");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "Only FP32 supported");
  
  const int num_tokens = input.size(0);
  const int d = input.size(1) / 2;
  
  auto output = torch::empty({num_tokens, d}, input.options());
  
  c10::xpu::XPUStream xpu_stream = c10::xpu::getCurrentXPUStream(input.device().index());
  sycl::queue& q = xpu_stream.queue();
  
  auto* out_ptr = output.data_ptr<float>();
  auto* in_ptr = input.data_ptr<float>();
  
  // Multi-token configuration
  const int WG_SIZE = 512;
  const int TOKENS_PER_WG = 4;  // Each WG processes 4 tokens
  const int num_wg = (num_tokens + TOKENS_PER_WG - 1) / TOKENS_PER_WG;
  
  sycl::range<1> global_range(num_wg * WG_SIZE);
  sycl::range<1> local_range(WG_SIZE);
  sycl::nd_range<1> ndr(global_range, local_range);
  
  syclexp::nd_launch(q, ndr,
                     syclexp::kernel_function<silu_and_mul_kernel_multi_token<4, 4>>,
                     out_ptr, in_ptr, num_tokens, d);
  
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &silu_and_mul_xpu_optimized,
        "SiLU-and-Mul with multi-token per WG");
}

TORCH_LIBRARY(silu_and_mul_xpu_opt, m) {
  m.def("silu_and_mul(Tensor input) -> Tensor", &silu_and_mul_xpu_optimized);
}
