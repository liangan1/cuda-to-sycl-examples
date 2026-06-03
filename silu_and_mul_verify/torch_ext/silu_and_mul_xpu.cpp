#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi = sycl::ext::oneapi::this_work_item;

inline float silu_sycl(float x) {
  return x / (1.0f + sycl::exp(-x));
}

template <int VEC_SIZE>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_and_mul_kernel(float* out, const float* input, int d) {
  auto item = syclwi::get_nd_item<1>();
  const int token = static_cast<int>(item.get_group(0));
  const int tid = static_cast<int>(item.get_local_id(0));
  const int block = static_cast<int>(item.get_local_range(0));

  const float* gate = input + token * 2 * d;
  const float* up = gate + d;
  float* out_ptr = out + token * d;

  const int num_vec = d / VEC_SIZE;
  using vec_t = sycl::vec<float, VEC_SIZE>;

  for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block) {
    const int base = vec_idx * VEC_SIZE;
    vec_t g;
    vec_t u;
    vec_t o;
    g.load(0, sycl::multi_ptr<const float, sycl::access::address_space::global_space>(gate + base));
    u.load(0, sycl::multi_ptr<const float, sycl::access::address_space::global_space>(up + base));
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      o[i] = silu_sycl(g[i]) * u[i];
    }
    o.store(0, sycl::multi_ptr<float, sycl::access::address_space::global_space>(out_ptr + base));
  }

  for (int i = num_vec * VEC_SIZE + tid; i < d; i += block) {
    out_ptr[i] = silu_sycl(gate[i]) * up[i];
  }
}

torch::Tensor silu_and_mul_xpu(torch::Tensor input) {
  TORCH_CHECK(input.is_xpu(), "input must be xpu");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "float32 only");
  TORCH_CHECK(input.dim() == 2, "input shape [tokens, 2*d]");
  TORCH_CHECK(input.size(1) % 2 == 0, "input second dim must be even");

  const int64_t tokens = input.size(0);
  const int64_t d = input.size(1) / 2;
  auto out = torch::empty({tokens, d}, input.options());

  c10::xpu::XPUStream s = c10::xpu::getCurrentXPUStream(input.device().index());
  sycl::queue& q = s.queue();

  constexpr int block = 512;
  sycl::nd_range<1> ndr{static_cast<size_t>(tokens * block), static_cast<size_t>(block)};
  syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_and_mul_kernel<4>>, out.data_ptr<float>(), input.data_ptr<float>(), static_cast<int>(d));

  // no q.wait(); rely on PyTorch stream semantics
  return out;
}

TORCH_LIBRARY(silu_and_mul_verify_xpu, m) {
  m.def("silu_and_mul(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(silu_and_mul_verify_xpu, XPU, m) {
  m.impl("silu_and_mul", &silu_and_mul_xpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_and_mul", &silu_and_mul_xpu, "silu_and_mul XPU verify op");
}
