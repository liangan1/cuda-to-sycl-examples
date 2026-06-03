#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <cmath>
#include <cstdio>

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

int main() {
  constexpr int tokens = 4;
  constexpr int d = 128;
  constexpr int in_size = tokens * 2 * d;
  constexpr int out_size = tokens * d;

  sycl::queue q{sycl::gpu_selector_v};
  float* h_in = new float[in_size];
  float* h_out = new float[out_size];
  for (int i = 0; i < in_size; ++i) h_in[i] = 0.01f * (i % 97);

  float* d_in = sycl::malloc_device<float>(in_size, q);
  float* d_out = sycl::malloc_device<float>(out_size, q);
  q.memcpy(d_in, h_in, sizeof(float) * in_size).wait();

  constexpr int block = 512;
  sycl::nd_range<1> ndr{tokens * block, block};
  syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_and_mul_kernel<4>>, d_out, d_in, d);
  q.wait();

  q.memcpy(h_out, d_out, sizeof(float) * out_size).wait();
  printf("SYCL standalone ok, sample=%f\n", h_out[0]);

  sycl::free(d_in, q);
  sycl::free(d_out, q);
  delete[] h_in;
  delete[] h_out;
  return 0;
}
