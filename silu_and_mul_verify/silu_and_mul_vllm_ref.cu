// vLLM reference style CUDA kernel (for mapping reference only)
// Source: https://github.com/vllm-project/vllm (activation_kernels.cu)
// License: Apache-2.0

#include <cuda_runtime.h>
#include <cmath>

template <typename T>
__device__ __forceinline__ T silu_cuda(T x) {
  return x / (T(1) + expf(-x));
}

template <typename T, int VEC_SIZE>
__global__ void silu_and_mul_kernel_vllm_ref(
    T* __restrict__ out,
    const T* __restrict__ input,
    int d) {
  int token = blockIdx.x;
  int tid = threadIdx.x;

  const T* gate = input + token * 2 * d;
  const T* up = gate + d;
  T* out_ptr = out + token * d;

  int num_vec = d / VEC_SIZE;
  using vec_t = float4;

  for (int vec_idx = tid; vec_idx < num_vec; vec_idx += blockDim.x) {
    int base = vec_idx * VEC_SIZE;
    vec_t g = *reinterpret_cast<const vec_t*>(gate + base);
    vec_t u = *reinterpret_cast<const vec_t*>(up + base);
    vec_t o;
    o.x = silu_cuda(g.x) * u.x;
    o.y = silu_cuda(g.y) * u.y;
    o.z = silu_cuda(g.z) * u.z;
    o.w = silu_cuda(g.w) * u.w;
    *reinterpret_cast<vec_t*>(out_ptr + base) = o;
  }

  for (int i = num_vec * VEC_SIZE + tid; i < d; i += blockDim.x) {
    out_ptr[i] = silu_cuda(gate[i]) * up[i];
  }
}
