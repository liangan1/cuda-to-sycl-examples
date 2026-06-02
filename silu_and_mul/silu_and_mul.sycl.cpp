// silu_and_mul.sycl.cpp
//
// SYCL implementation of SiLU-and-Mul fusion kernel using free-function kernel
// style, mirroring the CUDA implementation from vLLM.
//
// Kernel computes: out[i] = silu(gate[i]) * up[i]
// where silu(x) = x / (1 + exp(-x))
//
// Input layout: [num_tokens, 2, d] where input[:, 0, :] is gate, input[:, 1, :] is up
// Output layout: [num_tokens, d]
//
// Build:
//   icpx -fsycl -O3 -fsycl-targets=spir64 silu_and_mul.sycl.cpp -o silu_and_mul_sycl
//
// Run:
//   ./silu_and_mul_sycl

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <cstdio>
#include <cmath>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

// ---------------------------------------------------------------------------
// Kernel implementation (free-function style)
// ---------------------------------------------------------------------------

// Scalar SiLU activation
// CUDA:  return (T)(((float)x) / (1.0f + expf((float)-x)));
// SYCL:  return x / (1.0f + sycl::exp(-x));
inline float silu_op(float x) {
  return x / (1.0f + sycl::exp(-x));
}

// Fused SiLU-and-Mul kernel (SYCL free-function kernel)
// Semantic mapping to CUDA:
//   - CUDA: __global__ void kernel(...)
//   - SYCL: SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)) void kernel(...)
//
//   - CUDA: blockIdx.x
//   - SYCL: item.get_group(0)
//
//   - CUDA: threadIdx.x
//   - SYCL: item.get_local_id(0)
//
//   - CUDA: blockDim.x
//   - SYCL: item.get_local_range(0)
//
//   - CUDA: float4 (128-bit vectorized load/store)
//   - SYCL: sycl::vec<float, 4> with .load() and .store()
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_and_mul_kernel(
    float* __restrict__ out,         // [num_tokens, d]
    const float* __restrict__ input, // [num_tokens, 2 * d]
    int d)
{
  auto item = syclwi::get_nd_item<1>();
  
  // CUDA: const int token_idx = blockIdx.x;
  const int token_idx = item.get_group(0);
  
  // CUDA: const int tid = threadIdx.x;
  const int tid = item.get_local_id(0);
  
  // CUDA: const int block_size = blockDim.x;
  const int block_size = item.get_local_range(0);
  
  // Input layout: [num_tokens, 2, d]
  // gate = input[token_idx, 0, :]
  // up   = input[token_idx, 1, :]
  const float* gate = input + token_idx * 2 * d;
  const float* up   = input + token_idx * 2 * d + d;
  float* out_ptr    = out + token_idx * d;
  
  // Vectorized path: process 4 elements per thread using sycl::vec<float, 4>
  constexpr int VEC_SIZE = 4;
  const int num_vec = d / VEC_SIZE;
  
  using vec_t = sycl::vec<float, VEC_SIZE>;
  
  for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
    int base = vec_idx * VEC_SIZE;
    
    // CUDA: vec_t gate_vec = *reinterpret_cast<const vec_t*>(gate + base);
    // SYCL: vec_t gate_vec; gate_vec.load(0, multi_ptr(...));
    vec_t gate_vec, up_vec, out_vec;
    
    gate_vec.load(0, sycl::multi_ptr<const float, 
                      sycl::access::address_space::global_space>(gate + base));
    up_vec.load(0, sycl::multi_ptr<const float, 
                    sycl::access::address_space::global_space>(up + base));
    
    // Compute silu(gate[i]) * up[i] for each element
    // CUDA uses .x, .y, .z, .w
    // SYCL uses [0], [1], [2], [3]
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      out_vec[i] = silu_op(gate_vec[i]) * up_vec[i];
    }
    
    // Store result
    out_vec.store(0, sycl::multi_ptr<float, 
                     sycl::access::address_space::global_space>(out_ptr + base));
  }
  
  // Scalar tail for remaining elements
  for (int i = num_vec * VEC_SIZE + tid; i < d; i += block_size) {
    out_ptr[i] = silu_op(gate[i]) * up[i];
  }
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------

void silu_and_mul_sycl(sycl::queue& q, float* d_out, const float* d_input, 
                       int num_tokens, int d) {
  constexpr int BLOCK_SIZE = 256;
  
  // CUDA launch config: <<<num_tokens, BLOCK_SIZE>>>
  // SYCL nd_range: global_size = num_tokens * BLOCK_SIZE, local_size = BLOCK_SIZE
  sycl::nd_range<1> ndr{num_tokens * BLOCK_SIZE, BLOCK_SIZE};
  
  // CUDA: silu_and_mul_kernel<<<grid, block>>>(d_out, d_input, d);
  // SYCL: nd_launch with free-function kernel
  syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_and_mul_kernel>,
                     d_out, d_input, d);
}

// ---------------------------------------------------------------------------
// Demo main
// ---------------------------------------------------------------------------

int main() {
  constexpr int num_tokens = 4;
  constexpr int d = 128;
  constexpr int input_size = num_tokens * 2 * d;
  constexpr int output_size = num_tokens * d;
  
  // Create SYCL queue
  sycl::queue q{sycl::gpu_selector_v, 
                sycl::property::queue::in_order()};
  
  printf("Running on: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  
  // Allocate host memory
  float* h_input = new float[input_size];
  float* h_output = new float[output_size];
  
  // Initialize input: gate and up with simple values
  for (int t = 0; t < num_tokens; ++t) {
    for (int i = 0; i < d; ++i) {
      h_input[t * 2 * d + i] = (i % 10) * 0.1f;         // gate
      h_input[t * 2 * d + d + i] = ((i + 5) % 10) * 0.1f; // up
    }
  }
  
  // CUDA: cudaMalloc(&d_input, size);
  // SYCL: sycl::malloc_device<T>(count, queue)
  float* d_input = sycl::malloc_device<float>(input_size, q);
  float* d_output = sycl::malloc_device<float>(output_size, q);
  
  // CUDA: cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
  // SYCL: q.memcpy(d_input, h_input, bytes).wait();
  q.memcpy(d_input, h_input, input_size * sizeof(float)).wait();
  
  // Launch kernel
  silu_and_mul_sycl(q, d_output, d_input, num_tokens, d);
  
  // CUDA: cudaDeviceSynchronize();
  // SYCL: q.wait();
  q.wait();
  
  // Copy result back
  q.memcpy(h_output, d_output, output_size * sizeof(float)).wait();
  
  // Verify result (check first few elements)
  printf("\nSYCL SiLU-and-Mul Results:\n");
  for (int i = 0; i < 5; ++i) {
    float gate_val = h_input[i];
    float up_val = h_input[d + i];
    float silu_val = gate_val / (1.0f + std::exp(-gate_val));
    float expected = silu_val * up_val;
    printf("  out[%d] = %.6f (expected %.6f, gate=%.2f, up=%.2f)\n", 
           i, h_output[i], expected, gate_val, up_val);
  }
  
  // CUDA: cudaFree(d_input);
  // SYCL: sycl::free(d_input, q);
  delete[] h_input;
  delete[] h_output;
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  
  printf("\nSYCL kernel launch: SUCCESS\n");
  return 0;
}
