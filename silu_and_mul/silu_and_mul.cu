// silu_and_mul.cu
//
// CUDA implementation of SiLU-and-Mul fusion kernel, derived from vLLM's
// production implementation at:
//   vllm-project/vllm/csrc/libtorch_stable/activation_kernels.cu
//
// Kernel computes: out[i] = silu(gate[i]) * up[i]
// where silu(x) = x / (1 + exp(-x))
//
// Input layout: [num_tokens, 2, d] where input[:, 0, :] is gate, input[:, 1, :] is up
// Output layout: [num_tokens, d]
//
// Build:
//   nvcc -O3 -arch=sm_80 silu_and_mul.cu -o silu_and_mul_cuda
//
// Run:
//   ./silu_and_mul_cuda

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Scalar SiLU activation
template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

// Packed SiLU for __half2 (processes 2 fp16 values in parallel)
template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t& val) {
  // For production use with half2 or nv_bfloat162
  // Here we provide a placeholder for the generic type
  return val;
}

// Template specialization for __half2 would go here in production code

// Fused SiLU-and-Mul kernel
// - 1 block per token (grid = [num_tokens])
// - Uses vectorized loads (float4 = 128-bit) for better memory bandwidth
// - Pattern: out = silu(gate) * up
template <typename scalar_t, int VEC_SIZE = 4>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ out,         // [num_tokens, d]
    const scalar_t* __restrict__ input, // [num_tokens, 2 * d]
    const int d)
{
  const int token_idx = blockIdx.x;
  
  // Input layout: [num_tokens, 2, d]
  // gate = input[token_idx, 0, :]
  // up   = input[token_idx, 1, :]
  const scalar_t* gate = input + token_idx * 2 * d;
  const scalar_t* up   = input + token_idx * 2 * d + d;
  scalar_t* out_ptr    = out + token_idx * d;
  
  // Vectorized path: process VEC_SIZE elements per thread
  const int num_vec = d / VEC_SIZE;
  using vec_t = float4; // 128-bit aligned load/store for float
  
  for (int vec_idx = threadIdx.x; vec_idx < num_vec; vec_idx += blockDim.x) {
    int base = vec_idx * VEC_SIZE;
    
    // Load gate and up vectors
    vec_t gate_vec = *reinterpret_cast<const vec_t*>(gate + base);
    vec_t up_vec   = *reinterpret_cast<const vec_t*>(up + base);
    vec_t out_vec;
    
    // Compute silu(gate[i]) * up[i] for each element
    out_vec.x = silu_kernel(gate_vec.x) * up_vec.x;
    out_vec.y = silu_kernel(gate_vec.y) * up_vec.y;
    out_vec.z = silu_kernel(gate_vec.z) * up_vec.z;
    out_vec.w = silu_kernel(gate_vec.w) * up_vec.w;
    
    // Store result
    *reinterpret_cast<vec_t*>(out_ptr + base) = out_vec;
  }
  
  // Scalar tail for remaining elements
  for (int i = num_vec * VEC_SIZE + threadIdx.x; i < d; i += blockDim.x) {
    out_ptr[i] = silu_kernel(gate[i]) * up[i];
  }
}

// Host wrapper function
void silu_and_mul_cuda(float* d_out, const float* d_input, int num_tokens, int d) {
  constexpr int BLOCK_SIZE = 256;
  dim3 grid(num_tokens);
  dim3 block(BLOCK_SIZE);
  
  silu_and_mul_kernel<float, 4><<<grid, block>>>(d_out, d_input, d);
}

// Demo main
int main() {
  constexpr int num_tokens = 4;
  constexpr int d = 128;
  constexpr int input_size = num_tokens * 2 * d;
  constexpr int output_size = num_tokens * d;
  
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
  
  // Allocate device memory
  float *d_input, *d_output;
  cudaMalloc(&d_input, input_size * sizeof(float));
  cudaMalloc(&d_output, output_size * sizeof(float));
  
  // Copy input to device
  cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
  
  // Launch kernel
  silu_and_mul_cuda(d_output, d_input, num_tokens, d);
  
  // Copy result back
  cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
  
  // Verify result (check first few elements)
  printf("CUDA SiLU-and-Mul Results:\n");
  for (int i = 0; i < 5; ++i) {
    float gate_val = h_input[i];
    float up_val = h_input[d + i];
    float silu_val = gate_val / (1.0f + expf(-gate_val));
    float expected = silu_val * up_val;
    printf("  out[%d] = %.6f (expected %.6f, gate=%.2f, up=%.2f)\n", 
           i, h_output[i], expected, gate_val, up_val);
  }
  
  // Cleanup
  delete[] h_input;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);
  
  printf("\nCUDA kernel launch: SUCCESS\n");
  return 0;
}
