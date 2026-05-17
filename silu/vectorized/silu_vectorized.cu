// silu_vectorized.cu
//
// Standalone (no PyTorch) CUDA SiLU kernel that mirrors the structure of
// upstream PyTorch's `vectorized_elementwise_kernel`
// (pytorch/aten/src/ATen/native/cuda/CUDALoops.cuh).
//
//   y = x * sigmoid(x) = x / (1 + exp(-x))
//
// The interesting bits — the ones the side-by-side SYCL file mirrors:
//   - Each block processes a contiguous tile of `vec_size * blockDim.x` elems.
//   - Fast path (full tile):  do `thread_work_size = vec_size` aligned vector
//                             load / op / store via float4 reinterpret_cast.
//   - Tail path (partial tile): scalar unrolled loop with bounds check.
//
// Build:  nvcc -O3 -arch=sm_80 silu_vectorized.cu -o silu_vec_cuda

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

// ---- Compile-time knobs (the µarch tuning) ---------------------------------
constexpr int kBlockSize  = 128;                     // num_threads()
constexpr int kVecSize    = 4;                       // 16 / sizeof(float)
constexpr int kBlockWork  = kVecSize * kBlockSize;   // io_block_work_size

// ---- Element op (same line as PyTorch upstream functor) --------------------
__device__ __forceinline__ float silu_op(float v) {
    return v / (1.0f + __expf(-v));
}

// ---- The kernel: same two-branch shape as PyTorch vectorized_elementwise_kernel
template <int VEC>
__global__ void silu_vectorized_kernel(const float* __restrict__ x,
                                       float*       __restrict__ y,
                                       int N) {
    int tile_base = blockIdx.x * kBlockWork;
    int remaining = N - tile_base;

    if (remaining >= kBlockWork) {
        // -------- Fast path: aligned vector load/store ----------------------
        // Each thread owns VEC consecutive elements within the tile, located
        // at (tile_base + threadIdx.x * VEC + 0..VEC-1).
        int  base = tile_base + threadIdx.x * VEC;
        const float4* xv = reinterpret_cast<const float4*>(x + base);
        float4* yv       = reinterpret_cast<float4*>(y + base);
        float4 in  = *xv;
        float4 out;
        out.x = silu_op(in.x);
        out.y = silu_op(in.y);
        out.z = silu_op(in.z);
        out.w = silu_op(in.w);
        *yv = out;
    } else {
        // -------- Tail path: scalar unroll with bounds check ----------------
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int i = tile_base + threadIdx.x + j * blockDim.x;
            if (i < N) y[i] = silu_op(x[i]);
        }
    }
}

// ---- Host driver -----------------------------------------------------------
int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) h_x[i] = (i % 200 - 100) * 0.01f;

    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);

    int grid = (N + kBlockWork - 1) / kBlockWork;
    silu_vectorized_kernel<kVecSize><<<grid, kBlockSize>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    float ref = h_x[12345] / (1.0f + std::exp(-h_x[12345]));
    std::printf("CUDA vec  SiLU[12345] = %.6f   (ref %.6f)\n", h_y[12345], ref);

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
