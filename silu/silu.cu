// SiLU (Swish) activation:  y = x * sigmoid(x) = x / (1 + exp(-x))
// CUDA free-function kernel.
//
// Build:  nvcc -O3 -arch=sm_80 silu.cu -o silu_cuda

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

// ---- Device kernel: free function marked __global__ ------------------------
__global__ void silu_kernel(const float* __restrict__ x,
                            float*       __restrict__ y,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
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

    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    // ---- Kernel launch (triple-chevron syntax) -----------------------------
    // This is sugar; the CUDA runtime also exposes the explicit C API
    // cudaLaunchKernel, which is what `<<<>>>` lowers to. It is the direct
    // structural counterpart of SYCL's `nd_launch` (a plain C++ function call
    // taking the kernel address + args). Equivalent code:
    //
    //   void* args[] = { (void*)&d_x, (void*)&d_y, (void*)&N };
    //   cudaLaunchKernel((const void*)silu_kernel,
    //                    dim3(grid), dim3(BLOCK),
    //                    args, /*sharedMem=*/0, /*stream=*/0);
    //
    // See: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c
    silu_kernel<<<grid, BLOCK>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    // Spot check
    float ref = h_x[12345] / (1.0f + std::exp(-h_x[12345]));
    std::printf("CUDA  SiLU[12345] = %.6f   (ref %.6f)\n", h_y[12345], ref);

    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
