// SiLU (Swish) activation:  y = x * sigmoid(x) = x / (1 + exp(-x))
// SYCL free-function kernel (sycl_ext_oneapi_free_function_kernels).
//
// Build:  icpx -fsycl -O3 -fsycl-targets=spir64 silu.sycl.cpp -o silu_sycl
//
// Concept mapping vs CUDA:
//   __global__              ->  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(nd_range_kernel<1>)
//   blockIdx/threadIdx      ->  this_work_item::get_nd_item<1>().get_global_id(0)
//   <<<grid, block>>>       ->  syclexp::nd_launch(q, ndr, kernel_function<silu_kernel>, args...)
//   cudaMalloc/cudaMemcpy   ->  sycl::malloc_device / q.memcpy
//   cudaDeviceSynchronize   ->  q.wait()

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <cstdio>
#include <vector>
#include <cmath>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

// ---- Device kernel: free function (same shape as CUDA __global__) ----------
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_kernel(const float* x, float* y, int n) {
    auto it = syclwi::get_nd_item<1>();
    int i = it.get_global_id(0);
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + sycl::exp(-v));
    }
}

// ---- Host driver -----------------------------------------------------------
int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    sycl::queue q{sycl::default_selector_v};
    std::printf("Device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<float> h_x(N), h_y(N);
    for (int i = 0; i < N; ++i) h_x[i] = (i % 200 - 100) * 0.01f;

    float* d_x = sycl::malloc_device<float>(N, q);
    float* d_y = sycl::malloc_device<float>(N, q);
    q.memcpy(d_x, h_x.data(), bytes).wait();

    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    sycl::nd_range<1> ndr{sycl::range<1>(grid * BLOCK), sycl::range<1>(BLOCK)};

    // ---- Kernel launch -----------------------------------------------------
    // Free-function nd_launch: pass the queue, nd_range, the
    // kernel_function<silu_kernel> tag, and the kernel arguments by value.
    //
    // Structurally this matches CUDA's explicit runtime launch API
    // cudaLaunchKernel(funcAddr, gridDim, blockDim, args, smem, stream)
    // — a plain C++ function call taking the kernel address + arg pack.
    // (CUDA also offers the `<<<grid, block>>>` sugar that lowers to
    // cudaLaunchKernel; SYCL has no such sugar — nd_launch *is* the API.)
    // CUDA reference:
    //   https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c
    syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_kernel>, d_x, d_y, N);
    q.wait();

    q.memcpy(h_y.data(), d_y, bytes).wait();

    float ref = h_x[12345] / (1.0f + std::exp(-h_x[12345]));
    std::printf("SYCL  SiLU[12345] = %.6f   (ref %.6f)\n", h_y[12345], ref);

    sycl::free(d_x, q);
    sycl::free(d_y, q);
    return 0;
}
