// SiLU (Swish) activation:  y = x * sigmoid(x) = x / (1 + exp(-x))
// SYCL free-function kernel (sycl_ext_oneapi_free_function_kernels).
//
// Build:  icpx -fsycl -O3 -fsycl-targets=spir64 silu.sycl.cpp -o silu_sycl
//
// Concept mapping vs CUDA:
//   __global__              ->  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(nd_range_kernel<1>)
//   blockIdx/threadIdx      ->  this_work_item::get_nd_item<1>().get_global_id(0)
//   <<<grid, block>>>       ->  q.parallel_for(nd_range, kernel_id)  (or launch helper)
//   cudaMalloc/cudaMemcpy   ->  sycl::malloc_device / q.memcpy
//   cudaDeviceSynchronize   ->  q.wait()

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <cstdio>
#include <vector>
#include <cmath>

namespace syclex = sycl::ext::oneapi::experimental;

// ---- Device kernel: free function (same shape as CUDA __global__) ----------
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclex::nd_range_kernel<1>))
void silu_kernel(const float* x, float* y, int n) {
    auto it = syclex::this_work_item::get_nd_item<1>();
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
    // Resolve the free-function kernel by its address, then enqueue it.
    auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
        q.get_context());
    sycl::kernel k = syclex::get_kernel<silu_kernel>(bundle);

    q.submit([&](sycl::handler& h) {
        h.set_args(d_x, d_y, N);
        h.parallel_for(ndr, k);
    }).wait();

    q.memcpy(h_y.data(), d_y, bytes).wait();

    float ref = h_x[12345] / (1.0f + std::exp(-h_x[12345]));
    std::printf("SYCL  SiLU[12345] = %.6f   (ref %.6f)\n", h_y[12345], ref);

    sycl::free(d_x, q);
    sycl::free(d_y, q);
    return 0;
}
