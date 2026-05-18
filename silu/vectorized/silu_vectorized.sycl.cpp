// silu_vectorized.sycl.cpp
//
// Standalone (no PyTorch) SYCL SiLU kernel mirroring PyTorch CUDA's
// `vectorized_elementwise_kernel` (CUDALoops.cuh). Same two-branch shape
// (fast vectorized path + scalar tail), `float4` ↔ `sycl::vec<float, 4>`.
//
// SYCL kernel form: **free-function kernel** (sycl_ext_oneapi_free_function_kernels),
// to stay symmetric with silu.sycl.cpp (step 1) — the kernel is a plain top-level
// function marked with SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)),
// just like the CUDA `__global__` function it ports.
//
//   y = x * sigmoid(x) = x / (1 + exp(-x))
//
// Build:
//   icpx -fsycl -O3 -fsycl-targets=spir64 silu_vectorized.sycl.cpp \
//        -o silu_vec_sycl

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <cstdio>
#include <vector>
#include <cmath>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

// ---- Compile-time knobs (the µarch tuning) ---------------------------------
//   CUDA picks kBlockSize ≈ 128 (warps × occupancy).
//   Xe2 typically picks `syclMaxWorkItemsPerSubSlice()`; 256 is a reasonable
//   portable default for a demo.
constexpr int kWgSize    = 256;                  // ↔ blockDim.x
constexpr int kVecSize   = 4;                    // 16 / sizeof(float) — same as CUDA float4
constexpr int kBlockWork = kVecSize * kWgSize;   // ↔ io_block_work_size

// ---- Element op (same line as PyTorch upstream functor) --------------------
static inline float silu_op(float v) {
    return v / (1.0f + sycl::exp(-v));
}

// ---- The kernel: SYCL free function mirroring vectorized_elementwise_kernel
// Same shape as CUDA __global__: top-level function, captures via parameters,
// gets work-item coordinates from `this_work_item::get_nd_item<1>()`.
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_vectorized_kernel(const float* x, float* y, int N) {
    auto item = syclwi::get_nd_item<1>();
    int grpid = item.get_group(0);             // ↔ blockIdx.x
    int lid   = item.get_local_id(0);          // ↔ threadIdx.x
    int wgsz  = item.get_local_range(0);       // ↔ blockDim.x

    int tile_base = grpid * kBlockWork;
    int remaining = N - tile_base;

    if (remaining >= kBlockWork) {
        // -------- Fast path: aligned vector load/store ----------------------
        // Each work-item owns kVecSize consecutive elements in the tile.
        // sycl::vec<float,4> lowers to a 16-B SEND on Xe2, the same shape as
        // CUDA's float4 → 128-bit ld.global.
        int base = tile_base + lid * kVecSize;
        using vec_t = sycl::vec<float, kVecSize>;
        vec_t in;
        in.load(0, sycl::multi_ptr<const float,
                    sycl::access::address_space::global_space>(x + base));
        vec_t out;
        #pragma unroll
        for (int j = 0; j < kVecSize; ++j) out[j] = silu_op(in[j]);
        out.store(0, sycl::multi_ptr<float,
                     sycl::access::address_space::global_space>(y + base));
    } else {
        // -------- Tail path: scalar unroll with bounds check ----------------
        #pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
            int i = tile_base + lid + j * wgsz;
            if (i < N) y[i] = silu_op(x[i]);
        }
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

    int num_wg = (N + kBlockWork - 1) / kBlockWork;
    sycl::nd_range<1> ndr{sycl::range<1>(size_t(num_wg) * kWgSize),
                          sycl::range<1>(kWgSize)};

    // ---- Kernel launch -----------------------------------------------------
    // Free-function nd_launch: directly enqueue the kernel function with
    // its arguments — the structural counterpart of CUDA's explicit runtime
    // launch API `cudaLaunchKernel(funcAddr, gridDim, blockDim, args, smem,
    // stream)`. (CUDA also offers the `<<<grid, block>>>` sugar that lowers
    // to cudaLaunchKernel; SYCL has no such sugar — nd_launch *is* the API.)
    // CUDA reference:
    //   https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g5064cdf5d8e6741ace56fd8be951783c
    syclexp::nd_launch(q, ndr,
                       syclexp::kernel_function<silu_vectorized_kernel>,
                       d_x, d_y, N);
    q.wait();

    q.memcpy(h_y.data(), d_y, bytes).wait();

    float ref = h_x[12345] / (1.0f + std::exp(-h_x[12345]));
    std::printf("SYCL vec  SiLU[12345] = %.6f   (ref %.6f)\n",
                h_y[12345], ref);

    sycl::free(d_x, q);
    sycl::free(d_y, q);
    return 0;
}
