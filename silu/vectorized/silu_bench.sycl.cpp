// silu_bench.sycl.cpp
//
// Standalone micro-benchmark for SiLU on Intel XPU, comparing the naive
// one-element-per-work-item kernel against the vectorized (sycl::vec<float,4>
// + tile-per-workgroup + scalar-tail) kernel.
//
// Both kernels are SYCL **free-function kernels**
// (sycl_ext_oneapi_free_function_kernels), to stay symmetric with
// silu.sycl.cpp (step 1) and silu_vectorized.sycl.cpp (step 3) — each kernel
// is a plain top-level function marked with
// SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)), the direct
// counterpart of a CUDA `__global__`.
//
// Methodology follows the standard memory-bound bench used by tools like
// bytedance/xpu-perf:
//   - allocate input/output device buffers
//   - warmup iterations (kernel compile, cache warmup, queue priming)
//   - N timed iterations using SYCL event profiling (start..end on the
//     command_submit / start / end timestamps)
//   - report min / median / mean kernel time
//   - effective bandwidth = (bytes_read + bytes_written) / time
//   - report % of measured HBM peak (configurable via --peak-gbps)
//
// Build:
//   icpx -fsycl -O3 -fsycl-targets=spir64 silu_bench.sycl.cpp -o silu_bench
//
// Run:
//   ./silu_bench                           # default sizes
//   ./silu_bench --iters 100 --warmup 20
//   ./silu_bench --peak-gbps 456           # BMG-G31 measured peak
//   ./silu_bench --size 1048576            # single size

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

// ---------------------------------------------------------------------------
// Kernels (free-function form — same shape as CUDA __global__)
// Copies of silu.sycl.cpp + silu_vectorized.sycl.cpp, inlined here so the
// bench is single-file. See those files for the annotated originals.
// ---------------------------------------------------------------------------

static inline float silu_op(float v) {
    return v / (1.0f + sycl::exp(-v));
}

// Naive: one element per work-item, no vectorization, no tile awareness.
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_naive_kernel(const float* x, float* y, int N) {
    auto item = syclwi::get_nd_item<1>();
    int i = item.get_global_id(0);
    if (i < N) y[i] = silu_op(x[i]);
}

// Vectorized: tile-per-workgroup, sycl::vec<float,4> aligned load/store,
// scalar tail. Mirror of PyTorch CUDA's vectorized_elementwise_kernel.
constexpr int kVecSize    = 4;
constexpr int kWgSizeVec  = 256;
constexpr int kBlockWork  = kVecSize * kWgSizeVec;

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_vec_kernel(const float* x, float* y, int N) {
    auto item = syclwi::get_nd_item<1>();
    int grpid = item.get_group(0);
    int lid   = item.get_local_id(0);
    int wgsz  = item.get_local_range(0);

    int tile_base = grpid * kBlockWork;
    int remaining = N - tile_base;

    if (remaining >= kBlockWork) {
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
        #pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
            int i = tile_base + lid + j * wgsz;
            if (i < N) y[i] = silu_op(x[i]);
        }
    }
}

// Trivial init kernel — fill x with (i%200-100)*0.01f. Also a free function
// for consistency.
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void init_kernel(float* x, int N) {
    auto item = syclwi::get_nd_item<1>();
    int i = item.get_global_id(0);
    if (i < N) x[i] = ((i % 200) - 100) * 0.01f;
}

// ---------------------------------------------------------------------------
// Bench harness
// ---------------------------------------------------------------------------

struct BenchStats {
    double min_ms;
    double med_ms;
    double mean_ms;
};

// Run `iters` timed launches of `submit_fn`, return aggregate stats in ms.
// Uses SYCL event profiling (end - start, in ns).
template <typename SubmitFn>
static BenchStats time_kernel(sycl::queue& q, int warmup, int iters,
                              SubmitFn&& submit_fn) {
    // Warmup: same submission, but discard timings.
    for (int i = 0; i < warmup; ++i) submit_fn().wait();

    std::vector<double> ms; ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        sycl::event e = submit_fn();
        e.wait();
        auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        ms.push_back((t1 - t0) * 1e-6);  // ns -> ms
    }
    std::sort(ms.begin(), ms.end());
    double sum = 0; for (double v : ms) sum += v;
    return BenchStats{ ms.front(),
                       ms[ms.size() / 2],
                       sum / ms.size() };
}

static double gbps(int N, double ms) {
    // 1 fp32 read + 1 fp32 write per element
    double bytes = double(N) * 2 * sizeof(float);
    return bytes / (ms * 1e-3) / 1e9;
}

// ---------------------------------------------------------------------------
// Per-size run
// ---------------------------------------------------------------------------

struct RunResult {
    int          N;
    BenchStats   naive;
    BenchStats   vec;
};

static RunResult run_one_size(sycl::queue& q,
                              int N, int warmup, int iters) {
    float* d_x = sycl::malloc_device<float>(N, q);
    float* d_y = sycl::malloc_device<float>(N, q);

    // Init x via the init free-function kernel.
    {
        constexpr int kWgInit = 256;
        int grid = (N + kWgInit - 1) / kWgInit;
        sycl::nd_range<1> ndr{sycl::range<1>(size_t(grid) * kWgInit),
                              sycl::range<1>(kWgInit)};
        syclexp::nd_launch(q, ndr, syclexp::kernel_function<init_kernel>, d_x, N);
        q.wait();
    }

    // Naive launch: wrap nd_launch in q.submit() so we get a profiling event.
    constexpr int kWgSizeNaive = 256;
    int grid_naive = (N + kWgSizeNaive - 1) / kWgSizeNaive;
    sycl::nd_range<1> ndr_naive{
        sycl::range<1>(size_t(grid_naive) * kWgSizeNaive),
        sycl::range<1>(kWgSizeNaive)};
    auto submit_naive = [&]() {
        return q.submit([&](sycl::handler& h) {
            syclexp::nd_launch(h, ndr_naive,
                               syclexp::kernel_function<silu_naive_kernel>,
                               d_x, d_y, N);
        });
    };

    // Vectorized launch (same pattern).
    int num_wg = (N + kBlockWork - 1) / kBlockWork;
    sycl::nd_range<1> ndr_vec{
        sycl::range<1>(size_t(num_wg) * kWgSizeVec),
        sycl::range<1>(kWgSizeVec)};
    auto submit_vec = [&]() {
        return q.submit([&](sycl::handler& h) {
            syclexp::nd_launch(h, ndr_vec,
                               syclexp::kernel_function<silu_vec_kernel>,
                               d_x, d_y, N);
        });
    };

    BenchStats naive = time_kernel(q, warmup, iters, submit_naive);
    BenchStats vec   = time_kernel(q, warmup, iters, submit_vec);

    sycl::free(d_x, q);
    sycl::free(d_y, q);
    return RunResult{N, naive, vec};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Defaults
    int warmup = 10;
    int iters  = 50;
    double peak_gbps = 0.0;             // 0 -> don't report %peak
    std::vector<int> sizes = {
        1 << 16,   //  64 KiB elems = 256 KiB   (fits L1/L2)
        1 << 18,   // 256 KiB elems =   1 MiB   (LLC)
        1 << 20,   //   1 MiB elems =   4 MiB   (LLC edge)
        1 << 22,   //   4 MiB elems =  16 MiB   (HBM)
        1 << 24,   //  16 MiB elems =  64 MiB   (HBM)
        1 << 26,   //  64 MiB elems = 256 MiB   (HBM, large)
    };
    bool sizes_overridden = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_arg = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return std::string(argv[++i]);
        };
        if      (a == "--warmup")    warmup    = std::stoi(next_arg());
        else if (a == "--iters")     iters     = std::stoi(next_arg());
        else if (a == "--peak-gbps") peak_gbps = std::stod(next_arg());
        else if (a == "--size") {
            if (!sizes_overridden) { sizes.clear(); sizes_overridden = true; }
            sizes.push_back(std::stoi(next_arg()));
        } else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [--warmup N] [--iters N] [--peak-gbps F] [--size N]...\n", argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2;
        }
    }

    // Queue with profiling enabled.
    sycl::queue q{sycl::default_selector_v,
                  sycl::property_list{sycl::property::queue::enable_profiling{}}};
    auto dev = q.get_device();
    std::printf("Device              : %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Driver              : %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("Max work-group size : %u\n",
                (unsigned)dev.get_info<sycl::info::device::max_work_group_size>());
    std::printf("Warmup / iters      : %d / %d\n", warmup, iters);
    if (peak_gbps > 0) std::printf("Peak HBM (user)     : %.1f GB/s\n", peak_gbps);
    std::printf("\n");

    // Header
    if (peak_gbps > 0) {
        std::printf("%12s | %30s | %30s | %8s\n",
                    "N (elems)",
                    "naive  ms (min/med) | GB/s | %peak",
                    "vec    ms (min/med) | GB/s | %peak",
                    "speedup");
        std::printf("-------------+--------------------------------+--------------------------------+---------\n");
    } else {
        std::printf("%12s | %26s | %26s | %8s\n",
                    "N (elems)",
                    "naive  ms (min/med) | GB/s",
                    "vec    ms (min/med) | GB/s",
                    "speedup");
        std::printf("-------------+----------------------------+----------------------------+---------\n");
    }

    for (int N : sizes) {
        auto r = run_one_size(q, N, warmup, iters);
        double naive_gbps = gbps(N, r.naive.min_ms);
        double vec_gbps   = gbps(N, r.vec.min_ms);
        double speedup    = r.naive.min_ms / r.vec.min_ms;

        if (peak_gbps > 0) {
            std::printf("%12d | %7.3f/%7.3f | %6.1f | %5.1f%% | "
                        "%7.3f/%7.3f | %6.1f | %5.1f%% | %6.2fx\n",
                        N,
                        r.naive.min_ms, r.naive.med_ms, naive_gbps, 100.0 * naive_gbps / peak_gbps,
                        r.vec.min_ms,   r.vec.med_ms,   vec_gbps,   100.0 * vec_gbps   / peak_gbps,
                        speedup);
        } else {
            std::printf("%12d | %7.3f/%7.3f | %8.1f | %7.3f/%7.3f | %8.1f | %6.2fx\n",
                        N,
                        r.naive.min_ms, r.naive.med_ms, naive_gbps,
                        r.vec.min_ms,   r.vec.med_ms,   vec_gbps,
                        speedup);
        }
    }

    return 0;
}
