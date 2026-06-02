// silu_and_mul_bench.sycl.cpp
//
// Micro-benchmark for SiLU-and-Mul fusion kernel on Intel XPU
// Methodology follows xpu-perf framework (https://github.com/bytedance/xpu-perf):
//   - Allocate device buffers
//   - Warmup iterations (compile + cache warmup)
//   - N timed iterations using SYCL event profiling
//   - Report min/median/mean time and effective bandwidth
//
// Build:
//   icpx -fsycl -O3 -fsycl-targets=spir64 silu_and_mul_bench.sycl.cpp -o silu_and_mul_bench
//
// Run:
//   ./silu_and_mul_bench                          # default: num_tokens=8192, d=4096
//   ./silu_and_mul_bench --tokens 16384 --d 8192  # custom shape
//   ./silu_and_mul_bench --iters 100 --warmup 20  # custom iteration count

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi  = sycl::ext::oneapi::this_work_item;

// ---------------------------------------------------------------------------
// Kernel (free-function style)
// ---------------------------------------------------------------------------

inline float silu_op(float x) {
  return x / (1.0f + sycl::exp(-x));
}

SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void silu_and_mul_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    int d)
{
  auto item = syclwi::get_nd_item<1>();
  const int token_idx = item.get_group(0);
  const int tid = item.get_local_id(0);
  const int block_size = item.get_local_range(0);
  
  const float* gate = input + token_idx * 2 * d;
  const float* up   = input + token_idx * 2 * d + d;
  float* out_ptr    = out + token_idx * d;
  
  // Vectorized path
  constexpr int VEC_SIZE = 4;
  const int num_vec = d / VEC_SIZE;
  using vec_t = sycl::vec<float, VEC_SIZE>;
  
  for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
    int base = vec_idx * VEC_SIZE;
    vec_t gate_vec, up_vec, out_vec;
    
    gate_vec.load(0, sycl::multi_ptr<const float, 
                      sycl::access::address_space::global_space>(gate + base));
    up_vec.load(0, sycl::multi_ptr<const float, 
                    sycl::access::address_space::global_space>(up + base));
    
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      out_vec[i] = silu_op(gate_vec[i]) * up_vec[i];
    }
    
    out_vec.store(0, sycl::multi_ptr<float, 
                     sycl::access::address_space::global_space>(out_ptr + base));
  }
  
  // Scalar tail
  for (int i = num_vec * VEC_SIZE + tid; i < d; i += block_size) {
    out_ptr[i] = silu_op(gate[i]) * up[i];
  }
}

// ---------------------------------------------------------------------------
// Benchmark infrastructure
// ---------------------------------------------------------------------------

struct BenchResult {
  double min_ms;
  double median_ms;
  double mean_ms;
  double bandwidth_gbps;
  double peak_percent;
};

BenchResult benchmark_kernel(sycl::queue& q, int num_tokens, int d, 
                             int num_iters, int warmup_iters) {
  const int input_size = num_tokens * 2 * d;
  const int output_size = num_tokens * d;
  const size_t input_bytes = input_size * sizeof(float);
  const size_t output_bytes = output_size * sizeof(float);
  const size_t total_bytes = input_bytes + output_bytes; // read + write
  
  // Allocate device memory
  float* d_input = sycl::malloc_device<float>(input_size, q);
  float* d_output = sycl::malloc_device<float>(output_size, q);
  
  // Initialize with random data
  float* h_input = new float[input_size];
  for (int i = 0; i < input_size; ++i) {
    h_input[i] = (rand() % 1000) / 1000.0f - 0.5f;
  }
  q.memcpy(d_input, h_input, input_bytes).wait();
  delete[] h_input;
  
  constexpr int BLOCK_SIZE = 256;
  sycl::nd_range<1> ndr{num_tokens * BLOCK_SIZE, BLOCK_SIZE};
  
  // Warmup
  for (int i = 0; i < warmup_iters; ++i) {
    syclexp::nd_launch(q, ndr, syclexp::kernel_function<silu_and_mul_kernel>,
                       d_output, d_input, d);
  }
  q.wait();
  
  // Timed iterations with event profiling
  std::vector<double> times_ms;
  times_ms.reserve(num_iters);
  
  for (int i = 0; i < num_iters; ++i) {
    auto event = q.submit([&](sycl::handler& h) {
      h.parallel_for(ndr, [=](sycl::nd_item<1> item) {
        // Unfortunately, nd_launch doesn't return an event directly,
        // so we use a manual parallel_for here for profiling
        const int token_idx = item.get_group(0);
        const int tid = item.get_local_id(0);
        const int block_size = item.get_local_range(0);
        
        const float* gate = d_input + token_idx * 2 * d;
        const float* up   = d_input + token_idx * 2 * d + d;
        float* out_ptr    = d_output + token_idx * d;
        
        constexpr int VEC_SIZE = 4;
        const int num_vec = d / VEC_SIZE;
        using vec_t = sycl::vec<float, VEC_SIZE>;
        
        for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block_size) {
          int base = vec_idx * VEC_SIZE;
          vec_t gate_vec, up_vec, out_vec;
          
          gate_vec.load(0, sycl::multi_ptr<const float, 
                            sycl::access::address_space::global_space>(gate + base));
          up_vec.load(0, sycl::multi_ptr<const float, 
                          sycl::access::address_space::global_space>(up + base));
          
          #pragma unroll
          for (int j = 0; j < VEC_SIZE; ++j) {
            out_vec[j] = silu_op(gate_vec[j]) * up_vec[j];
          }
          
          out_vec.store(0, sycl::multi_ptr<float, 
                           sycl::access::address_space::global_space>(out_ptr + base));
        }
        
        for (int j = num_vec * VEC_SIZE + tid; j < d; j += block_size) {
          out_ptr[j] = silu_op(gate[j]) * up[j];
        }
      });
    });
    
    event.wait();
    
    // Get kernel execution time from event profiling
    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    double time_ns = static_cast<double>(end - start);
    times_ms.push_back(time_ns / 1e6); // convert to ms
  }
  
  // Compute statistics
  std::sort(times_ms.begin(), times_ms.end());
  double min_ms = times_ms.front();
  double median_ms = times_ms[num_iters / 2];
  double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / num_iters;
  
  // Bandwidth calculation (using minimum time for peak performance)
  double bandwidth_gbps = (total_bytes / 1e9) / (min_ms / 1000.0);
  
  // Cleanup
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  
  // Assume ~530 GB/s measured HBM bandwidth for BMG-G31
  // (user can override with --peak-gbps flag)
  double peak_gbps = 530.0;
  double peak_percent = (bandwidth_gbps / peak_gbps) * 100.0;
  
  return {min_ms, median_ms, mean_ms, bandwidth_gbps, peak_percent};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("Options:\n");
  printf("  --tokens N      Number of tokens (default: 8192)\n");
  printf("  --d N           Hidden dimension d (default: 4096)\n");
  printf("  --iters N       Number of timed iterations (default: 50)\n");
  printf("  --warmup N      Number of warmup iterations (default: 10)\n");
  printf("  --peak-gbps X   Peak HBM bandwidth in GB/s (default: 530.0)\n");
}

int main(int argc, char** argv) {
  int num_tokens = 8192;
  int d = 4096;
  int num_iters = 50;
  int warmup_iters = 10;
  double peak_gbps = 530.0;
  
  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
      num_tokens = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
      d = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      num_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--peak-gbps") == 0 && i + 1 < argc) {
      peak_gbps = atof(argv[++i]);
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }
  
  // Create queue with profiling enabled
  sycl::queue q{sycl::gpu_selector_v,
                {sycl::property::queue::in_order(),
                 sycl::property::queue::enable_profiling()}};
  
  printf("========================================\n");
  printf("SiLU-and-Mul Fusion Kernel Benchmark\n");
  printf("========================================\n");
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  printf("Shape: [num_tokens=%d, d=%d]\n", num_tokens, d);
  printf("Input:  %d tokens × 2 × %d = %.2f MB\n", 
         num_tokens, d, (num_tokens * 2 * d * sizeof(float)) / 1e6);
  printf("Output: %d tokens × %d = %.2f MB\n",
         num_tokens, d, (num_tokens * d * sizeof(float)) / 1e6);
  printf("Warmup: %d iterations\n", warmup_iters);
  printf("Timed:  %d iterations\n", num_iters);
  printf("----------------------------------------\n\n");
  
  auto result = benchmark_kernel(q, num_tokens, d, num_iters, warmup_iters);
  
  printf("Results:\n");
  printf("  Min time:       %.3f ms\n", result.min_ms);
  printf("  Median time:    %.3f ms\n", result.median_ms);
  printf("  Mean time:      %.3f ms\n", result.mean_ms);
  printf("  Bandwidth:      %.2f GB/s\n", result.bandwidth_gbps);
  printf("  Peak BW (assumed): %.2f GB/s\n", peak_gbps);
  printf("  %% of peak:      %.1f%%\n", (result.bandwidth_gbps / peak_gbps) * 100.0);
  printf("\n");
  
  return 0;
}
