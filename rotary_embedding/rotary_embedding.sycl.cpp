// rotary_embedding.sycl.cpp
// Step 1: Standalone SYCL Implementation of Rotary Position Embedding
// Following SKILLS.md semantic mapping from vLLM CUDA kernel

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi = sycl::ext::oneapi::this_work_item;

// Rotary Position Embedding kernel
// CUDA Reference: vLLM csrc/pos_encoding_kernels.cu
// Each work-group handles one token
// Threads parallelize over embed_dim (head_size / 2)
// Each thread processes all heads for its assigned dimension

template<typename T>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,    // [num_tokens]
    T* __restrict__ query,                    // [num_tokens, num_heads * head_size]
    T* __restrict__ key,                      // [num_tokens, num_kv_heads * head_size]
    const T* __restrict__ cos_sin_cache,      // [max_position, head_size]
    int head_size,
    int num_heads,
    int num_kv_heads,
    int query_stride,
    int key_stride)
{
    // Step 2: Thread Indexing (SKILLS.md pattern)
    auto item = syclwi::get_nd_item<1>();
    
    // Each work-group handles one token (CUDA: blockIdx.x)
    const int token_idx = item.get_group(0);
    
    // Get position for this token
    int64_t pos = positions[token_idx];
    const T* cache_ptr = cos_sin_cache + pos * head_size;
    
    // Split cos/sin cache
    const int embed_dim = head_size / 2;
    const T* cos_ptr = cache_ptr;
    const T* sin_ptr = cache_ptr + embed_dim;
    
    // Step 2: Thread loop over embed dimension (CUDA: threadIdx.x)
    for (int i = item.get_local_id(0); i < embed_dim; i += item.get_local_range(0)) {
        // Load cos/sin for this dimension
        const T cos_val = cos_ptr[i];
        const T sin_val = sin_ptr[i];
        
        // Rotate all query heads for this dimension
        for (int head = 0; head < num_heads; ++head) {
            const int offset = token_idx * query_stride + head * head_size;
            const T q0 = query[offset + i];
            const T q1 = query[offset + i + embed_dim];
            
            // Apply rotation: [q0, q1] → [q0*cos - q1*sin, q0*sin + q1*cos]
            query[offset + i] = q0 * cos_val - q1 * sin_val;
            query[offset + i + embed_dim] = q0 * sin_val + q1 * cos_val;
        }
        
        // Rotate all key heads for this dimension
        for (int head = 0; head < num_kv_heads; ++head) {
            const int offset = token_idx * key_stride + head * head_size;
            const T k0 = key[offset + i];
            const T k1 = key[offset + i + embed_dim];
            
            // Apply rotation
            key[offset + i] = k0 * cos_val - k1 * sin_val;
            key[offset + i + embed_dim] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// Helper: Generate cos/sin cache for rotary embeddings
void generate_cos_sin_cache(float* cache, int max_position, int head_size, float base = 10000.0f) {
    const int embed_dim = head_size / 2;
    
    for (int pos = 0; pos < max_position; ++pos) {
        float* cos_ptr = cache + pos * head_size;
        float* sin_ptr = cos_ptr + embed_dim;
        
        for (int i = 0; i < embed_dim; ++i) {
            float freq = 1.0f / std::pow(base, 2.0f * i / static_cast<float>(head_size));
            float angle = pos * freq;
            cos_ptr[i] = std::cos(angle);
            sin_ptr[i] = std::sin(angle);
        }
    }
}

// Standalone test
int main() {
    sycl::queue q{sycl::default_selector_v};
    
    std::cout << "=================================================================\n";
    std::cout << "SKILLS.md Step 1: Rotary Embedding SYCL Kernel (Standalone Test)\n";
    std::cout << "=================================================================\n";
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";
    
    // Test configuration
    const int num_tokens = 4;
    const int num_heads = 8;
    const int num_kv_heads = 8;  // Same as num_heads for this test
    const int head_size = 64;
    const int max_position = 100;
    const float base = 10000.0f;
    
    const int query_size = num_tokens * num_heads * head_size;
    const int key_size = num_tokens * num_kv_heads * head_size;
    const int cache_size = max_position * head_size;
    const int query_stride = num_heads * head_size;
    const int key_stride = num_kv_heads * head_size;
    
    std::cout << "Configuration:\n";
    std::cout << "  num_tokens: " << num_tokens << "\n";
    std::cout << "  num_heads: " << num_heads << "\n";
    std::cout << "  head_size: " << head_size << "\n";
    std::cout << "  max_position: " << max_position << "\n\n";
    
    // Allocate device memory
    int64_t* d_positions = sycl::malloc_device<int64_t>(num_tokens, q);
    float* d_query = sycl::malloc_device<float>(query_size, q);
    float* d_key = sycl::malloc_device<float>(key_size, q);
    float* d_cos_sin_cache = sycl::malloc_device<float>(cache_size, q);
    
    // Allocate host memory
    int64_t* h_positions = new int64_t[num_tokens];
    float* h_query_in = new float[query_size];
    float* h_query_out = new float[query_size];
    float* h_key_in = new float[key_size];
    float* h_key_out = new float[key_size];
    float* h_cos_sin_cache = new float[cache_size];
    
    // Initialize positions (0, 1, 2, 3)
    for (int i = 0; i < num_tokens; ++i) {
        h_positions[i] = i;
    }
    
    // Initialize query/key with simple pattern
    for (int i = 0; i < query_size; ++i) {
        h_query_in[i] = static_cast<float>(i % 100) * 0.01f;
    }
    for (int i = 0; i < key_size; ++i) {
        h_key_in[i] = static_cast<float>(i % 100) * 0.01f + 0.5f;
    }
    
    // Generate cos/sin cache
    generate_cos_sin_cache(h_cos_sin_cache, max_position, head_size, base);
    
    // Copy to device
    q.memcpy(d_positions, h_positions, num_tokens * sizeof(int64_t)).wait();
    q.memcpy(d_query, h_query_in, query_size * sizeof(float)).wait();
    q.memcpy(d_key, h_key_in, key_size * sizeof(float)).wait();
    q.memcpy(d_cos_sin_cache, h_cos_sin_cache, cache_size * sizeof(float)).wait();
    
    // Step 3: Launch Configuration (SKILLS.md pattern)
    // CUDA: <<<num_tokens, min(head_size / 2, 512)>>>
    const int work_groups = num_tokens;  // 1 work-group per token
    const int local_size = std::min(head_size / 2, 512);
    
    sycl::nd_range<1> ndr{
        sycl::range<1>(size_t(work_groups) * local_size),  // global_size = grid × block
        sycl::range<1>(local_size)                          // local_size = block
    };
    
    std::cout << "Launch config:\n";
    std::cout << "  work_groups: " << work_groups << "\n";
    std::cout << "  local_size: " << local_size << "\n";
    std::cout << "  global_size: " << work_groups * local_size << "\n\n";
    
    // Launch kernel
    std::cout << "Launching kernel...\n";
    syclexp::nd_launch(q, ndr,
                       syclexp::kernel_function<rotary_embedding_kernel<float>>,
                       d_positions, d_query, d_key, d_cos_sin_cache,
                       head_size, num_heads, num_kv_heads, 
                       query_stride, key_stride);
    q.wait();
    std::cout << "Kernel completed\n\n";
    
    // Copy results back
    q.memcpy(h_query_out, d_query, query_size * sizeof(float)).wait();
    q.memcpy(h_key_out, d_key, key_size * sizeof(float)).wait();
    
    // Verification: manually compute expected result for first token, first head, first pair
    std::cout << "Verification (token=0, head=0, first pair):\n";
    int token = 0;
    int head = 0;
    int dim = 0;
    int64_t pos = h_positions[token];
    
    // Get cos/sin from cache
    const int embed_dim = head_size / 2;
    float cos_val = h_cos_sin_cache[pos * head_size + dim];
    float sin_val = h_cos_sin_cache[pos * head_size + embed_dim + dim];
    
    // Get input query pair
    int offset = token * query_stride + head * head_size;
    float q0_in = h_query_in[offset + dim];
    float q1_in = h_query_in[offset + dim + embed_dim];
    
    // Compute expected rotation
    float q0_expected = q0_in * cos_val - q1_in * sin_val;
    float q1_expected = q0_in * sin_val + q1_in * cos_val;
    
    // Get actual output
    float q0_actual = h_query_out[offset + dim];
    float q1_actual = h_query_out[offset + dim + embed_dim];
    
    std::cout << "  Input: [" << q0_in << ", " << q1_in << "]\n";
    std::cout << "  cos/sin: [" << cos_val << ", " << sin_val << "]\n";
    std::cout << "  Expected: [" << q0_expected << ", " << q1_expected << "]\n";
    std::cout << "  Actual: [" << q0_actual << ", " << q1_actual << "]\n";
    std::cout << "  Diff: [" << std::abs(q0_actual - q0_expected) 
              << ", " << std::abs(q1_actual - q1_expected) << "]\n";
    
    float max_diff = std::max(std::abs(q0_actual - q0_expected),
                              std::abs(q1_actual - q1_expected));
    
    std::cout << "\n";
    if (max_diff < 1e-5) {
        std::cout << "✅ PASS: Standalone test successful (max_diff = " << max_diff << ")\n";
    } else {
        std::cout << "❌ FAIL: max_diff = " << max_diff << " (threshold: 1e-5)\n";
    }
    
    // Cleanup
    sycl::free(d_positions, q);
    sycl::free(d_query, q);
    sycl::free(d_key, q);
    sycl::free(d_cos_sin_cache, q);
    delete[] h_positions;
    delete[] h_query_in;
    delete[] h_query_out;
    delete[] h_key_in;
    delete[] h_key_out;
    delete[] h_cos_sin_cache;
    
    std::cout << "=================================================================\n";
    
    return (max_diff < 1e-5) ? 0 : 1;
}
