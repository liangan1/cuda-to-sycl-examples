// rotary_embedding_xpu.cpp
// Step 2: PyTorch C++ Extension for Rotary Position Embedding
// Following SKILLS.md PyTorch integration pattern

#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <algorithm>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclwi = sycl::ext::oneapi::this_work_item;

// SYCL kernel definition (inline for PyTorch extension)
template<typename T>
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,
    T* __restrict__ query,
    T* __restrict__ key,
    const T* __restrict__ cos_sin_cache,
    int head_size,
    int num_heads,
    int num_kv_heads,
    int query_stride,
    int key_stride)
{
    // Get nd_item
    auto item = syclwi::get_nd_item<1>();
    
    // Each work-group handles one token
    const int token_idx = item.get_group(0);
    
    // Get position for this token
    int64_t pos = positions[token_idx];
    const T* cache_ptr = cos_sin_cache + pos * head_size;
    
    // Split cos/sin cache
    const int embed_dim = head_size / 2;
    const T* cos_ptr = cache_ptr;
    const T* sin_ptr = cache_ptr + embed_dim;
    
    // Thread loop over embed dimension
    for (int i = item.get_local_id(0); i < embed_dim; i += item.get_local_range(0)) {
        // Load cos/sin for this dimension
        const T cos_val = cos_ptr[i];
        const T sin_val = sin_ptr[i];
        
        // Rotate all query heads for this dimension
        for (int head = 0; head < num_heads; ++head) {
            const int offset = token_idx * query_stride + head * head_size;
            const T q0 = query[offset + i];
            const T q1 = query[offset + i + embed_dim];
            
            // Apply rotation
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

// PyTorch wrapper function
void rotary_embedding_xpu(
    torch::Tensor& query,              // [num_tokens, num_heads * head_size]
    torch::Tensor& key,                // [num_tokens, num_kv_heads * head_size]
    torch::Tensor const& positions,    // [num_tokens]
    torch::Tensor const& cos_sin_cache // [max_position, head_size]
) {
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(cos_sin_cache.is_contiguous(), "cos_sin_cache must be contiguous");
    
    TORCH_CHECK(query.device().is_xpu(), "query must be on XPU");
    TORCH_CHECK(key.device().is_xpu(), "key must be on XPU");
    TORCH_CHECK(positions.device().is_xpu(), "positions must be on XPU");
    TORCH_CHECK(cos_sin_cache.device().is_xpu(), "cos_sin_cache must be on XPU");
    
    // Get dimensions
    const int num_tokens = query.size(0);
    const int num_heads = query.size(1);
    const int head_size = query.size(2);
    const int num_kv_heads = key.size(1);
    const int query_stride = query.stride(0);
    const int key_stride = key.stride(0);
    
    TORCH_CHECK(head_size % 2 == 0, "head_size must be even");
    
    // Step 2: Get PyTorch queue (SKILLS.md pattern - CRITICAL!)
    c10::xpu::XPUStream xpu_stream = c10::xpu::getCurrentXPUStream(query.device().index());
    sycl::queue& q = xpu_stream.queue();
    
    // Get raw pointers
    int64_t* positions_ptr = positions.data_ptr<int64_t>();
    float* cos_sin_cache_ptr = cos_sin_cache.data_ptr<float>();
    
    // Step 3: Launch Configuration (SKILLS.md pattern)
    // CUDA: <<<num_tokens, min(head_size / 2, 512)>>>
    const int work_groups = num_tokens;  // 1 work-group per token
    const int local_size = std::min(head_size / 2, 512);
    
    sycl::nd_range<1> ndr{
        sycl::range<1>(size_t(work_groups) * local_size),
        sycl::range<1>(local_size)
    };
    
    // Dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "rotary_embedding_xpu", [&] {
        scalar_t* query_ptr = query.data_ptr<scalar_t>();
        scalar_t* key_ptr = key.data_ptr<scalar_t>();
        
        syclexp::nd_launch(q, ndr,
                           syclexp::kernel_function<rotary_embedding_kernel<scalar_t>>,
                           positions_ptr, query_ptr, key_ptr, 
                           (scalar_t*)cos_sin_cache_ptr,
                           head_size, num_heads, num_kv_heads, 
                           query_stride, key_stride);
    });
    
    // Step 2: NO explicit q.wait() - PyTorch manages synchronization!
}

// TORCH_LIBRARY registration
TORCH_LIBRARY(rotary_embedding_xpu, m) {
    m.def("rotary_embedding(Tensor(a!) query, Tensor(b!) key, Tensor positions, Tensor cos_sin_cache) -> ()");
}

TORCH_LIBRARY_IMPL(rotary_embedding_xpu, XPU, m) {
    m.impl("rotary_embedding", &rotary_embedding_xpu);
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_embedding", &rotary_embedding_xpu, "Rotary Position Embedding (XPU)");
}
