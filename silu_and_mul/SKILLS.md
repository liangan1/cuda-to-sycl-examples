# CUDA → SYCL Translation Skills (Element-wise Kernels)

**Task**: Translate NVIDIA CUDA element-wise kernels to Intel SYCL with 1:1 semantic mapping

**Scope**: Standalone C++ + PyTorch integration for memory-bound kernels (activations, normalization, etc.)

---

## Critical Mistakes Made (Lessons Learned)

### 1. ❌ PyTorch Queue API Error
**Wrong**: `c10::xpu::get_queue_from_stream()` (doesn't exist)  
**Right**: `c10::xpu::getCurrentXPUStream(device_idx).queue()`

### 2. ❌ Unnecessary q.wait()
**Wrong**: Adding explicit `q.wait()` after kernel launch  
**Right**: PyTorch manages stream synchronization automatically - **never add explicit waits**

### 3. ❌ Copying AMD Configuration
**Wrong**: Dynamic vec_size/block_size tuning from AMD (wavefront=64)  
**Right**: Fixed config adapted for Intel Xe (sub-group=16)  
**Reason**: AMD wavefront ≠ Intel sub-group, wrong assumptions = 40% perf loss

### 4. ❌ Assuming Warp Size = 32
**Wrong**: Using CUDA warp-aligned block_size (256 = 8 warps)  
**Right**: Adapt to Intel sub-group size (512 = 32 sub-groups, optimal for Xe)

### 5. ❌ Missing Free-Function Kernel Extension
**Wrong**: Using lambda kernels without extension check  
**Right**: Verify `SYCL_EXT_ONEAPI_FREE_FUNCTION_PROPERTY` is available in oneAPI 2025.0+

---

## 1:1 CUDA → SYCL Semantic Mapping

### Step 1: Kernel Signature Translation

| CUDA | SYCL Free-Function Kernel |
|------|---------------------------|
| `__global__ void kernel(...)` | `SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>)) void kernel(...)` |
| Arguments directly accessible | Same - arguments directly accessible |

**Key**: SYCL free-function kernel syntax requires extension annotation, but logic is identical.

### Step 2: Thread Indexing

| CUDA | SYCL | Get It Via |
|------|------|------------|
| `blockIdx.x` | `item.get_group(0)` | `auto item = syclwi::get_nd_item<1>()` |
| `threadIdx.x` | `item.get_local_id(0)` | Same item object |
| `blockDim.x` | `item.get_local_range(0)` | Same item object |
| `gridDim.x` | `item.get_group_range(0)` | Same item object |

**Pattern**:
```cpp
// CUDA                          // SYCL
int tid = blockIdx.x * 256       auto item = syclwi::get_nd_item<1>();
          + threadIdx.x;         int tid = item.get_group(0) * 512 
                                          + item.get_local_id(0);
```

### Step 3: Vectorization (Critical for Bandwidth)

| CUDA | SYCL | Access Pattern |
|------|------|----------------|
| `float4 v = *(float4*)ptr` | `sycl::vec<float, 4> v; v.load(0, multi_ptr(ptr))` | 128-bit aligned load |
| `v.x, v.y, v.z, v.w` | `v[0], v[1], v[2], v[3]` | Element access |
| `*(float4*)ptr = v` | `v.store(0, multi_ptr(ptr))` | 128-bit aligned store |

**Critical**: SYCL requires explicit `.load()/.store()` - don't use reinterpret_cast!

### Step 4: Math Functions

| CUDA | SYCL |
|------|------|
| `expf(x)` | `sycl::exp(x)` |
| `__expf(x)` (fast) | `sycl::native::exp(x)` |
| `1.0f / (1.0f + expf(-x))` | Same - no change |

**Note**: SYCL has `sycl::native::*` for fast math, but default is fine for correctness.

### Step 5: Launch Configuration

| CUDA | SYCL | Semantic |
|------|------|----------|
| `<<<grid, block>>>` | `nd_range{grid * block, block}` | Global size = grid × block |
| `grid = num_tokens` | Same | 1 work-group per token |
| `block = 256` | Adapt to `512` | Intel sub-group size 16 vs NVIDIA warp 32 |

**Translation**:
```cpp
// CUDA: <<<num_tokens, 256>>>
sycl::nd_range<1> ndr{num_tokens * 512, 512};
syclexp::nd_launch(q, ndr, syclexp::kernel_function<kernel_name<VEC_SIZE>>, args...);
```

---

## PyTorch Integration Workflow

### 1. Standalone SYCL Kernel (Verify Correctness)
- Create `kernel.sycl.cpp` with free-function kernel
- Test with standalone main() using raw SYCL buffers
- Verify: `icpx -fsycl -O3 kernel.sycl.cpp && ./a.out`

### 2. PyTorch C++ Extension (Production)
**File**: `torch_ext/kernel_xpu.cpp`

**Required Headers**:
```cpp
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
```

**Get PyTorch Queue** (Critical!):
```cpp
c10::xpu::XPUStream xpu_stream = c10::xpu::getCurrentXPUStream(input.device().index());
sycl::queue& q = xpu_stream.queue();
// Never add q.wait() - PyTorch manages sync!
```

**TORCH_LIBRARY Registration**:
```cpp
TORCH_LIBRARY(my_op_xpu, m) {
  m.def("my_op(Tensor input) -> Tensor", &my_op_xpu);
}
```

### 3. Build Script (setup.py)
**Key**: Set `CXX=icpx` and add `-fsycl` flags:
```python
os.environ['CXX'] = 'icpx'
ext_modules=[
    CppExtension(name='my_op_xpu', sources=['my_op_xpu.cpp'],
                 extra_compile_args={'cxx': ['-fsycl', '-O3', '-fsycl-targets=spir64']},
                 extra_link_args=['-fsycl'])
]
```

### 4. Accuracy Test
**Pattern**:
```python
def test(batch, dim):
    input_xpu = torch.randn(batch, dim, device='xpu')
    output_xpu = torch.ops.my_op_xpu.my_op(input_xpu)
    output_cpu = reference_pytorch_op(input_xpu.cpu())
    assert torch.allclose(output_xpu.cpu(), output_cpu, rtol=1e-5, atol=1e-6)
```

**Coverage**: Test 14+ batch sizes × 2+ dims = 28 cases minimum

### 5. Performance Benchmark (xpu-perf format)
**Measure**: Bandwidth (GB/s) and % of HBM peak  
**Shape Coverage**: Follow upstream (e.g., bytedance/xpu-perf for reference)  
**Metric**: `Bandwidth = (Input_MB + Output_MB) / Time_ms × 1000`

---

## Configuration Adaptation Checklist

| Parameter | CUDA Default | Intel Xe Adaptation | Reason |
|-----------|--------------|---------------------|--------|
| **block_size** | 256 (8 warps) | 512 (32 sub-groups) | Sub-group size 16 vs warp 32 |
| **vec_size** | 4 (float4) | 4 (vec<float,4>) | Same - 128-bit optimal |
| **Grid strategy** | 1 block/token | Same | Element-wise = no change needed |
| **Dynamic config** | ❌ Don't copy AMD | Use fixed Intel-optimized | Wrong assumptions = perf loss |

**Rule**: Start with CUDA's grid/block semantics, only adapt block_size for sub-group alignment.

---

## Performance Expectations

**CUDA → SYCL Semantic Translation**: ✅ Achieves correctness, not performance parity  

**Typical Gap**:
- NVIDIA A100 (vLLM optimized): 75% HBM peak
- Intel Xe (direct translation): 10-15% HBM peak
- **Gap is architectural, not translation error**

**Why Gap Exists**:
1. Different memory hierarchy (L1/L2 cache behavior)
2. Sub-group vs warp execution model
3. Kernel not yet optimized for Intel (SLM, block2d, etc.)

**Next Steps Beyond Translation**:
- Intel VTune profiling → identify bottlenecks
- SLM (Shared Local Memory) usage
- Block2D load/store for Xe matrix extensions
- Multi-token per work-group (reduce launch overhead)

---

## Quick Reference: Common Pitfalls

| Pitfall | Wrong | Right |
|---------|-------|-------|
| Queue API | `get_queue_from_stream()` | `getCurrentXPUStream().queue()` |
| Explicit sync | `q.wait()` after launch | **Never** - PyTorch handles it |
| Vector load | `*(float4*)ptr` | `vec.load(0, multi_ptr(ptr))` |
| Block size | Copy CUDA 256 | Adapt to 512 for Intel |
| AMD config | Copy dynamic tuning | Fixed config for Intel |
| nd_range | `{grid, block}` | `{grid * block, block}` ← **common error!** |

---

## File Structure Template

```
my_kernel/
├── my_kernel.cu              # CUDA reference (for comparison)
├── my_kernel.sycl.cpp        # Standalone SYCL (testing)
├── torch_ext/
│   ├── setup.py              # Build: CXX=icpx, -fsycl
│   └── my_kernel_xpu.cpp     # PyTorch extension
├── test_accuracy.py          # 28+ test cases
├── bench_xpu_perf.py         # Bandwidth benchmark
└── README.md                 # Document CUDA→SYCL mapping
```

---

## References

**SYCL Free-Function Kernels**: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_free_function_kernels.asciidoc  
**PyTorch XPU**: https://pytorch.org/docs/stable/notes/get_start_xpu.html  
**PyTorch C++ Extensions**: https://pytorch.org/tutorials/advanced/cpp_extension.html

---

## Summary: 5-Step CUDA → SYCL Translation

1. **Kernel signature**: Add SYCL extension annotations
2. **Thread indexing**: `blockIdx.x` → `item.get_group(0)`, get item via `syclwi::get_nd_item<1>()`
3. **Vectorization**: `float4` → `sycl::vec<float,4>` with `.load()/.store()`
4. **Launch config**: `<<<grid, block>>>` → `nd_range{grid * block, block}`, adapt block_size
5. **PyTorch queue**: `getCurrentXPUStream().queue()`, **no explicit sync**

**Goal**: Semantic equivalence first, performance optimization second (requires Intel-specific tuning beyond translation).
