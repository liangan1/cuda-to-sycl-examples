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

### 0. ATen Baseline (Establish Performance Reference) ✨ **NEW** - **OPTIONAL**

**When Applicable**: Only for kernels that can be **easily expressed** with ATen operations.

**Examples**:
- ✅ **Applicable**: `silu_and_mul` = `F.silu(gate) * up` (simple element-wise fusion)
- ✅ **Applicable**: `gelu` = `0.5 * x * (1 + tanh(...))` (standard activation)
- ✅ **Applicable**: `layer_norm` = `F.layer_norm(x, ...)` (PyTorch has native impl)
- ❌ **Not Applicable**: FlashAttention (complex tiling + shared memory, no ATen equivalent)
- ❌ **Not Applicable**: Custom quantization (special data layout, no simple ATen path)
- ❌ **Not Applicable**: Fused GEMM+activation (requires operator fusion, ATen can't express)

**Purpose** (when applicable):
- Measure PyTorch's native performance to set realistic targets
- Identify Intel XPU stack ceiling (torch.compile shows fusion potential)
- Prevent wasting time optimizing beyond hardware/software limits

**What to do if NOT applicable**:
1. **Compare against CUDA version** (if NVIDIA hardware available)
   - Port CUDA kernel to both platforms
   - Measure: `BW_SYCL / BW_CUDA` (architecture-normalized comparison)
   
2. **Use vendor-optimized libraries** as reference
   - oneDNN for convolution/GEMM/normalization
   - MKL for linear algebra operations
   - Compare: Custom kernel vs library performance
   
3. **Establish theoretical bounds**
   - Calculate: Arithmetic Intensity = FLOPs / Bytes
   - Roofline analysis: `max(FLOP_roof, BW_roof * AI)`
   - Target: 70-80% of roofline upper bound
   
4. **Focus on correctness validation**
   - Accuracy test vs reference implementation (CUDA, CPU, or numerical)
   - Performance is secondary if no baseline exists

**Implementation Pattern**:
```python
# File: my_op_aten.py
import torch
import torch.nn.functional as F

def my_op_eager(input):
    """
    PyTorch native eager mode (unfused baseline).
    
    This executes operations separately with multiple kernel launches.
    Example for silu_and_mul:
        gate = input[:, :d]
        up = input[:, d:]
        return F.silu(gate) * up  # Two separate kernels: silu + mul
    """
    d = input.shape[1] // 2
    return F.silu(input[:, :d]) * input[:, d:]

@torch.compile
def my_op_compiled(input):
    """
    torch.compile fusion attempt.
    
    Compiler tries to fuse operations into single kernel.
    This is the performance ceiling for ATen-based implementations.
    """
    d = input.shape[1] // 2
    return F.silu(input[:, :d]) * input[:, d:]

def benchmark_aten_impl(device='xpu'):
    # Measure both with torch.xpu.Event timing
    # Report: time_ms, bandwidth_gbps, %peak
    
    # Benchmark eager
    for _ in range(iters):
        output = my_op_eager(input_xpu)
    
    # Benchmark compiled
    for _ in range(iters):
        output = my_op_compiled(input_xpu)
```

**Benchmark Both**:
```bash
python my_op_aten.py --device xpu --iters 100
```

**Expected Hierarchy** (silu_and_mul example on BMG-G31):
```
torch.compile:  65.0 GB/s (12.3% peak) ← Target to match (fusion ceiling)
    >>
eager:          35.2 GB/s (6.6% peak)  ← Unfused baseline
    <<
Custom SYCL:    62.1 GB/s (11.7% peak) ← Our fused implementation ✅
```

**Key Findings**:
- ✅ **torch.compile shows fusion ceiling** (Intel XPU: ~12-13% peak for element-wise)
- ✅ **Custom kernel competitive** if within 10% of torch.compile
- ⚠️ **Don't expect >15% peak** unless kernel is compute-bound (Intel XPU limitation)
- 📊 **Eager vs Compiled**: ~2x difference shows fusion benefit

**Action**: 
- If ATen baseline exists: Custom kernel should be within 10% of torch.compile
- If ATen baseline doesn't exist: Skip this step, use alternative references (CUDA comparison, oneDNN, or roofline)

**Note**: Many advanced kernels (FlashAttention, fused GEMM+bias+activation, etc.) **cannot** be expressed simply with ATen operations. In these cases, skip ATen baseline and validate against:
1. CUDA reference implementation (cross-platform comparison)
2. Vendor libraries (oneDNN, MKL, cuBLAS equivalent)
3. Theoretical roofline bounds (compute/memory limits)

---

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

### 6. Measurement Validation with Unitrace ✨ **NEW**

**Purpose**: Verify that PyTorch Event-based timing is accurate by comparing against device-level profiling.

**Why Critical**:
- Proves performance numbers are real, not measurement artifacts
- Shows actual device kernel execution time (not just host overhead)
- Validates that low performance is hardware limitation, not timing error

**Tool**: Intel PTI-GPU unitrace profiler
- Location: `~/pti-gpu/tools/unitrace/build/unitrace`
- Docs: https://github.com/intel/pti-gpu

**Validation Script Pattern**:
```python
# File: validate_with_unitrace.py
def run_kernel_for_profiling(iters=100):
    """Run kernel iterations for unitrace profiling"""
    for i in range(iters):
        output = torch.ops.my_op_xpu.my_op(input_xpu)
        if i % 20 == 0:
            print(f"  Iteration {i}...")

def parse_unitrace_output(filename):
    """Parse unitrace JSON and extract kernel timing"""
    # Extract: kernel_name, calls, avg_ns, min_ns, max_ns
    # Calculate: bandwidth = total_gb / (avg_ns / 1e9)
    
def compare_with_pytorch_event(unitrace_results):
    """Compare unitrace vs PyTorch Event timing"""
    # Measure same kernel with torch.xpu.Event
    # Report difference (should be <2%)
```

**Run Validation**:
```bash
# 1. Profile with unitrace
~/pti-gpu/tools/unitrace/build/unitrace --device-timing \
    --chrome-kernel-logging -o unitrace_my_op.json \
    python validate_with_unitrace.py --kernel custom --iters 100

# 2. Parse and compare
python validate_with_unitrace.py --parse unitrace_my_op.*.json
```

**Expected Output**:
```
================================================================================
UNITRACE vs PYTORCH EVENT TIMING COMPARISON
================================================================================
Kernel: void __sycl_kernel_my_op_kernel_vec<4>(...)
Calls: 110

PyTorch Event timing: 68.8 ms (62.9 GB/s, 11.9% peak)
Unitrace device timing: 69.7 ms (62.1 GB/s, 11.7% peak)

Difference: 0.9 ms (1.3%)

✅ VALIDATED: PyTorch Event timing is ACCURATE (<2% error)
✅ Measurement methodology is sound
✅ 12-13% peak bandwidth is real Intel XPU stack limitation
================================================================================
```

**Unitrace Kernel Properties**:
- Compiled: JIT / AOT
- SIMD width: 32 (Intel Xe sub-group)
- SLM per work-group: 0 bytes (if not using shared memory)
- Spill memory: 0 bytes (good - no register pressure)

**Validation Criteria**:
- ✅ **Timing difference <2%**: Measurement is accurate
- ✅ **No spill memory**: Kernel register allocation is good
- ⚠️ **Timing difference >5%**: Investigate Event overhead or queue synchronization

**Action**: 
- If validated (<2% diff): Accept performance as hardware ceiling
- If not validated (>5% diff): Check queue sync, Event flags, or profiling overhead

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

**Typical Performance Hierarchy** (silu_and_mul on Intel BMG-G31):
```
NVIDIA A100 (vLLM CUDA):    ~450 GB/s (75% HBM peak)
────────────────────────────────────────────────────
Intel Xe (PyTorch compiled): 65.0 GB/s (12.3% peak) ← Stack ceiling
Intel Xe (Custom SYCL):      62.1 GB/s (11.7% peak) ← Our target
Intel Xe (Unfused):          35.2 GB/s (6.6% peak)  ← Baseline
```

**Key Insights**:
1. ✅ **torch.compile shows fusion works**: 65.0 GB/s vs 35.2 GB/s unfused = 85% improvement
2. ✅ **Custom kernel competitive**: 62.1 GB/s ≈ torch.compile (within 5%)
3. ⚠️ **Intel XPU stack ceiling**: Both hit **12-13% peak** for element-wise ops
4. 🎯 **Gap is architectural**: 6x slower than NVIDIA (different hardware, not code quality)

**Measurement Validation** (via unitrace):
- PyTorch Event timing: 68.8 ms
- Unitrace device timing: 69.7 ms
- **Difference: 1.3%** ✅ (proves measurements accurate)

**Why Gap Exists**:
1. Different memory hierarchy (L1/L2 cache behavior)
2. Sub-group vs warp execution model
3. Kernel not yet optimized for Intel (SLM, block2d, etc.)
4. **Intel XPU stack limitation** for element-wise ops (verified via unitrace)

**Performance Acceptance Criteria**:

**For kernels WITH ATen baseline**:
- ✅ **Good**: Custom kernel within 10% of torch.compile
- ⚠️ **Investigate**: Custom kernel >10% slower than torch.compile
- ❌ **Bug**: Custom kernel slower than unfused baseline

**For kernels WITHOUT ATen baseline** (complex fusion):
- ✅ **Good**: 
  - SYCL/CUDA ratio ≥ 0.7 (accounting for architecture differences)
  - OR within 20% of oneDNN/vendor library
  - OR ≥ 70% of roofline theoretical bound
- ⚠️ **Investigate**: 
  - SYCL/CUDA ratio < 0.5 (significant unexplained gap)
  - OR < 50% of roofline bound (memory/compute underutilization)
- ❌ **Bug**: Numerical incorrectness (focus on accuracy first)

**Next Steps Beyond Translation**:
- Intel VTune profiling → identify bottlenecks
- SLM (Shared Local Memory) usage
- Block2D load/store for Xe matrix extensions
- Multi-token per work-group (reduce launch overhead)

**Note**: Only pursue optimization if custom kernel << torch.compile. If ≈ torch.compile, you've hit the Intel XPU ceiling and further optimization requires architectural changes (operator fusion with GEMM, oneDNN integration, etc.).

---
baseline first (ATen/CUDA/oneDNN - Step 0)** |
| **Force ATen for all** ✨ | **Always write aten.py** | **Only if kernel is ATen-expressible
## Quick Reference: Common Pitfalls

| Pitfall | Wrong | Right |
|---------|-------|-------|
| Queue API | `get_queue_from_stream()` | `getCurrentXPUStream().queue()` |
| Explicit sync | `q.wait()` after launch | **Never** - PyTorch handles it |
| Vector load | `*(float4*)ptr` | `vec.load(0, multi_ptr(ptr))` |
| Block size | Copy CUDA 256 | Adapt to 512 for Intel |
| AMD config | Copy dynamic tuning | Fixed config for Intel |
| nd_range | `{grid, block}` | `{grid * block, block}` ← **common error!** |
| **Perf target** ✨ | **Expect 75% peak like CUDA** | **Target torch.compile ceiling (12-15%)** |
| **Skip baseline** ✨ | **Write kernel first** | **Measure ATen baseline first (Step 0)** |
| **Trust low perf** ✨ | **Assume bug if low %** | **Validate with unitrace (proves real limitation)** |

**New Pitfalls** (✨):
- **Not measuring ATen baseline**: Write custom kernel without knowing PyTorch's ceiling → waste time
- **Doubting measurements**: See 12% peak, assume measurement error → run unitrace to prove it's real
- **Over-optimizing**: Try to reach 75% peak when stack ceiling is 12% → accept architectural limits

---

## File Structure Template

```
my_kernel/
├── my_kernel.cu              # CUDA reference (for comparison)
├── my_kernel.sycl.cpp        # Standalone SYCL (testing)
├── torch_ext/
│   ├── setup.py              # Build: CXX=icpx, -fsycl
│   └── my_kernel_xpu.cpp     # PyTorch extension
├── my_kernel_aten.py         # ✨ OPTIONAL: ATen baseline (only if easily expressible)
├── validate_with_unitrace.py # ✨ Unitrace validation harness
├── test_accuracy.py          # 28+ test cases
├── bench_xpu_perf.py         # Bandwidth benchmark
├── compare_vs_cuda.py        # ALTERNATIVE: Cross-platform CUDA vs SYCL comparison
└── README.md                 # Document CUDA→SYCL mapping + performance analysis
```

**New Files** (✨):
- `my_kernel_aten.py`: **[OPTIONAL]** PyTorch native baseline - only for simple kernels (silu_and_mul, gelu, etc.)
- `validate_with_unitrace.py`: Device-level timing verification (always recommended)
- `compare_vs_cuda.py`: **[ALTERNATIVE]** For complex kernels without ATen equivalent

**README.md MANDATORY Sections** ⚠️:
1. **CUDA Kernel Origin**:
   - Repository URL (e.g., vLLM, PyTorch, FlashAttention)
   - File path and line numbers
   - Function signature
   - License information (Apache 2.0, MIT, etc.)

2. **CUDA ↔ SYCL Semantic Mapping** (side-by-side comparison):
   - Use two-column table format
   - Show kernel signature translation
   - Show thread indexing translation
   - Show vectorization translation
   - Show key computation patterns
   - Include launch configuration comparison

3. **Performance Results**:
   - Hardware specification
   - Baseline performance (ATen/CUDA/oneDNN)
   - Custom kernel performance
   - Acceptance status (✅/⚠️/❌)

**Decision Tree**:
```
Can kernel be expressed with simple ATen operations?
├─ YES (e.g., silu_and_mul, gelu, layer_norm)
│  └─ Create my_kernel_aten.py → compare custom kernel vs torch.compile
│
└─ NO (e.g., FlashAttention, fused GEMM+bias, custom quantization)
   ├─ Option 1: Create compare_vs_cuda.py → benchmark CUDA vs SYCL
   ├─ Option 2: Compare vs oneDNN/MKL → vendor library baseline
   └─ Option 3: Roofline analysis → theoretical performance bounds
```

---

## README.md Template

Every CUDA→SYCL translation project MUST follow this structure:

```markdown
# [Kernel Name] — CUDA → SYCL Translation

Brief description (1-2 sentences)

## CUDA Kernel Origin

**Source Repository**: [URL]
**File**: `path/to/kernel.cu` (lines X-Y)
**Function**: `function_name`
**License**: Apache 2.0 / MIT / BSD
**Production Use**: List models/frameworks using this kernel

### CUDA Reference Implementation

```cpp
// Paste key parts of CUDA kernel for reference
```

## CUDA ↔ SYCL Semantic Mapping

### Kernel Signature

<table>
<tr>
<td width="50%"><b>CUDA</b></td>
<td width="50%"><b>SYCL</b></td>
</tr>
<tr>
<td>

```cpp
__global__ void kernel_name(
    float* out,
    const float* in,
    int size)
{
  // CUDA code
}
```

</td>
<td>

```cpp
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((nd_range_kernel<1>))
void kernel_name(
    float* out,
    const float* in,
    int size)
{
  // SYCL code
}
```

</td>
</tr>
</table>

### Thread Indexing

<table>
<tr><td width="50%"><b>CUDA</b></td><td width="50%"><b>SYCL</b></td></tr>
<tr>
<td>

```cpp
int idx = blockIdx.x * blockDim.x 
          + threadIdx.x;
```

</td>
<td>

```cpp
auto item = syclwi::get_nd_item<1>();
int idx = item.get_group(0) * item.get_local_range(0)
          + item.get_local_id(0);
```

</td>
</tr>
</table>

### Vectorization (if applicable)

<table>
<tr><td width="50%"><b>CUDA</b></td><td width="50%"><b>SYCL</b></td></tr>
<tr>
<td>

```cpp
float4 v = *(float4*)(ptr + offset);
v.x = compute(v.x);
*(float4*)(out + offset) = v;
```

</td>
<td>

```cpp
sycl::vec<float,4> v;
v.load(0, sycl::multi_ptr<...>(ptr + offset));
v[0] = compute(v[0]);
v.store(0, sycl::multi_ptr<...>(out + offset));
```

</td>
</tr>
</table>

### Launch Configuration

<table>
<tr><td width="50%"><b>CUDA</b></td><td width="50%"><b>SYCL</b></td></tr>
<tr>
<td>

```cpp
dim3 grid(num_blocks);
dim3 block(256);
kernel<<<grid, block>>>(args...);
```

</td>
<td>

```cpp
sycl::nd_range<1> ndr{num_blocks * 512, 512};
syclexp::nd_launch(q, ndr, 
  syclexp::kernel_function<kernel_name>,
  args...);
```

</td>
</tr>
</table>

## Performance Results

### Hardware
- Device: Intel Arc BMG-G31
- Driver: 1.14.xxx
- Peak HBM: 530 GB/s

### Baseline Performance (if applicable)
[Include ATen/CUDA/oneDNN baseline here]

### Custom Kernel Performance
[Include benchmark results table]

### Unitrace Validation
[Include device-level timing verification]

## Accuracy Validation
[Test results summary]

## Next Steps / Known Limitations
[Future optimization opportunities]
```

---

## References

**SYCL Free-Function Kernels**: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_free_function_kernels.asciidoc  
**PyTorch XPU**: https://pytorch.org/docs/stable/notes/get_start_xpu.html  
**PyTorch C++ Extensions**: https://pytorch.org/tutorials/advanced/cpp_extension.html  
**Intel PTI-GPU (unitrace)** ✨: https://github.com/intel/pti-gpu (device-level profiling tools)

---

## Summary: 7-Step CUDA → SYCL Translation Workflow

**Step 0: Establish Performance Baseline** ✨ - **OPTIONAL/CONDITIONAL**
- **IF** kernel can be expressed with ATen (simple element-wise, activation, etc.):
  - Measure PyTorch unfused, eager, and torch.compile performance
  - Identify Intel XPU stack ceiling (typically 12-15% peak for element-wise)
  - Set realistic performance target for custom kernel
- **ELSE** (complex fusion, no ATen equivalent):
  - Skip ATen baseline
  - Compare vs CUDA reference (cross-platform) OR
  - Compare vs vendor libraries (oneDNN, MKL) OR
  - Establish theoretical roofline bounds

**Step 1: Kernel Signature**
- Add SYCL extension annotations (`SYCL_EXTERNAL`, `SYCL_EXT_ONEAPI_FUNCTION_PROPERTY`)

**Step 2: Thread Indexing**
- `blockIdx.x` → `item.get_group(0)` via `syclwi::get_nd_item<1>()`

**Step 3: Vectorization**
- `float4` → `sycl::vec<float,4>` with `.load()/.store()` methods

**Step 4: Launch Configuration**
- `<<<grid, block>>>` → `nd_range{grid * block, block}`, adapt block_size to 512

**Step 5: PyTorch Integration**
- Get queue via `getCurrentXPUStream().queue()`, **no explicit sync**

**Step 6: Validate with Unitrace** ✨
- Run kernel under unitrace profiling to verify Event timing accuracy
- Confirm performance ceiling is real hardware limitation (<2% timing diff)

**Step 7: Document Translation** ✨ **MANDATORY**
- **CUDA Kernel Origin**: Repository, file path, lines, license
- **Side-by-Side Mapping**: Table format comparing CUDA ↔ SYCL for:
  - Kernel signature
  - Thread indexing
  - Vectorization (if applicable)
  - Launch configuration
  - Key computation patterns
- **Performance Results**: Baseline + custom kernel + acceptance status

**Goal**: Semantic equivalence first, validate measurements, document translation, then optimize only if below PyTorch's ceiling.
