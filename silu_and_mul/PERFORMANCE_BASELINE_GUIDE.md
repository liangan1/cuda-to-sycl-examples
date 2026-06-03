# Performance Baseline Selection Guide

**Question**: How to establish performance reference for CUDA → SYCL kernel translation?

**Answer**: Depends on kernel complexity and ATen expressibility.

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────┐
│ Can kernel be expressed with simple ATen/PyTorch operations?   │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
              YES                          NO
                │                           │
                v                           v
    ┌────────────────────────┐   ┌────────────────────────┐
    │   ATen Baseline        │   │  Alternative Baselines │
    │   (Step 0 - OPTIONAL)  │   │  (Choose one below)    │
    └────────────────────────┘   └────────────────────────┘
                │                           │
                v                           v
    ┌────────────────────────┐   ┌────────────────────────┐
    │ Compare custom kernel  │   │ 1. CUDA Reference      │
    │ vs torch.compile       │   │ 2. Vendor Libraries    │
    │                        │   │ 3. Roofline Theory     │
    │ Target: within 10%     │   │ 4. Accuracy Only       │
    └────────────────────────┘   └────────────────────────┘
```

---

## Category 1: ATen-Expressible Kernels ✅

**Characteristic**: Can be written as composition of standard PyTorch operations.

### Examples

| Kernel | ATen Expression | Baseline Script |
|--------|----------------|-----------------|
| `silu_and_mul` | `F.silu(gate) * up` | `silu_and_mul_aten.py` |
| `gelu` | `0.5 * x * (1 + tanh(...))` | `gelu_aten.py` |
| `swiglu` | `F.silu(gate) * up` | Same as silu_and_mul |
| `rms_norm` | `x * rsqrt(mean(x^2) + eps)` | `rms_norm_aten.py` |
| `softmax` | `F.softmax(x, dim=-1)` | Direct PyTorch API |
| `layer_norm` | `F.layer_norm(x, ...)` | Direct PyTorch API |

### Workflow

1. **Create `<kernel>_aten.py`**:
   ```python
   def kernel_eager(input):
       """PyTorch native eager mode (unfused baseline)"""
       # Separate operations, multiple kernel launches
       # e.g., F.silu(gate) * up
       
   @torch.compile
   def kernel_compiled(input):
       """torch.compile fusion attempt (performance ceiling)"""
       # Compiler tries to fuse operations
   ```

2. **Benchmark both**:
   ```bash
   python kernel_aten.py --device xpu --iters 100
   ```

3. **Set target**:
   - Custom kernel should match `torch.compile` (within 10%)
   - If torch.compile achieves 12% peak → target 11-13% peak

4. **Example output**:
   ```
   eager:       35.2 GB/s (6.6% peak)   ← Unfused baseline (multiple kernels)
   compiled:    65.0 GB/s (12.3% peak)  ← Target ceiling (fusion achieved)
   Custom SYCL: 62.1 GB/s (11.7% peak)  ← ✅ Within 5% of compiled
   ```

---

## Category 2: Complex Fusion Kernels ❌

**Characteristic**: Cannot be simply expressed with ATen (shared memory, tiling, special layouts).

### Examples

| Kernel | Why NOT ATen-expressible | Alternative Baseline |
|--------|--------------------------|----------------------|
| **FlashAttention** | Tiling + shared memory + online softmax | CUDA reference + roofline |
| **Fused GEMM+Bias+Activation** | Requires operator fusion | oneDNN `matmul_post_ops` |
| **PagedAttention (vLLM)** | Custom KV-cache layout | CUDA reference |
| **Grouped Query Attention** | Complex multi-head layout | CUDA reference |
| **Custom Quantization** | Non-standard data types | CUDA reference |
| **Fused MoE Gating** | Sparse routing + GEMM | CUDA reference + theory |

### Alternative 1: CUDA Reference Comparison

**When to use**: You have access to NVIDIA GPU for baseline measurement.

**Script**: `compare_vs_cuda.py`

```python
def benchmark_cuda(kernel_fn, input_cuda):
    # Measure on CUDA device
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        output = kernel_fn(input_cuda)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def benchmark_xpu(kernel_fn, input_xpu):
    # Measure on XPU device
    # Same pattern with torch.xpu.Event

# Compare
bw_cuda = benchmark_cuda(custom_kernel_cuda, input_cuda)
bw_xpu = benchmark_xpu(custom_kernel_xpu, input_xpu)
ratio = bw_xpu / bw_cuda

print(f"CUDA: {bw_cuda:.1f} GB/s ({bw_cuda/600*100:.1f}% of A100 peak)")
print(f"XPU:  {bw_xpu:.1f} GB/s ({bw_xpu/530*100:.1f}% of BMG peak)")
print(f"Architecture-normalized ratio: {ratio:.2f}")
```

**Acceptance**:
- ✅ Ratio ≥ 0.7: Good (accounting for architecture differences)
- ⚠️ Ratio < 0.5: Investigate (significant unexplained gap)

### Alternative 2: Vendor Library Baseline

**When to use**: Kernel has equivalent in oneDNN, MKL, or other vendor libraries.

**Examples**:
- GEMM → oneDNN `matmul` or MKL `sgemm`
- Convolution → oneDNN `convolution_forward`
- LayerNorm → oneDNN `layer_normalization`
- Softmax → oneDNN `softmax`

**Script**: `compare_vs_onednn.py`

```python
import intel_extension_for_pytorch as ipex

def benchmark_onednn(input_xpu):
    # Use oneDNN-optimized path
    output = torch.matmul(input_xpu, weight_xpu)  # Auto uses oneDNN
    # Measure bandwidth

def benchmark_custom(input_xpu):
    # Use custom kernel
    output = torch.ops.my_op_xpu.my_op(input_xpu)
    # Measure bandwidth

# Compare
print(f"oneDNN:       {bw_onednn:.1f} GB/s")
print(f"Custom SYCL:  {bw_custom:.1f} GB/s")
print(f"Ratio: {bw_custom/bw_onednn:.2f}")
```

**Acceptance**:
- ✅ Within 20% of oneDNN: Good (vendor libs are highly optimized)
- ⚠️ < 50% of oneDNN: Investigate

### Alternative 3: Roofline Theoretical Bounds

**When to use**: No CUDA baseline, no vendor library equivalent.

**Method**:
1. **Calculate Arithmetic Intensity**:
   ```
   AI = FLOPs_per_element / Bytes_per_element
   ```

2. **Determine Roofline**:
   ```
   Theoretical_Performance = min(
       Peak_FLOPS,
       Peak_Bandwidth * AI
   )
   ```

3. **Set target**: 70-80% of theoretical bound

**Example** (FlashAttention):
```
Hardware: BMG-G31
- Peak BF16 FLOPS: 183.5 TF/s
- Peak BW: 530 GB/s

FlashAttention (seq_len=1024, d=64):
- FLOPs per token: ~4 * d * seq_len = 262k FLOPs
- Bytes per token: ~8 * d * seq_len = 524k Bytes
- AI = 0.5 FLOP/Byte

Roofline:
- Compute bound: 183.5 TF/s
- Memory bound: 530 GB/s * 0.5 = 265 GF/s
- Bottleneck: Memory-bound (265 GF/s)

Target: 70% of 265 GF/s = 185 GF/s
```

**Script**: `roofline_analysis.py`

```python
def roofline_bound(flops, bytes, peak_flops_tfs, peak_bw_gbs):
    ai = flops / bytes
    compute_bound = peak_flops_tfs
    memory_bound = peak_bw_gbs * ai / 1000  # GB/s to TF/s
    return min(compute_bound, memory_bound)

# Calculate
theoretical = roofline_bound(flops=262e3, bytes=524e3, 
                              peak_flops_tfs=183.5, peak_bw_gbs=530)
target = theoretical * 0.7  # 70% efficiency target

print(f"Theoretical bound: {theoretical:.1f} GF/s")
print(f"Target (70%):      {target:.1f} GF/s")
print(f"Measured:          {measured:.1f} GF/s ({measured/target*100:.1f}%)")
```

**Acceptance**:
- ✅ ≥ 70% of roofline: Excellent
- ⚠️ < 50% of roofline: Investigate (memory/compute underutilization)

### Alternative 4: Accuracy-Only Validation

**When to use**: 
- No performance baseline available
- Research/experimental kernel
- Focus is on correctness, not performance

**Approach**:
1. **Validate numerical accuracy** (vs CUDA or reference implementation)
2. **Document current performance** (for future comparison)
3. **Skip performance optimization** until baseline becomes available

**Script**: `test_accuracy_only.py`

```python
def test_accuracy_vs_cuda():
    # Run on both platforms
    output_cuda = custom_kernel_cuda(input_cuda)
    output_xpu = custom_kernel_xpu(input_xpu)
    
    # Compare results
    diff = torch.abs(output_cuda.cpu() - output_xpu.cpu())
    assert torch.max(diff) < 1e-5, "Numerical mismatch"
    
    print("✅ Accuracy validated - numerically equivalent")
    print("⏸️  Performance optimization deferred (no baseline)")
```

---

## Summary Table

| Kernel Category | Baseline Method | Target Metric | Script Template |
|----------------|-----------------|---------------|-----------------|
| **Simple element-wise** (silu, gelu, etc.) | ATen (eager vs compiled) | Within 10% of torch.compile | `kernel_aten.py` |
| **Complex fusion** with CUDA ref | CUDA cross-platform | XPU/CUDA ratio ≥ 0.7 | `compare_vs_cuda.py` |
| **Standard ops** (gemm, conv, norm) | Vendor libraries (oneDNN) | Within 20% of oneDNN | `compare_vs_onednn.py` |
| **Novel algorithms** (FlashAttn, etc.) | Roofline theory | ≥ 70% of roofline bound | `roofline_analysis.py` |
| **Research/experimental** | Accuracy only | Numerical correctness | `test_accuracy_only.py` |

---

## Recommendations

### For Production Kernels
1. **Start with accuracy**: Always validate correctness first
2. **Choose appropriate baseline**: Use decision tree above
3. **Validate measurements**: Use unitrace for all performance claims
4. **Document limitations**: Note if no baseline exists

### For Research Kernels
1. **Accuracy is mandatory**: Numerical correctness required
2. **Performance is optional**: OK to defer optimization
3. **Document current state**: Record baseline performance for future

### For Migration Projects (CUDA → SYCL)
1. **Measure CUDA baseline first**: Benchmark on NVIDIA hardware
2. **Port to SYCL**: Semantic translation
3. **Compare cross-platform**: Calculate architecture-normalized ratio
4. **Iterate if needed**: Optimize only if ratio < 0.7

---

## Tools Summary

| Tool | Purpose | Required? |
|------|---------|-----------|
| `kernel_aten.py` | PyTorch native baseline | Only if ATen-expressible |
| `compare_vs_cuda.py` | Cross-platform comparison | If CUDA hardware available |
| `compare_vs_onednn.py` | Vendor library comparison | If oneDNN has equivalent |
| `roofline_analysis.py` | Theoretical bound | For novel algorithms |
| `validate_with_unitrace.py` | Measurement verification | **Always recommended** |
| `test_accuracy.py` | Correctness validation | **Always required** |

---

## Examples from This Repository

### silu_and_mul (ATen-expressible) ✅
- **Baseline**: `silu_and_mul_aten.py`
- **Method**: Compare vs torch.compile
- **Result**: 62.1 GB/s (custom) vs 65.0 GB/s (compiled) = 95.5% ✅
- **Validation**: unitrace confirms 1.3% timing accuracy ✅

### FlashAttention (NOT ATen-expressible) ❌
- **Baseline**: Would use CUDA reference OR roofline
- **Method**: Cross-platform comparison OR theoretical bound
- **No ATen script**: Cannot be expressed with simple PyTorch ops

---

## When in Doubt

**Priority**:
1. **Correctness first** - Always validate accuracy
2. **Choose best baseline** - Use decision tree
3. **Validate measurements** - Use unitrace
4. **Document clearly** - Note baseline method in README

**Never**:
- ❌ Force ATen for non-expressible kernels
- ❌ Skip accuracy validation
- ❌ Trust measurements without unitrace verification
- ❌ Compare apples to oranges (different workloads)
