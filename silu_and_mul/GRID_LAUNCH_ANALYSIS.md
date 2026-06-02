# CUDA Grid/Block Launch Configuration Analysis

## vLLM/AIter Activation Kernel Launch Strategy

### Grid & Block Configuration (from activation_kernels.cu)

```cpp
#define COMPUTE_ACTIVATION_KERNEL_PARAMS
    int d = input.size(-1) / 2;                          // Hidden dimension
    int64_t num_tokens = input.numel() / input.size(-1); // Batch size
    
    // Step 1: Compute vec_size (vectorization factor)
    int vec_size = nextPow2(d / 64);
    vec_size = vec_size < 2 ? 2 : vec_size;
    vec_size = vec_size > max_vec_size ? max_vec_size : vec_size;  // max = 8
    
    // Step 2: Compute num_wave (number of wavefronts per block)
    int num_wave = nextPow2(d / 64 / vec_size);
    num_wave = num_wave > max_wave_num ? max_wave_num : num_wave;  // max = 8
    
    // Step 3: Launch configuration
    dim3 grid(num_tokens);           // 1 block per token
    dim3 block(num_wave * 64);       // num_wave * wavefront_size
```

---

## Key Design Decisions

### 1. **Grid Size: 1 Block per Token**
```cpp
dim3 grid(num_tokens);
```

**Rationale**:
- Each block processes 1 row of input (1 token)
- Natural parallelism: N tokens → N blocks
- No inter-block synchronization needed
- Simplifies kernel logic (no multi-token handling)

**Trade-off**:
- Small batch → few blocks → poor GPU utilization
- Launch overhead dominates for batch < 1024

---

### 2. **Block Size: Dynamic Based on Hidden Dimension**

```cpp
dim3 block(num_wave * 64);
```

**Calculation**:
| Hidden Dim (d) | vec_size | num_wave | Block Size | Threads per Token |
|----------------|----------|----------|------------|-------------------|
| 1024           | 2        | 8        | 512        | 512               |
| 4096           | 8        | 8        | 512        | 512               |
| 11008 (LLaMA)  | 16       | 8        | 512        | 512               |
| 14336 (LLaMA)  | 16       | 8        | 512        | 512               |

**Formula**:
```
vec_size = nextPow2(d / 64)  // Clamp to [2, 8]
num_wave = nextPow2(d / (64 * vec_size))  // Clamp to [1, 8]
block_size = num_wave * 64
```

**Example** (d = 11008):
```
vec_size = nextPow2(11008 / 64) = nextPow2(172) = 128 → clamped to 8
num_wave = nextPow2(11008 / (64 * 8)) = nextPow2(21.5) = 32 → clamped to 8
block_size = 8 * 64 = 512
```

**Why 64?**
- AMD wavefront size = 64 threads (ROCm/HIP)
- NVIDIA warp size = 32 threads (CUDA)
- Aligns with hardware execution units

---

### 3. **Vectorization Strategy**

**vec_size adapts to hidden dimension**:
- Small dim (1024): vec_size = 2 → process 2 floats per iteration
- Medium dim (4096): vec_size = 8 → process 8 floats per iteration
- Large dim (11008+): vec_size = 16 → process 16 floats per iteration

**Benefits**:
- Larger dim → higher vectorization → better memory bandwidth utilization
- Adaptive strategy balances occupancy and memory throughput

---

## Comparison: Current Implementation vs vLLM/AIter

### Current (Our SYCL Implementation)
```cpp
constexpr int BLOCK_SIZE = 256;  // Fixed
constexpr int VEC_SIZE = 4;       // Fixed
dim3 grid(num_tokens);
dim3 block(BLOCK_SIZE);
```

**Issues**:
1. **Fixed block size** (256) may be suboptimal for different dims
2. **Fixed vec_size** (4) misses opportunity for higher vectorization
3. **No adaptation** to workload characteristics

### vLLM/AIter (Dynamic Configuration)
```cpp
int vec_size = adaptive(d);       // 2, 4, 8, or 16
int num_wave = adaptive(d);       // 1-8
dim3 block(num_wave * 64);        // 64-512
```

**Advantages**:
1. **Adaptive** to hidden dimension
2. **Higher vectorization** for large dims (LLaMA: vec_size=16)
3. **Optimized occupancy** via num_wave tuning

---

## Performance Impact

### Test Case: dim=1024 (our benchmark)

**Current Implementation**:
- Block size: 256 (fixed)
- VEC_SIZE: 4 (fixed)
- Threads per element: 256 / (1024/4) = 1 thread per vector
- **Result**: 66.44 GB/s (12.5% peak)

**vLLM Configuration**:
- vec_size: 2 (nextPow2(1024/64) = 16 → clamped to min 2)
- num_wave: 8 (nextPow2(1024/128) = 8)
- Block size: 8 * 64 = 512
- **Expected**: Higher occupancy, better memory utilization

### Test Case: dim=11008 (LLaMA-3.1-8B FFN)

**Current Implementation**:
- Block size: 256 (fixed)
- VEC_SIZE: 4 (fixed)
- Work per thread: 11008 / (256*4) = 10.75 vectors
- **Problem**: Low vectorization efficiency for large dim

**vLLM Configuration**:
- vec_size: 8 or 16 (adaptive)
- num_wave: 8
- Block size: 512
- Work per thread: 11008 / (512*16) = 1.34 vectors
- **Benefit**: 4x higher vectorization (16 vs 4)

---

## Optimization Recommendations for SYCL

### 1. **Implement Adaptive Block Size**
```cpp
// Replace fixed BLOCK_SIZE = 256
int compute_block_size(int d) {
  int vec_size = std::min(16, next_pow2(d / 64));
  vec_size = std::max(2, vec_size);
  
  int num_wave = next_pow2(d / (64 * vec_size));
  num_wave = std::min(8, num_wave);
  
  return num_wave * 64;  // Align to SYCL sub-group size (16 or 32)
}
```

### 2. **Implement Adaptive Vectorization**
```cpp
// Replace fixed VEC_SIZE = 4
template <int VEC_SIZE>
void launch_kernel(sycl::queue& q, int d, int num_tokens) {
  int block_size = compute_block_size(d);
  sycl::nd_range<1> ndr{num_tokens * block_size, block_size};
  nd_launch(q, ndr, kernel_func<VEC_SIZE>, ...);
}

// Dispatch at runtime
if (d >= 8192) {
  launch_kernel<16>(...);  // Large FFN (LLaMA-70B)
} else if (d >= 4096) {
  launch_kernel<8>(...);   // Medium FFN (LLaMA-7B/13B)
} else {
  launch_kernel<4>(...);   // Small FFN
}
```

### 3. **Expected Performance Improvement**

**dim=1024** (current benchmark):
- Current: 256 threads, vec_size=4 → 66.44 GB/s (12.5% peak)
- Optimized: 512 threads, vec_size=2 → **Expected: ~120 GB/s (23% peak)**

**dim=11008** (LLaMA-3.1-8B):
- Current: 256 threads, vec_size=4 → **Expected: ~80 GB/s (15% peak)**
- Optimized: 512 threads, vec_size=16 → **Expected: ~320 GB/s (60% peak)**

---

## Why vLLM Uses This Strategy

### 1. **Hardware Characteristics**
- AMD: Wavefront size = 64, max 8 wavefronts per CU
- NVIDIA: Warp size = 32, max 64 warps per SM
- Dynamic tuning matches hardware execution model

### 2. **Workload Diversity**
- Small models: dim ~ 1024-2048
- Large models: dim ~ 8192-14336
- One-size-fits-all (BLOCK_SIZE=256) is suboptimal

### 3. **Memory Bandwidth Optimization**
- Larger vec_size → fewer memory transactions
- Higher num_wave → better latency hiding
- Adaptive strategy maximizes throughput across workloads

---

## Action Items for Our SYCL Implementation

1. ✅ **Current**: Fixed BLOCK_SIZE=256, VEC_SIZE=4
2. ⏭️ **Next**: Implement adaptive block size and vec_size
3. ⏭️ **Benchmark**: Test on dims [1024, 4096, 8192, 11008, 14336]
4. ⏭️ **Profile**: Use Intel VTune to verify memory bandwidth saturation
5. ⏭️ **Compare**: Measure against PyTorch native ops

---

## References

- **Source**: `/home/liangan1/pytorch/third_party/aiter/csrc/kernels/activation_kernels.cu`
- **vLLM CUDA**: https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu
- **AMD ROCm**: Wavefront size = 64
- **NVIDIA CUDA**: Warp size = 32
- **Intel Xe**: Sub-group size = 16 (BMG) or 32 (PVC)

---

Last updated: 2026-06-02
