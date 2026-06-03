# CUDA -> SYCL Translation Skills (Compact)

## 1. Purpose
Translate CUDA kernels to Intel SYCL with:
- semantic equivalence first,
- correctness validation,
- reproducible performance reporting.

Target: standalone SYCL kernel + PyTorch XPU C++ extension.

## 2. Scope
Best fit:
- element-wise/fused memory-bound kernels (silu/gelu/add/mul/simple norm)
- utility kernels in LLM inference path

Usually not ATen-expressible:
- FlashAttention-like kernels
- fused GEMM+epilogue kernels
- complex quantization/layout kernels

## 3. Must-Follow Rules
1. Queue API:
- Use `c10::xpu::getCurrentXPUStream(device_idx).queue()`
- Do not use nonexistent stream->queue helpers

2. Sync:
- Do not call `q.wait()` in extension op path
- PyTorch stream semantics should handle sync

3. Launch mapping:
- CUDA `<<<grid, block>>>` -> SYCL `nd_range{grid * block, block}`
- Global size must be `grid * block`

4. Intel defaults:
- Start with `block_size=512` for Xe element-wise kernels
- Avoid copying AMD/CUDA tuning assumptions directly

5. Vectorization:
- Prefer `sycl::vec<T,4>` with `.load()`/`.store()`
- Avoid unsafe reinterpret-cast vector memory ops

---

## 4. CUDA -> SYCL Mapping
### 4.1 Signature
CUDA:
```cpp
__global__ void k(args...) { ... }
```
SYCL free-function:
```cpp
SYCL_EXTERNAL
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((sycl::ext::oneapi::experimental::nd_range_kernel<1>))
void k(args...) { ... }
```

### 4.2 Indexing
- `blockIdx.x` -> `item.get_group(0)`
- `threadIdx.x` -> `item.get_local_id(0)`
- `blockDim.x` -> `item.get_local_range(0)`

Pattern:
```cpp
auto item = syclwi::get_nd_item<1>();
for (int i = item.get_local_id(0); i < N; i += item.get_local_range(0)) {
  ...
}
```

### 4.3 Math
- `expf(x)` -> `sycl::exp(x)`
- optional fast path: `sycl::native::exp(x)` after numerical validation

### 4.4 Launch
```cpp
sycl::nd_range<1> ndr{grid * block, block};
syclexp::nd_launch(q, ndr, syclexp::kernel_function<k>, args...);
```

---

## 5. Minimal Workflow
### Step 0 (conditional baseline)
If ATen-expressible (e.g., `silu_and_mul`):
- measure eager + `torch.compile` + customized SYCL kernel
- use `torch.compile` as target ceiling
- performance section must include all three rows: `eager`, `compiled`, `custom`

If not ATen-expressible:
- compare against CUDA reference and/or vendor library and/or roofline

### Step 1 Standalone SYCL correctness
- implement `kernel.sycl.cpp`
- build/run with `icpx -fsycl -O3 ...`
- compare numerically with CPU/CUDA reference

### Step 2 PyTorch extension integration
Headers:
```cpp
#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
```
Queue:
```cpp
c10::xpu::XPUStream s = c10::xpu::getCurrentXPUStream(input.device().index());
sycl::queue& q = s.queue();
```
Registration:
```cpp
TORCH_LIBRARY(my_op_xpu, m) { m.def("my_op(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(my_op_xpu, XPU, m) { m.impl("my_op", &my_op_xpu); }
```

### Step 3 Build script essentials
- `CXX=icpx`
- compile flags: `-fsycl -O3 -fsycl-targets=spir64`
- link flags: `-fsycl`

### Step 4 Accuracy testing
- >= 20 representative shape cases recommended
- fp32 tolerance: `rtol=1e-5, atol=1e-6`

### Step 5 Performance benchmarking
- use `torch.xpu.Event` timing
- report `time_ms`, `bandwidth_gbps`, `%peak`
- if ATen exists, benchmark and report all of: `eager`, `compiled`, `custom`
- performance shapes should reference bytedance/xpu-perf style configs when possible
- for `silu_and_mul`, use `silu`-style shapes as default reference:
  - `dim_size = 1024`
  - `batch_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]`
  - if runtime is long, run a representative subset (for example: `1024, 8192, 65536`) and document it

Bandwidth:
```text
BW(GB/s) = (bytes_read + bytes_written) / time_s / 1e9
```

### Step 6 Timing validation (recommended)
- validate with Intel PTI-GPU `unitrace`
- target Event vs device timing diff < 2%

### Step 7 Performance analysis (required)
- Analyze at least 3 representative shapes: small / medium / production-like
- Include at least one baseline:
  - ATen-expressible kernel: eager + `torch.compile`
  - non-ATen kernel: CUDA and/or vendor library and/or roofline estimate
- Report speedup and gap:
  - `speedup_vs_eager = time_eager / time_custom`
  - `gap_vs_target = (bw_target - bw_custom) / bw_target`
- Give a conclusion label per shape:
  - `PASS`: within 10% of target baseline
  - `TUNE`: 10-25% below target
  - `INVESTIGATE`: >25% below target or unstable timing
- If `%peak > 100%`, mark as model/traffic over-estimate and explain byte model.

### Step 8 README summary report (required)
After kernel generation and validation, `README.md` must include a concise summary report with all items below:
1. CUDA kernel reference:
  - source URL (clickable)
  - source file path + line range
  - function name
2. Generated SYCL kernel location:
  - local file path
  - function entry name(s)
3. SYCL kernel analysis:
  - whether vectorization is used (for example `sycl::vec<T,4>`) and where
  - memory access/coalescing status
  - key launch parameters (`block_size`, `grid_size`, work per thread)
4. Alignment result:
  - semantic alignment result (PASS/FAIL + notes)
  - launch grid mapping alignment result (CUDA grid/block vs SYCL nd_range)
5. Accuracy and performance data:
  - accuracy summary (`cases`, `max_abs_error`, `max_rel_error`, status)
  - performance table (ATen case must contain `eager`, `compiled`, `custom`)
  - representative shape set (xpu-perf style) and device info

---

## 6. Required Output Format
### 6.1 Kernel Origin
- source repository URL
- file path + line range
- function name
- license

### 6.2 CUDA <-> SYCL Mapping Table
Include at least:
- signature mapping
- thread indexing mapping
- vectorization mapping (if used)
- launch mapping

### 6.3 Correctness Report Template
```text
[Accuracy]
cases: <N>
max_abs_error: <value>
max_rel_error: <value>
status: PASS|FAIL
```

### 6.4 Performance Report Template
```text
[Performance]
device: <name>
shape: <...>
impl: eager|compiled|custom
time_ms: <...>
bandwidth_gbps: <...>
percent_peak: <...>
```

If ATen implementation exists, this section is mandatory in table form with 3 rows:
```text
shape=<...>
eager:    time_ms=<...>, bandwidth_gbps=<...>, percent_peak=<...>
compiled: time_ms=<...>, bandwidth_gbps=<...>, percent_peak=<...>
custom:   time_ms=<...>, bandwidth_gbps=<...>, percent_peak=<...>
```

### 6.5 Timing Validation Template (if unitrace used)
```text
[Timing Validation]
event_ms: <...>
unitrace_ms: <...>
diff_percent: <...>
status: VALIDATED|CHECK_REQUIRED
```

### 6.6 Performance Analysis Template
```text
[Performance Analysis]
target_baseline: eager|compiled|cuda|vendor|roofline
shape: <...>
speedup_vs_eager: <...>
gap_vs_target_percent: <...>
label: PASS|TUNE|INVESTIGATE
notes: <brief bottleneck or byte-model explanation>
```

### 6.7 README Summary Template (mandatory)
```text
[README Summary]
1) CUDA kernel reference
- url: <https://...>
- file: <path>:<start>-<end>
- function: <name>

2) Generated SYCL kernel location
- file: <local path>
- function: <name>

3) SYCL kernel analysis
- vectorization: YES|NO (details)
- memory_coalescing: GOOD|PARTIAL|POOR (details)
- launch_params: block_size=<...>, grid_size=<...>, work_per_thread=<...>

4) Alignment
- semantic_alignment: PASS|FAIL (details)
- launch_mapping_alignment: PASS|FAIL
  - cuda: grid=<...>, block=<...>
  - sycl: global=<...>, local=<...>

5) Accuracy and performance
- accuracy: cases=<...>, max_abs_error=<...>, max_rel_error=<...>, status=PASS|FAIL
- performance_table:
  - shape=<...>: eager=<...>, compiled=<...>, custom=<...>
  - shape=<...>: eager=<...>, compiled=<...>, custom=<...>
- device: <name>
```

---

## 7. Acceptance Criteria
With ATen baseline:
- Good: custom within 10% of `torch.compile`
- Investigate: custom >10% slower than `torch.compile`
- Bug: custom slower than unfused eager (unless justified)

Without ATen baseline:
- use CUDA/library/roofline comparison; correctness first

Always fail if:
- tolerance violation,
- invalid launch mapping,
- queue API misuse,
- explicit sync misuse in extension path,
- missing required `README.md` summary report fields.

---

## 8. Minimal Layout
```text
my_kernel/
  my_kernel.cu
  my_kernel.sycl.cpp
  torch_ext/
    my_kernel_xpu.cpp
    setup.py
  test_accuracy.py
  bench_xpu_perf.py
  README.md
```
Optional: `my_kernel_aten.py`, `validate_with_unitrace.py`, `compare_vs_cuda.py`.

## 9. References
- bytedance/xpu-perf:
  https://github.com/bytedance/xpu-perf
- SYCL free-function kernels:
  https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_free_function_kernels.asciidoc
- PyTorch XPU notes:
  https://pytorch.org/docs/stable/notes/get_start_xpu.html
- PyTorch C++ extension:
  https://pytorch.org/tutorials/advanced/cpp_extension.html
- Intel PTI-GPU / unitrace:
  https://github.com/intel/pti-gpu
- vLLM:
  https://github.com/vllm-project/vllm

## 10. Pre-Merge Checklist
- [ ] origin + license documented
- [ ] CUDA<->SYCL mapping table included
- [ ] accuracy tests pass
- [ ] benchmark output uses required format
- [ ] performance analysis section included with PASS/TUNE/INVESTIGATE labels
- [ ] at least 3 representative shapes benchmarked
- [ ] if ATen exists: eager/compiled/custom performance comparison included
- [ ] no explicit `q.wait()` in extension path
- [ ] correct PyTorch XPU queue usage
