# silu_and_mul_verify README Summary

[README Summary]

## 1) CUDA kernel reference
- url: https://github.com/vllm-project/vllm (activation_kernels.cu)
- file: silu_and_mul_vllm_ref.cu:14-40
- function: silu_and_mul_kernel_vllm_ref
- license: Apache-2.0

## 2) Generated SYCL kernel location
- standalone file: silu_and_mul.sycl.cpp:18-49
- extension file: torch_ext/silu_and_mul_xpu.cpp:18-48
- function: silu_and_mul_kernel
- PyTorch op entry: torch_ext/silu_and_mul_xpu.cpp:50-70, function silu_and_mul_xpu

## 3) SYCL kernel analysis
- vectorization: YES
  - uses sycl::vec<float, 4> at torch_ext/silu_and_mul_xpu.cpp:29
  - uses vec load/store API at torch_ext/silu_and_mul_xpu.cpp:36-37 and 42
- memory_coalescing: GOOD
  - input layout is [tokens, 2*d], gate/up are contiguous slices
  - for each token, threads iterate vec_idx with stride=block, each iteration loads/stores contiguous float4 chunks
  - tail path handles remainder scalarly; only affects non-multiple-of-4 suffix
- launch_params:
  - block_size=512
  - grid_size=tokens
  - work_per_thread: strided loop over vec indices and scalar tail

## 4) Alignment
- semantic_alignment: PASS
  - formula aligned: out[i] = silu(gate[i]) * up[i]
  - validated by allclose against PyTorch reference
- launch_mapping_alignment: PASS
  - CUDA: grid=tokens, block=512
  - SYCL: nd_range global=tokens*512, local=512
  - mapping: blockIdx.x -> item.get_group(0), threadIdx.x -> item.get_local_id(0)

## 5) Accuracy and performance

### Accuracy
- cases=4
- max_abs_error=9.536743e-07
- max_rel_error=2.361176e-07
- status=PASS

### Performance setup
- device: Intel(R) Arc(TM) Pro B60 Graphics
- dtype: float32
- shape policy: xpu-perf style silu shapes (dim_size=1024)
- representative subset: batch_size=[1024, 8192, 65536]

### Performance table (time_ms, bandwidth_gbps, percent_peak)

| shape (batch, dim) | eager | compiled | custom |
|---|---:|---:|---:|
| (1024, 1024) | 0.032152 ms, 391.35 GB/s, 73.84% | 0.021012 ms, 598.85 GB/s, 112.99% | 0.011329 ms, 1110.70 GB/s, 209.57% |
| (8192, 1024) | 0.433555 ms, 232.18 GB/s, 43.81% | 0.257583 ms, 390.80 GB/s, 73.74% | 0.257841 ms, 390.41 GB/s, 73.66% |
| (65536, 1024) | 3.403662 ms, 236.60 GB/s, 44.64% | 2.031243 ms, 396.46 GB/s, 74.80% | 2.051072 ms, 392.63 GB/s, 74.08% |

Notes:
- Small-shape case can show >100% peak due to byte-model simplification and event-timing granularity.
- At medium/large shapes, custom is close to compiled and significantly faster than eager.
