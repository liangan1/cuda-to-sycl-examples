# cuda-to-sycl-examples

Side-by-side **CUDA ↔ SYCL** kernel examples for engineers porting NVIDIA code
to Intel XPU. The thesis: porting is structurally mechanical — the programming
model, math, indexing, and memory model all map one-to-one; only µarch tuning
constants change. **All contents in this repo are gernated by AI**. 

## Usage 
You can use the skills/SKILLS.md to translate the CUDA elementwise kernel into the sycl. a refrence ben found in [silu_and_mul_verify](https://github.com/liangan1/cuda-to-sycl-examples/tree/main/silu_and_mul_verify)

## CUDA <-> SYCL Mapping (Line-by-Line, Two Columns)

The mapping below is code-first: each row shows one CUDA line and its SYCL equivalent.
Comments are kept concise and only added where they clarify semantics.

### 1) Kernel Semantic Mapping (Code in Two Columns)

| CUDA | SYCL |
|---|---|
| `template <typename T, int VEC_SIZE> __global__ void silu_and_mul_kernel_vllm_ref(T* out, const T* input, int d) {` | `template <int VEC_SIZE> void silu_and_mul_kernel(float* out, const float* input, int d) { // nd_range kernel body` |
| `int token = blockIdx.x; // block maps to one token` | `const int token = static_cast<int>(item.get_group(0)); // work-group maps to one token` |
| `int tid = threadIdx.x; // lane in block` | `const int tid = static_cast<int>(item.get_local_id(0)); // lane in work-group` |
| `const int block = blockDim.x; // threads per block` | `const int block = static_cast<int>(item.get_local_range(0)); // work-items per group` |
| `const T* gate = input + token * 2 * d;` | `const float* gate = input + token * 2 * d;` |
| `const T* up = gate + d;` | `const float* up = gate + d;` |
| `T* out_ptr = out + token * d;` | `float* out_ptr = out + token * d;` |
| `int num_vec = d / VEC_SIZE;` | `const int num_vec = d / VEC_SIZE;` |
| `using vec_t = float4; // vectorized load/store` | `using vec_t = sycl::vec<float, VEC_SIZE>; // vectorized load/store` |
| `for (int vec_idx = tid; vec_idx < num_vec; vec_idx += blockDim.x) { /* vector path */ }` | `for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block) { /* vector path */ }` |
| `for (int i = num_vec * VEC_SIZE + tid; i < d; i += blockDim.x) out_ptr[i] = silu_cuda(gate[i]) * up[i]; // tail` | `for (int i = num_vec * VEC_SIZE + tid; i < d; i += block) out_ptr[i] = silu_sycl(gate[i]) * up[i]; // tail` |
| `}` | `}` |

### 2) Launch Grid Mapping (Code in Two Columns)

| CUDA | SYCL |
|---|---|
| `dim3 grid(tokens); // one block per token` | `constexpr int block = 512;` |
| `dim3 block(512); // 512 threads per block` | `sycl::nd_range<1> ndr{tokens * block, block}; // global, local` |
| `silu_and_mul_kernel_vllm_ref<float, 4><<<grid, block>>>(out, input, d);` | `sycl::ext::oneapi::experimental::nd_launch(q, ndr, sycl::ext::oneapi::experimental::kernel_function<silu_and_mul_kernel<4>>, d_out, d_in, d);` |

### Quick Interpretation

- `blockIdx.x` maps to `item.get_group(0)`.
- `threadIdx.x` maps to `item.get_local_id(0)`.
- `blockDim.x` maps to `item.get_local_range(0)`.
- `<<<grid, block>>>` maps to `nd_range(global=tokens*block, local=block)`.
- The compute equation and work partition stay unchanged; only API/launch syntax changes.
## License

Apache-2.0
