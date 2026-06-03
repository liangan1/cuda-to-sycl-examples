# cuda-to-sycl-examples

Side-by-side **CUDA ↔ SYCL** kernel examples for engineers porting NVIDIA code
to Intel XPU. The thesis: porting is structurally mechanical — the programming
model, math, indexing, and memory model all map one-to-one; only µarch tuning
constants change. **All contents in this repo are gernated by AI**. 

## Usage 
You can use the skills/SKILLS.md to translate the CUDA elementwise kernel into the sycl. a refrence ben found in [silu_and_mul_verify](https://github.com/liangan1/cuda-to-sycl-examples/tree/main/silu_and_mul_verify)

## CUDA <-> SYCL Mapping (Code + Explanation)

下面不是纯表格，而是直接贴出关键代码并说明一一对应关系。

### 1) Kernel 语义 mapping（代码对照）

CUDA (来自 `silu_and_mul_vllm_ref.cu`):

```cpp
template <typename T, int VEC_SIZE>
__global__ void silu_and_mul_kernel_vllm_ref(T* out, const T* input, int d) {
	int token = blockIdx.x;
	int tid = threadIdx.x;

	const T* gate = input + token * 2 * d;
	const T* up = gate + d;
	T* out_ptr = out + token * d;

	int num_vec = d / VEC_SIZE;
	using vec_t = float4;

	for (int vec_idx = tid; vec_idx < num_vec; vec_idx += blockDim.x) {
		// vector path
	}
	for (int i = num_vec * VEC_SIZE + tid; i < d; i += blockDim.x) {
		out_ptr[i] = silu_cuda(gate[i]) * up[i];
	}
}
```

SYCL (来自 `silu_and_mul.sycl.cpp`):

```cpp
template <int VEC_SIZE>
void silu_and_mul_kernel(float* out, const float* input, int d) {
	auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
	const int token = static_cast<int>(item.get_group(0));
	const int tid = static_cast<int>(item.get_local_id(0));
	const int block = static_cast<int>(item.get_local_range(0));

	const float* gate = input + token * 2 * d;
	const float* up = gate + d;
	float* out_ptr = out + token * d;

	const int num_vec = d / VEC_SIZE;
	using vec_t = sycl::vec<float, VEC_SIZE>;

	for (int vec_idx = tid; vec_idx < num_vec; vec_idx += block) {
		// vector path
	}
	for (int i = num_vec * VEC_SIZE + tid; i < d; i += block) {
		out_ptr[i] = silu_sycl(gate[i]) * up[i];
	}
}
```

说明（逐项对应）:

| CUDA | SYCL |
|---|---|
| `__global__` kernel | `nd_range` kernel function |
| `blockIdx.x` | `item.get_group(0)` |
| `threadIdx.x` | `item.get_local_id(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `float4` 向量读写 | `sycl::vec<float, VEC_SIZE>` 向量读写 |
| 标量尾部循环 (`i += blockDim.x`) | 标量尾部循环 (`i += block`) |
| `silu_cuda(gate[i]) * up[i]` | `silu_sycl(gate[i]) * up[i]` |

### 2) Launch grid mapping（代码对照）

CUDA launch:

```cpp
dim3 grid(tokens);
dim3 block(512);
silu_and_mul_kernel_vllm_ref<float, 4><<<grid, block>>>(out, input, d);
```

SYCL launch:

```cpp
constexpr int block = 512;
sycl::nd_range<1> ndr{tokens * block, block};
sycl::ext::oneapi::experimental::nd_launch(
		q,
		ndr,
		sycl::ext::oneapi::experimental::kernel_function<silu_and_mul_kernel<4>>,
		d_out,
		d_in,
		d);
```

说明（逐项对应）:

| CUDA | SYCL |
|---|---|
| `grid.x = tokens` | `global_range = tokens * block` |
| `block.x = 512` | `local_range = 512` |
| `<<<grid, block>>>` | `nd_range(global, local)` |
| 1 block 负责 1 token | 1 work-group 负责 1 token |
| block 内线程做 stride 循环 | group 内 work-item 做 stride 循环 |

一句话总结: 对这个 elementwise kernel，CUDA -> SYCL 的迁移本质是把 block/thread 的索引 API 换成 group/local_id API，把 launch 语法换成 `nd_range`，计算语义和访存分工保持不变。
## License

Apache-2.0
