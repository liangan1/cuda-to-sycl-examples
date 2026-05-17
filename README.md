# cuda-to-sycl-examples

Side-by-side CUDA ↔ SYCL kernel examples for engineers porting NVIDIA code
to Intel XPU. The thesis: porting is structurally mechanical — the
programming model, math, indexing, and memory model all map one-to-one;
only µarch tuning constants change.

## Layout

- [silu/](silu/) — SiLU (`y = x * sigmoid(x)`) demonstrated at four layers:
  1. [silu.cu](silu/silu.cu) + [silu.sycl.cpp](silu/silu.sycl.cpp) —
     bare free-function kernel + host driver. Body byte-for-byte identical.
  2. [silu/torch_ext/](silu/torch_ext/) — same kernel wrapped as a PyTorch
     C++ extension; only `at::cuda::*` ↔ `c10::xpu::*` differs.
  3. [silu/upstream/](silu/upstream/) — the **actual** code shipping in
     PyTorch + torch-xpu-ops today, placed side-by-side, plus a deeper
     concept mapping inside `gpu_kernel` (`vectorized_elementwise_kernel`
     ↔ `VectorizedElementwiseKernel`).
  4. [silu/vectorized/](silu/vectorized/) — standalone vectorized port
     (`float4` ↔ `sycl::vec<float, 4>`, tile-per-WG + scalar-tail) derived
     from PyTorch CUDA's `vectorized_elementwise_kernel`, with a SYCL
     event-profiling benchmark comparing naive vs vectorized on Intel XPU.

See [silu/README.md](silu/README.md) for the full narrative.

## License

Apache-2.0
