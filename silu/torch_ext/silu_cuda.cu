// silu_cuda.cu
// SiLU as a PyTorch CUDA extension (free-function kernel).
//
//   y = x * sigmoid(x) = x / (1 + exp(-x))
//
// Build (JIT via torch.utils.cpp_extension.load) or setup.py — see silu_torch.py.

#include <torch/extension.h>
#include <cuda_runtime.h>

// ---- Device kernel: plain __global__ free function -------------------------
template <typename scalar_t>
__global__ void silu_kernel(const scalar_t* __restrict__ x,
                            scalar_t*       __restrict__ y,
                            int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        scalar_t v = x[i];
        y[i] = v / (scalar_t(1) + ::exp(-v));
    }
}

// ---- C++ wrapper called from Python ----------------------------------------
torch::Tensor silu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(),       "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    const int64_t n = x.numel();

    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((n + BLOCK - 1) / BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "silu_forward", [&] {
        silu_kernel<scalar_t><<<grid, BLOCK, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            n);
    });
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_forward", &silu_forward, "SiLU forward (CUDA)");
}
