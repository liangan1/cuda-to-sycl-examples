"""
silu_torch.py — Build & test the SiLU CUDA / SYCL PyTorch extensions.

Usage:
    python silu_torch.py            # auto-detects CUDA or XPU
"""
import torch
from torch.utils.cpp_extension import load


def build_cuda():
    return load(
        name="silu_cuda_ext",
        sources=["silu_cuda.cu"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def build_sycl():
    # Requires PyTorch >= 2.6 with XPU support (SyclExtension).
    from torch.utils.cpp_extension import SyclExtension, BuildExtension
    from torch.utils.cpp_extension import load as torch_load
    return torch_load(
        name="silu_sycl_ext",
        sources=["silu_sycl.cpp"],
        extra_cflags=["-O3"],
        extra_sycl_cflags=["-O3"],
        verbose=True,
        is_python_module=True,
    )


def test(ext, device):
    x = torch.randn(1 << 20, device=device, dtype=torch.float32)
    y = ext.silu_forward(x.contiguous())
    ref = torch.nn.functional.silu(x)
    err = (y - ref).abs().max().item()
    print(f"[{device}] max|y - ref| = {err:.3e}")
    assert err < 1e-5, "SiLU mismatch"


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("=== CUDA ===")
        test(build_cuda(), "cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("=== XPU (SYCL) ===")
        test(build_sycl(), "xpu")
