#!/usr/bin/env python3
"""
Tensor-level accuracy test for SiLU-and-Mul fusion kernel

Compares XPU custom op against PyTorch CPU reference implementation
Tests multiple shapes and dtypes to ensure correctness

Usage:
    python test_accuracy.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys

try:
    import silu_and_mul_xpu
    HAS_CUSTOM_OP = True
except ImportError:
    print("Warning: silu_and_mul_xpu extension not found. Build it with:")
    print("  cd torch_ext && python setup.py install")
    HAS_CUSTOM_OP = False


def silu_and_mul_ref(input_tensor):
    """
    Reference implementation using PyTorch ops
    
    Args:
        input_tensor: [num_tokens, 2 * d]
    
    Returns:
        output: [num_tokens, d]
    """
    num_tokens, hidden_dim_x2 = input_tensor.shape
    assert hidden_dim_x2 % 2 == 0
    d = hidden_dim_x2 // 2
    
    gate = input_tensor[:, :d]      # [num_tokens, d]
    up = input_tensor[:, d:]        # [num_tokens, d]
    
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # F.silu is the PyTorch builtin
    out = F.silu(gate) * up
    
    return out


def test_accuracy(num_tokens, d, dtype=torch.float32, device='xpu', rtol=1e-5, atol=1e-6):
    """
    Test accuracy for a given shape and dtype
    
    Args:
        num_tokens: batch size
        d: hidden dimension
        dtype: torch dtype
        device: 'xpu' or 'cpu'
        rtol: relative tolerance
        atol: absolute tolerance
    
    Returns:
        bool: True if test passes
    """
    # Generate random input
    torch.manual_seed(42)
    input_cpu = torch.randn(num_tokens, 2 * d, dtype=dtype)
    
    # CPU reference
    ref_output = silu_and_mul_ref(input_cpu)
    
    if device == 'xpu' and HAS_CUSTOM_OP:
        # XPU custom op
        input_xpu = input_cpu.to('xpu')
        xpu_output = torch.ops.silu_and_mul_xpu.silu_and_mul(input_xpu)
        xpu_output_cpu = xpu_output.cpu()
        
        # Compare
        max_diff = torch.max(torch.abs(xpu_output_cpu - ref_output)).item()
        rel_diff = torch.max(torch.abs((xpu_output_cpu - ref_output) / (ref_output + 1e-10))).item()
        
        passed = torch.allclose(xpu_output_cpu, ref_output, rtol=rtol, atol=atol)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} | Shape [{num_tokens:6}, {d:5}] | dtype {str(dtype):13} | "
              f"max_diff {max_diff:.2e} | rel_diff {rel_diff:.2e}")
        
        return passed
    else:
        print(f"SKIP  | Shape [{num_tokens:6}, {d:5}] | dtype {str(dtype):13} | "
              f"XPU custom op not available")
        return True


def run_all_tests():
    """Run comprehensive accuracy tests"""
    
    print("=" * 80)
    print("SiLU-and-Mul Tensor-Level Accuracy Test")
    print("=" * 80)
    print()
    
    if not HAS_CUSTOM_OP:
        print("ERROR: Custom op not found. Please build the extension first.")
        print("  cd torch_ext && python setup.py install")
        return False
    
    # Check XPU availability
    if not torch.xpu.is_available():
        print("ERROR: XPU device not available")
        return False
    
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print()
    
    # Test shapes (following xpu-perf format)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    dim_sizes = [1024, 4096]
    dtypes = [torch.float32]  # TODO: Add float16, bfloat16 support
    
    all_passed = True
    
    for dtype in dtypes:
        for d in dim_sizes:
            for num_tokens in batch_sizes:
                passed = test_accuracy(num_tokens, d, dtype=dtype)
                all_passed = all_passed and passed
    
    print()
    print("=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
