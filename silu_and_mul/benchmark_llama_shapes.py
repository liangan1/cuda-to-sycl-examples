#!/usr/bin/env python3
"""
Corrected benchmark with LLaMA production shapes.
Tests larger, more realistic workloads to see if bandwidth improves.
"""

import torch
import sys

try:
    import silu_and_mul_xpu
    HAS_CUSTOM = True
except:
    HAS_CUSTOM = False
    print("WARNING: Custom kernel not available")

def measure_bandwidth(fn, input_xpu, desc, iters=100, warmup=20):
    """Measure performance of a function"""
    # Warmup
    for _ in range(warmup):
        out = fn(input_xpu)
    torch.xpu.synchronize()
    
    # Timed runs
    times = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        
        start.record()
        out = fn(input_xpu)
        end.record()
        
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end))
    
    min_ms = min(times)
    
    # Calculate bandwidth correctly
    if "our custom" in desc.lower():
        # Input: [num_tokens, 2*d] FP32, Output: [num_tokens, d] FP32
        num_tokens, two_d = input_xpu.shape
        d = two_d // 2
        input_bytes = num_tokens * 2 * d * 4
        output_bytes = num_tokens * d * 4
        total_bytes = input_bytes + output_bytes
    elif "silu_mul_unfused" in desc.lower():
        # Unfused: read input [N, 2*d], write output [N, d]
        num_tokens, two_d = input_xpu.shape
        d = two_d // 2
        total_bytes = (num_tokens * 2 * d + num_tokens * d) * 4
    else:
        # Element-wise: input + output
        total_bytes = input_xpu.numel() * 4 * 2
    
    bw_gbps = (total_bytes / 1e9) / (min_ms / 1000.0)
    peak_bw = 530.0  # BMG-G31 HBM peak
    pct_peak = (bw_gbps / peak_bw) * 100
    
    print(f"{desc:45s} | {min_ms:8.3f} ms | {bw_gbps:7.1f} GB/s | {pct_peak:5.1f}%")
    return bw_gbps

def test_large_shapes():
    """Test with LLaMA production shapes"""
    print("=" * 105)
    print("CORRECTED Benchmark: LLaMA Production Shapes")
    print("=" * 105)
    print(f"{'Operation':<45} | {'Time':>11} | {'Bandwidth':>12} | {'% Peak':>5}")
    print("-" * 105)
    
    # LLaMA shapes: d = 11008, 14336 (FFN intermediate dimension)
    shapes = [
        (1024, 11008 * 2, "LLaMA-3.1-8B batch=1024"),
        (4096, 11008 * 2, "LLaMA-3.1-8B batch=4096"),
        (8192, 11008 * 2, "LLaMA-3.1-8B batch=8192"),
        (16384, 11008 * 2, "LLaMA-3.1-8B batch=16K"),
        (32768, 11008 * 2, "LLaMA-3.1-8B batch=32K"),
        (8192, 14336 * 2, "LLaMA-3.1-70B batch=8192"),
    ]
    
    for num_tokens, two_d, label in shapes:
        d = two_d // 2
        print(f"\n[{label}: shape=[{num_tokens}, {two_d}], d={d}]")
        
        # Create input
        input_xpu = torch.randn(num_tokens, two_d, device='xpu', dtype=torch.float32)
        gate = input_xpu[:, :d]
        up = input_xpu[:, d:]
        
        # Test 1: PyTorch native SiLU
        silu_fn = lambda x: torch.nn.functional.silu(x)
        measure_bandwidth(silu_fn, gate, "PyTorch F.silu(gate)")
        
        # Test 2: PyTorch unfused SiLU + mul
        def pytorch_silu_mul(x):
            g = x[:, :d]
            u = x[:, d:]
            return torch.nn.functional.silu(g) * u
        measure_bandwidth(pytorch_silu_mul, input_xpu, "PyTorch silu_mul_unfused")
        
        # Test 3: Our custom kernel
        if HAS_CUSTOM:
            custom_fn = lambda x: torch.ops.silu_and_mul_xpu.silu_and_mul(x)
            measure_bandwidth(custom_fn, input_xpu, "Our custom fused kernel")
    
    print("=" * 105)
    print("\nKey Insights:")
    print("1. Larger shapes (d=11008, 14336) should show better bandwidth utilization")
    print("2. If PyTorch native still shows ~10-15% peak, it's a stack limitation")
    print("3. Our fused kernel should consistently outperform unfused PyTorch ops")

if __name__ == "__main__":
    if not torch.xpu.is_available():
        print("ERROR: XPU not available!")
        sys.exit(1)
    
    test_large_shapes()
