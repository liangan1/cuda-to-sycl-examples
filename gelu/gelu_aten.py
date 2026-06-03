#!/usr/bin/env python3
"""
PyTorch ATen-level reference implementation of GELU.

GELU formula: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

This serves as the accuracy and performance baseline for custom kernel development.
Provides three implementations:
1. Eager mode (unfused): Separate operations
2. Eager mode (F.gelu): PyTorch native GELU
3. torch.compile: Compiler fusion attempt

Following SKILLS.md Step 0: ATen Baseline
"""

import torch
import torch.nn.functional as F
import math

def gelu_unfused(x: torch.Tensor) -> torch.Tensor:
    """
    Unfused implementation: compute GELU with separate operations.
    
    Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    This is the naive baseline - PyTorch will execute multiple kernel launches.
    Expected bandwidth: Lower due to multiple HBM round-trips.
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    tanh_inner = torch.tanh(inner)
    return 0.5 * x * (1.0 + tanh_inner)

def gelu_eager(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch native GELU (eager mode).
    
    Uses F.gelu() which is optimized in PyTorch's C++ backend.
    Expected: Best performance for ATen baseline.
    """
    return F.gelu(x, approximate='tanh')  # Use tanh approximation (same as formula above)

@torch.compile
def gelu_compiled(x: torch.Tensor) -> torch.Tensor:
    """
    torch.compile version - attempts to fuse operations.
    
    Expected: Similar to F.gelu() or slightly better if compiler optimizes.
    """
    return F.gelu(x, approximate='tanh')

def benchmark_aten_impl(device='xpu', dtype=torch.float32, warmup=10, iters=100):
    """
    Benchmark all three ATen implementations.
    
    Following SKILLS.md Step 0 workflow.
    """
    # Test shape: 1M elements (4 MB input + 4 MB output = 8 MB total)
    N = 1024 * 1024
    input_tensor = torch.randn(N, device=device, dtype=dtype)
    
    # Calculate theoretical bandwidth
    input_bytes = N * 4  # float32
    output_bytes = N * 4
    total_gb = (input_bytes + output_bytes) / 1e9
    
    results = {}
    
    # Test each implementation
    for name, func in [
        ('unfused', gelu_unfused),
        ('eager', gelu_eager),
        ('compiled', gelu_compiled),
    ]:
        # Warmup
        for _ in range(warmup):
            _ = func(input_tensor)
        torch.xpu.synchronize()
        
        # Benchmark
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            output = func(input_tensor)
        end.record()
        torch.xpu.synchronize()
        
        time_ms = start.elapsed_time(end) / iters
        bandwidth = total_gb / (time_ms / 1000.0)
        
        results[name] = (time_ms, bandwidth)
        print(f"{name:12s}: {time_ms:6.3f} ms, {bandwidth:6.1f} GB/s ({bandwidth/530*100:4.1f}% peak)")
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark ATen GELU implementations')
    parser.add_argument('--device', default='xpu', choices=['xpu', 'cuda'])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 80)
    print("ATen GELU Implementation Benchmark")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Shape: [1048576] (1M elements)")
    print(f"Data transfer per iteration: 8.0 MB (4 MB in + 4 MB out)")
    print(f"Peak HBM: 530 GB/s (BMG-G31)")
    print("=" * 80)
    
    results = benchmark_aten_impl(device=args.device, warmup=args.warmup, iters=args.iters)
    
    print("\n" + "=" * 80)
    print("Performance Hierarchy:")
    print("=" * 80)
    sorted_results = sorted(results.items(), key=lambda x: x[1][1])
    for name, (time_ms, bw) in sorted_results:
        print(f"  {name:12s}: {bw:6.1f} GB/s ({bw/530*100:4.1f}% peak)")
    
    print("\nKey Findings:")
    print("  - Unfused baseline: Multiple kernel launches")
    print("  - PyTorch eager/compiled: Optimized GELU implementation")
    print("  - Custom SYCL kernel target: Match or exceed eager performance")
    print("  - Intel XPU stack ceiling: ~12-15% peak for element-wise ops")
