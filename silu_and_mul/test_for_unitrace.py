#!/usr/bin/env python3
"""
Simple test for unitrace profiling - run our kernel and measure actual device time
"""

import torch
import silu_and_mul_xpu
import sys

def run_test():
    # Test shape
    N, d = 32768, 11008
    two_d = 2 * d
    
    print(f"Running kernel on shape [{N}, {two_d}]")
    
    # Create input
    input_xpu = torch.randn(N, two_d, device='xpu', dtype=torch.float32)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        output = torch.ops.silu_and_mul_xpu.silu_and_mul(input_xpu)
    torch.xpu.synchronize()
    
    # Run 100 iterations for profiling
    print("Running 100 iterations for profiling...")
    for i in range(100):
        output = torch.ops.silu_and_mul_xpu.silu_and_mul(input_xpu)
        if i % 20 == 0:
            print(f"  Iteration {i}...")
    
    torch.xpu.synchronize()
    print("Done!")
    
    # Calculate expected bandwidth
    input_bytes = N * 2 * d * 4
    output_bytes = N * d * 4
    total_bytes = input_bytes + output_bytes
    print(f"\nExpected total data transfer: {total_bytes/1e9:.3f} GB per iteration")
    print(f"100 iterations: {100 * total_bytes/1e9:.3f} GB total")

if __name__ == "__main__":
    run_test()
