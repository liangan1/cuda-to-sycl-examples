#!/usr/bin/env python3
"""
Performance Comparison: Standalone SYCL vs PyTorch Custom Op

Compares the performance of:
1. Standalone SYCL kernel (silu_and_mul_bench)
2. PyTorch custom op (torch.ops.silu_and_mul_xpu.silu_and_mul)

on identical shapes to measure PyTorch overhead.

Usage:
    python compare_standalone_vs_pytorch.py
    python compare_standalone_vs_pytorch.py --shapes "1024,1024 8192,4096"
"""

import argparse
import subprocess
import sys
import time
import torch

try:
    import silu_and_mul_xpu
    HAS_CUSTOM_OP = True
except ImportError:
    print("ERROR: silu_and_mul_xpu not found. Build it first:")
    print("  cd torch_ext && python setup.py install")
    HAS_CUSTOM_OP = False
    sys.exit(1)


def run_standalone_benchmark(num_tokens, d, iters=100, warmup=20):
    """Run standalone SYCL benchmark and parse output"""
    cmd = [
        './build/silu_and_mul_bench',
        '--tokens', str(num_tokens),
        '--d', str(d),
        '--iters', str(iters),
        '--warmup', str(warmup)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse output format:
        #   Min time:       0.257 ms
        #   Bandwidth:      48.98 GB/s
        min_ms = None
        bw_gbps = None
        
        for line in output.split('\n'):
            if 'Min time:' in line:
                min_ms = float(line.split(':')[1].strip().split()[0])
            elif 'Bandwidth:' in line:
                bw_gbps = float(line.split(':')[1].strip().split()[0])
        
        if min_ms is not None and bw_gbps is not None:
            return min_ms, bw_gbps
        
        print(f"WARNING: Could not parse standalone output:\n{output}")
        return None, None
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR running standalone benchmark: {e}")
        print(f"stderr: {e.stderr}")
        return None, None
    except FileNotFoundError:
        print(f"ERROR: {cmd[0]} not found. Build it first:")
        print(f"  cd /home/liangan1/cuda_to_sycl_examples/silu_and_mul")
        print(f"  icpx -fsycl -O3 -fsycl-targets=spir64 silu_and_mul_bench.sycl.cpp -o build/silu_and_mul_bench")
        return None, None


def run_pytorch_benchmark(num_tokens, d, iters=100, warmup=20, device='xpu'):
    """Run PyTorch custom op benchmark"""
    if not torch.xpu.is_available():
        print("ERROR: XPU not available")
        return None, None
    
    # Allocate input
    input_shape = (num_tokens, 2 * d)
    input_xpu = torch.randn(input_shape, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(warmup):
        output = torch.ops.silu_and_mul_xpu.silu_and_mul(input_xpu)
    torch.xpu.synchronize()
    
    # Timed iterations
    times_ms = []
    for _ in range(iters):
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        
        start_event.record()
        output = torch.ops.silu_and_mul_xpu.silu_and_mul(input_xpu)
        end_event.record()
        
        torch.xpu.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times_ms.append(elapsed_ms)
    
    min_ms = min(times_ms)
    
    # Calculate bandwidth (same formula as standalone)
    input_bytes = num_tokens * 2 * d * 4  # float32
    output_bytes = num_tokens * d * 4
    total_bytes = input_bytes + output_bytes
    bw_gbps = (total_bytes / 1e9) / (min_ms / 1000.0)
    
    return min_ms, bw_gbps


def print_comparison_table(results):
    """Print comparison results in a nice table"""
    print("\n" + "="*100)
    print("Performance Comparison: Standalone SYCL vs PyTorch Custom Op")
    print("="*100)
    print(f"{'Shape':<20} | {'Implementation':<20} | {'Min (ms)':<12} | {'BW (GB/s)':<12} | {'Overhead':>10}")
    print("-"*100)
    
    for shape_str, standalone, pytorch in results:
        if standalone and pytorch:
            s_time, s_bw = standalone
            p_time, p_bw = pytorch
            overhead_percent = ((p_time - s_time) / s_time) * 100
            
            print(f"{shape_str:<20} | {'Standalone SYCL':<20} | {s_time:>10.3f}   | {s_bw:>10.2f}   |")
            print(f"{'':<20} | {'PyTorch Custom Op':<20} | {p_time:>10.3f}   | {p_bw:>10.2f}   | {overhead_percent:>9.1f}%")
            print("-"*100)
        else:
            print(f"{shape_str:<20} | ERROR: benchmark failed")
            print("-"*100)
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare standalone SYCL vs PyTorch custom op')
    parser.add_argument('--shapes', type=str, default="1024,1024 8192,4096 16384,4096 32768,4096",
                        help='Space-separated list of "tokens,d" shapes (e.g., "1024,1024 8192,4096")')
    parser.add_argument('--iters', type=int, default=100, help='Number of timed iterations')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup iterations')
    args = parser.parse_args()
    
    # Parse shapes
    shapes = []
    for shape_str in args.shapes.split():
        tokens, d = map(int, shape_str.split(','))
        shapes.append((tokens, d, f"[{tokens}, {d}]"))
    
    print(f"Testing {len(shapes)} shapes with {args.iters} iterations (warmup: {args.warmup})")
    print(f"Shapes: {[s[2] for s in shapes]}")
    
    results = []
    
    for num_tokens, d, shape_str in shapes:
        print(f"\n[{shape_str}] Running standalone SYCL benchmark...")
        s_time, s_bw = run_standalone_benchmark(num_tokens, d, args.iters, args.warmup)
        
        print(f"[{shape_str}] Running PyTorch custom op benchmark...")
        p_time, p_bw = run_pytorch_benchmark(num_tokens, d, args.iters, args.warmup)
        
        results.append((shape_str, (s_time, s_bw) if s_time else None, 
                                   (p_time, p_bw) if p_time else None))
    
    print_comparison_table(results)
    
    # Summary statistics
    valid_results = [(s, p) for _, s, p in results if s and p]
    if valid_results:
        avg_overhead = sum(((p[0] - s[0]) / s[0]) * 100 for s, p in valid_results) / len(valid_results)
        print(f"Average PyTorch overhead: {avg_overhead:.2f}%")
        print(f"(PyTorch time / Standalone time - 1) * 100")
        print()


if __name__ == '__main__':
    main()
