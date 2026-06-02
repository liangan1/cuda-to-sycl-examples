#!/usr/bin/env python3
"""
xpu-perf style benchmark for SiLU-and-Mul fusion kernel

Follows the benchmark format from https://github.com/bytedance/xpu-perf
Tests shapes from bench_config.json with warmup, timing, and bandwidth calculation

Usage:
    python bench_xpu_perf.py [--config bench_config.json] [--iters 100] [--warmup 20]
"""

import torch
import json
import time
import argparse
import sys
import numpy as np
from typing import Dict, List, Tuple

try:
    import silu_and_mul_xpu
    HAS_CUSTOM_OP = True
except ImportError:
    print("Warning: silu_and_mul_xpu extension not found")
    HAS_CUSTOM_OP = False


def measure_bandwidth(func, *args, warmup=20, iters=100, device='xpu'):
    """
    Measure kernel execution time and bandwidth
    
    Args:
        func: callable kernel function
        args: kernel arguments
        warmup: number of warmup iterations
        iters: number of timed iterations
        device: device type
    
    Returns:
        tuple: (min_time_ms, median_time_ms, mean_time_ms, bandwidth_gbps)
    """
    # Warmup
    for _ in range(warmup):
        result = func(*args)
        if device == 'xpu':
            torch.xpu.synchronize()
    
    # Timed iterations
    times_ms = []
    
    if device == 'xpu':
        # Use XPU events for accurate timing
        start_events = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.xpu.Event(enable_timing=True) for _ in range(iters)]
        
        for i in range(iters):
            start_events[i].record()
            result = func(*args)
            end_events[i].record()
        
        torch.xpu.synchronize()
        
        for i in range(iters):
            times_ms.append(start_events[i].elapsed_time(end_events[i]))
    else:
        for _ in range(iters):
            start = time.perf_counter()
            result = func(*args)
            if device == 'xpu':
                torch.xpu.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)
    
    times_ms = np.array(times_ms)
    min_time_ms = np.min(times_ms)
    median_time_ms = np.median(times_ms)
    mean_time_ms = np.mean(times_ms)
    
    # Calculate bandwidth (bytes read + bytes written) / time
    input_tensor = args[0]
    output_shape = result.shape
    bytes_read = input_tensor.numel() * input_tensor.element_size()
    bytes_written = np.prod(output_shape) * input_tensor.element_size()
    total_bytes = bytes_read + bytes_written
    
    bandwidth_gbps = (total_bytes / 1e9) / (min_time_ms / 1000)
    
    return min_time_ms, median_time_ms, mean_time_ms, bandwidth_gbps


def run_benchmark(batch_size: int, dim_size: int, dtype_str: str, 
                  warmup: int, iters: int) -> Dict:
    """
    Run benchmark for a single configuration
    
    Args:
        batch_size: number of tokens
        dim_size: hidden dimension
        dtype_str: data type string ('float32', 'float16', 'bfloat16')
        warmup: warmup iterations
        iters: timed iterations
    
    Returns:
        dict: benchmark results
    """
    # Map dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]
    
    # Create input tensor: [batch_size, 2 * dim_size]
    input_tensor = torch.randn(batch_size, 2 * dim_size, dtype=dtype, device='xpu')
    
    # Run benchmark
    def kernel_func(x):
        return torch.ops.silu_and_mul_xpu.silu_and_mul(x)
    
    min_time, median_time, mean_time, bandwidth = measure_bandwidth(
        kernel_func, input_tensor, warmup=warmup, iters=iters
    )
    
    # Calculate effective memory size
    input_mb = (batch_size * 2 * dim_size * input_tensor.element_size()) / 1e6
    output_mb = (batch_size * dim_size * input_tensor.element_size()) / 1e6
    total_mb = input_mb + output_mb
    
    return {
        'batch_size': batch_size,
        'dim_size': dim_size,
        'dtype': dtype_str,
        'input_shape': f'[{batch_size}, {2*dim_size}]',
        'output_shape': f'[{batch_size}, {dim_size}]',
        'total_mb': total_mb,
        'min_time_ms': min_time,
        'median_time_ms': median_time,
        'mean_time_ms': mean_time,
        'bandwidth_gbps': bandwidth,
    }


def load_config(config_file: str) -> List[Dict]:
    """Load benchmark configurations from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    test_cases = []
    for case in config['cases']:
        for dtype in case['dtype']:
            for batch_size in case['batch_size']:
                for dim_size in case['dim_size']:
                    test_cases.append({
                        'batch_size': batch_size,
                        'dim_size': dim_size,
                        'dtype': dtype,
                    })
    
    return test_cases


def main():
    parser = argparse.ArgumentParser(description='xpu-perf style benchmark for SiLU-and-Mul')
    parser.add_argument('--config', type=str, default='bench_config.json',
                        help='Path to benchmark config JSON file')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of timed iterations')
    parser.add_argument('--warmup', type=int, default=20,
                        help='Number of warmup iterations')
    parser.add_argument('--peak-gbps', type=float, default=530.0,
                        help='Peak HBM bandwidth in GB/s (for % of peak calculation)')
    args = parser.parse_args()
    
    # Check prerequisites
    if not HAS_CUSTOM_OP:
        print("ERROR: silu_and_mul_xpu extension not found")
        print("Build it with: cd torch_ext && python setup.py install")
        return 1
    
    if not torch.xpu.is_available():
        print("ERROR: XPU device not available")
        return 1
    
    # Load test cases
    test_cases = load_config(args.config)
    
    print("=" * 100)
    print("SiLU-and-Mul Fusion Kernel Benchmark (xpu-perf format)")
    print("=" * 100)
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"Config: {args.config}")
    print(f"Warmup: {args.warmup} iterations")
    print(f"Timed:  {args.iters} iterations")
    print(f"Peak BW: {args.peak_gbps:.1f} GB/s")
    print("=" * 100)
    print()
    
    # Table header
    header = (
        f"{'Batch':>8} | {'Dim':>6} | {'DType':>8} | {'Input':>16} | {'Output':>16} | "
        f"{'MB':>8} | {'Min(ms)':>9} | {'BW(GB/s)':>10} | {'%Peak':>7}"
    )
    print(header)
    print("-" * 100)
    
    # Run benchmarks
    results = []
    for case in test_cases:
        try:
            result = run_benchmark(
                case['batch_size'],
                case['dim_size'],
                case['dtype'],
                args.warmup,
                args.iters
            )
            results.append(result)
            
            pct_peak = (result['bandwidth_gbps'] / args.peak_gbps) * 100
            
            print(
                f"{result['batch_size']:8} | "
                f"{result['dim_size']:6} | "
                f"{result['dtype']:8} | "
                f"{result['input_shape']:>16} | "
                f"{result['output_shape']:>16} | "
                f"{result['total_mb']:8.2f} | "
                f"{result['min_time_ms']:9.4f} | "
                f"{result['bandwidth_gbps']:10.2f} | "
                f"{pct_peak:6.1f}%"
            )
            
        except Exception as e:
            print(f"ERROR at batch_size={case['batch_size']}, dim_size={case['dim_size']}: {e}")
    
    print()
    print("=" * 100)
    
    # Summary statistics
    if results:
        bandwidths = [r['bandwidth_gbps'] for r in results]
        pct_peaks = [(r['bandwidth_gbps'] / args.peak_gbps) * 100 for r in results]
        
        print(f"Summary:")
        print(f"  Bandwidth: min={min(bandwidths):.2f} GB/s, "
              f"max={max(bandwidths):.2f} GB/s, "
              f"mean={np.mean(bandwidths):.2f} GB/s")
        print(f"  % of Peak: min={min(pct_peaks):.1f}%, "
              f"max={max(pct_peaks):.1f}%, "
              f"mean={np.mean(pct_peaks):.1f}%")
        
        # Find best performing shape
        best_idx = np.argmax(bandwidths)
        best = results[best_idx]
        print(f"  Best: batch_size={best['batch_size']}, "
              f"BW={best['bandwidth_gbps']:.2f} GB/s ({(best['bandwidth_gbps']/args.peak_gbps)*100:.1f}% peak)")
    
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
