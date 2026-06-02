#!/usr/bin/env python3
"""
Performance comparison: Baseline vs Optimized SiLU-and-Mul kernel

Compares:
1. Baseline (semantic equivalence): silu_and_mul_xpu
2. Optimized (Phase 1-3): silu_and_mul_xpu_optimized

Target: 75% HBM peak (~400 GB/s on BMG-G31)
"""

import argparse
import sys
import torch

# Try to import both versions
try:
    import silu_and_mul_xpu
    HAS_BASELINE = True
except ImportError:
    print("WARNING: Baseline version not found. Build it with:")
    print("  cd torch_ext && python setup.py install")
    HAS_BASELINE = False

try:
    import silu_and_mul_xpu_optimized
    HAS_OPTIMIZED = True
except ImportError:
    print("WARNING: Optimized version not found. Build it with:")
    print("  cd torch_ext && python setup_optimized.py install")
    HAS_OPTIMIZED = False

if not (HAS_BASELINE or HAS_OPTIMIZED):
    print("ERROR: No versions available for testing")
    sys.exit(1)


def measure_performance(kernel_fn, input_xpu, iters=100, warmup=20):
    """Measure kernel performance"""
    # Warmup
    for _ in range(warmup):
        output = kernel_fn(input_xpu)
    torch.xpu.synchronize()
    
    # Timed iterations
    times_ms = []
    for _ in range(iters):
        start_event = torch.xpu.Event(enable_timing=True)
        end_event = torch.xpu.Event(enable_timing=True)
        
        start_event.record()
        output = kernel_fn(input_xpu)
        end_event.record()
        
        torch.xpu.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        times_ms.append(elapsed_ms)
    
    min_ms = min(times_ms)
    median_ms = sorted(times_ms)[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)
    
    return min_ms, median_ms, mean_ms, output


def calculate_bandwidth(num_tokens, d, time_ms):
    """Calculate effective bandwidth"""
    input_bytes = num_tokens * 2 * d * 4  # float32
    output_bytes = num_tokens * d * 4
    total_bytes = input_bytes + output_bytes
    bandwidth_gbps = (total_bytes / 1e9) / (time_ms / 1000.0)
    return bandwidth_gbps


def test_accuracy(baseline_out, optimized_out, rtol=1e-5, atol=1e-6):
    """Compare accuracy between baseline and optimized"""
    if baseline_out is None or optimized_out is None:
        return True, 0.0
    
    max_diff = torch.max(torch.abs(baseline_out - optimized_out)).item()
    matches = torch.allclose(baseline_out, optimized_out, rtol=rtol, atol=atol)
    return matches, max_diff


def main():
    parser = argparse.ArgumentParser(description='Compare baseline vs optimized kernel')
    parser.add_argument('--shapes', type=str, 
                        default="1024,1024 8192,4096 16384,4096 32768,4096 65536,4096 1024,11008",
                        help='Space-separated "tokens,d" shapes')
    parser.add_argument('--iters', type=int, default=100, help='Timed iterations')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    parser.add_argument('--peak-bw', type=float, default=530.0, help='Peak HBM BW in GB/s')
    args = parser.parse_args()
    
    if not torch.xpu.is_available():
        print("ERROR: XPU not available")
        sys.exit(1)
    
    # Parse shapes
    shapes = []
    for shape_str in args.shapes.split():
        tokens, d = map(int, shape_str.split(','))
        shapes.append((tokens, d))
    
    print("="*100)
    print("Performance Comparison: Baseline vs Optimized SiLU-and-Mul Kernel")
    print("="*100)
    print(f"Target: 75% of {args.peak_bw} GB/s = {args.peak_bw * 0.75:.1f} GB/s")
    print(f"Testing {len(shapes)} shapes with {args.iters} iterations (warmup: {args.warmup})")
    print()
    
    results = []
    
    for num_tokens, d in shapes:
        shape_str = f"[{num_tokens}, {d}]"
        input_shape = (num_tokens, 2 * d)
        input_xpu = torch.randn(input_shape, device='xpu', dtype=torch.float32)
        
        baseline_out = None
        optimized_out = None
        baseline_time = None
        optimized_time = None
        
        # Test baseline
        if HAS_BASELINE:
            print(f"[{shape_str}] Testing baseline...")
            min_ms, median_ms, mean_ms, baseline_out = measure_performance(
                torch.ops.silu_and_mul_xpu.silu_and_mul, input_xpu, args.iters, args.warmup
            )
            baseline_time = min_ms
            baseline_bw = calculate_bandwidth(num_tokens, d, min_ms)
            baseline_pct = (baseline_bw / args.peak_bw) * 100
        
        # Test optimized
        if HAS_OPTIMIZED:
            print(f"[{shape_str}] Testing optimized...")
            min_ms, median_ms, mean_ms, optimized_out = measure_performance(
                torch.ops.silu_and_mul_xpu_opt.silu_and_mul, input_xpu, args.iters, args.warmup
            )
            optimized_time = min_ms
            optimized_bw = calculate_bandwidth(num_tokens, d, min_ms)
            optimized_pct = (optimized_bw / args.peak_bw) * 100
        
        # Accuracy check
        if HAS_BASELINE and HAS_OPTIMIZED:
            matches, max_diff = test_accuracy(baseline_out, optimized_out)
            accuracy_str = f"✓ max_diff={max_diff:.2e}" if matches else f"✗ FAILED (diff={max_diff:.2e})"
        else:
            accuracy_str = "N/A"
        
        results.append({
            'shape': shape_str,
            'baseline_time': baseline_time,
            'baseline_bw': baseline_bw if baseline_time else None,
            'baseline_pct': baseline_pct if baseline_time else None,
            'optimized_time': optimized_time,
            'optimized_bw': optimized_bw if optimized_time else None,
            'optimized_pct': optimized_pct if optimized_time else None,
            'speedup': (baseline_time / optimized_time) if (baseline_time and optimized_time) else None,
            'accuracy': accuracy_str
        })
    
    # Print results table
    print("\n" + "="*100)
    print(f"{'Shape':<20} | {'Baseline':<30} | {'Optimized':<30} | {'Speedup':>8} | {'Accuracy':<20}")
    print("-"*100)
    
    for r in results:
        baseline_str = f"{r['baseline_time']:.3f}ms {r['baseline_bw']:.1f}GB/s ({r['baseline_pct']:.1f}%)" if r['baseline_time'] else "N/A"
        optimized_str = f"{r['optimized_time']:.3f}ms {r['optimized_bw']:.1f}GB/s ({r['optimized_pct']:.1f}%)" if r['optimized_time'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        
        print(f"{r['shape']:<20} | {baseline_str:<30} | {optimized_str:<30} | {speedup_str:>8} | {r['accuracy']:<20}")
    
    print("="*100)
    
    # Summary
    if HAS_BASELINE and HAS_OPTIMIZED:
        avg_speedup = sum(r['speedup'] for r in results if r['speedup']) / len([r for r in results if r['speedup']])
        max_optimized_bw = max(r['optimized_bw'] for r in results if r['optimized_bw'])
        max_optimized_pct = max(r['optimized_pct'] for r in results if r['optimized_pct'])
        
        print(f"\nSummary:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Best optimized BW: {max_optimized_bw:.1f} GB/s ({max_optimized_pct:.1f}% of peak)")
        print(f"  Target (75% peak): {args.peak_bw * 0.75:.1f} GB/s")
        
        if max_optimized_pct >= 75.0:
            print(f"  ✅ TARGET ACHIEVED! ({max_optimized_pct:.1f}% >= 75%)")
        else:
            gap = 75.0 - max_optimized_pct
            print(f"  ⚠️  Gap to target: {gap:.1f} percentage points")
            print(f"  Need: {(args.peak_bw * 0.75):.1f} GB/s, Got: {max_optimized_bw:.1f} GB/s")
    
    print()


if __name__ == '__main__':
    main()
