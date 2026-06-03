"""
Step 5: Performance Benchmark for Rotary Embedding
Following SKILLS.md benchmark pattern
"""

import torch
import time
import numpy as np
import argparse
import sys

# Import custom extension
try:
    import rotary_embedding_xpu
except ImportError:
    print("ERROR: rotary_embedding_xpu extension not found")
    print("Build with: cd torch_ext && python setup.py install")
    sys.exit(1)


def generate_cos_sin_cache(max_position, head_size, base=10000.0, device='cpu'):
    """Generate cos/sin cache on specified device"""
    embed_dim = head_size // 2
    cache = torch.zeros(max_position, head_size, dtype=torch.float32, device=device)
    
    for pos in range(max_position):
        for i in range(embed_dim):
            freq = 1.0 / (base ** (2.0 * i / head_size))
            angle = pos * freq
            cache[pos, i] = np.cos(angle)
            cache[pos, i + embed_dim] = np.sin(angle)
    
    return cache


def benchmark_rotary_embedding(
    num_tokens,
    num_heads,
    num_kv_heads,
    head_size,
    max_position=2048,
    warmup=10,
    iters=100,
    device='xpu'
):
    """
    Benchmark rotary embedding for a specific configuration
    
    Args:
        num_tokens: Number of tokens
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_size: Head dimension
        max_position: Maximum position in cache
        warmup: Warmup iterations
        iters: Benchmark iterations
        device: Device to run on
    
    Returns:
        dict: Benchmark results
    """
    # Allocate inputs
    query = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32, device=device)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float32, device=device)
    positions = torch.randint(0, max_position, (num_tokens,), dtype=torch.int64, device=device)
    cos_sin_cache = generate_cos_sin_cache(max_position, head_size, device=device)
    
    # Warmup
    for _ in range(warmup):
        torch.ops.rotary_embedding_xpu.rotary_embedding(
            query, key, positions, cos_sin_cache
        )
    
    # Synchronize before timing
    if device == 'xpu':
        torch.xpu.synchronize()
    
    # Benchmark
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        torch.ops.rotary_embedding_xpu.rotary_embedding(
            query, key, positions, cos_sin_cache
        )
    end_event.record()
    
    torch.xpu.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    time_per_iter_ms = elapsed_ms / iters
    
    # Calculate memory traffic (bytes)
    # Read: query + key + positions + cos_sin_cache (for accessed positions)
    # Write: query + key (inplace modification)
    query_bytes = num_tokens * num_heads * head_size * 4  # float32
    key_bytes = num_tokens * num_kv_heads * head_size * 4
    positions_bytes = num_tokens * 8  # int64
    cache_bytes = num_tokens * head_size * 4  # Only accessed positions
    
    read_bytes = query_bytes + key_bytes + positions_bytes + cache_bytes
    write_bytes = query_bytes + key_bytes
    total_bytes = read_bytes + write_bytes
    
    bandwidth_gb_s = (total_bytes / 1e9) / (time_per_iter_ms / 1000)
    
    # HBM peak for BMG-G31: 530 GB/s
    hbm_peak_gb_s = 530.0
    percent_peak = (bandwidth_gb_s / hbm_peak_gb_s) * 100
    
    return {
        'time_ms': time_per_iter_ms,
        'bandwidth_gb_s': bandwidth_gb_s,
        'percent_peak': percent_peak,
        'total_mb': total_bytes / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Rotary Embedding')
    parser.add_argument('--device', type=str, default='xpu', help='Device to use')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Benchmark iterations')
    args = parser.parse_args()
    
    print("="*80)
    print("Step 5: Rotary Embedding Performance Benchmark")
    print("="*80)
    
    if args.device == 'xpu':
        if not torch.xpu.is_available():
            print("ERROR: XPU device not available")
            return
        print(f"Device: {torch.xpu.get_device_name(0)}")
    
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}\n")
    
    # Test configurations (representative LLaMA shapes)
    configs = [
        # (name, num_tokens, num_heads, num_kv_heads, head_size)
        ("LLaMA-7B (decode)", 1, 32, 32, 128),
        ("LLaMA-7B (small batch)", 4, 32, 32, 128),
        ("LLaMA-7B (medium batch)", 16, 32, 32, 128),
        ("LLaMA-7B (large batch)", 64, 32, 32, 128),
        ("LLaMA-7B (prefill)", 256, 32, 32, 128),
        
        ("LLaMA-13B (decode)", 1, 40, 40, 128),
        ("LLaMA-13B (batch)", 16, 40, 40, 128),
        
        ("LLaMA-70B GQA (decode)", 1, 64, 8, 128),
        ("LLaMA-70B GQA (batch)", 16, 64, 8, 128),
    ]
    
    print(f"{'Configuration':<30} {'Time (ms)':<12} {'Bandwidth':<15} {'% Peak':<10} {'Data (MB)':<12}")
    print("-"*80)
    
    for name, num_tokens, num_heads, num_kv_heads, head_size in configs:
        try:
            results = benchmark_rotary_embedding(
                num_tokens=num_tokens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                warmup=args.warmup,
                iters=args.iters,
                device=args.device
            )
            
            print(f"{name:<30} {results['time_ms']:>11.4f} "
                  f"{results['bandwidth_gb_s']:>11.2f} GB/s "
                  f"{results['percent_peak']:>8.2f}% "
                  f"{results['total_mb']:>11.2f}")
        
        except Exception as e:
            print(f"{name:<30} ERROR: {e}")
    
    print("="*80)


if __name__ == '__main__':
    main()
