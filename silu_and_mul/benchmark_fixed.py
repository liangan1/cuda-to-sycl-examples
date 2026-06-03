#!/usr/bin/env python3
"""
FIXED benchmark - ensure contiguous tensors for fair comparison
"""

import torch
import sys

try:
    import silu_and_mul_xpu
    HAS_CUSTOM = True
except:
    HAS_CUSTOM = False

def measure_bandwidth(fn, input_xpu, desc, iters=100, warmup=20):
    """Measure performance with proper timing"""
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
    mean_ms = sum(times) / len(times)
    
    # Calculate bandwidth
    input_bytes = input_xpu.numel() * 4
    output_bytes = out.numel() * 4
    total_bytes = (input_bytes + output_bytes) / 1e9
    
    bw_gbps = total_bytes / (min_ms / 1000.0)
    pct_peak = (bw_gbps / 530.0) * 100
    
    print(f"{desc:50s} | {min_ms:7.3f} ms | {bw_gbps:7.1f} GB/s | {pct_peak:5.1f}%")
    return bw_gbps

def test_fixed():
    """Test with CONTIGUOUS tensors"""
    print("=" * 110)
    print("FIXED Benchmark: Contiguous Tensors Only")
    print("=" * 110)
    print(f"{'Operation':<50} | {'Time':>10} | {'Bandwidth':>12} | {'% Peak':>5}")
    print("-" * 110)
    
    shapes = [
        (8192, 4096, "Small"),
        (32768, 4096, "Medium"),
        (8192, 11008, "LLaMA-8B d=11008"),
        (32768, 11008, "LLaMA-8B d=11008 large"),
    ]
    
    for num_tokens, d, label in shapes:
        print(f"\n[{label}: shape=[{num_tokens}, {d}]]")
        
        # Create CONTIGUOUS separate tensors (not slices!)
        gate_xpu = torch.randn(num_tokens, d, device='xpu', dtype=torch.float32)
        up_xpu = torch.randn(num_tokens, d, device='xpu', dtype=torch.float32)
        
        # For our custom kernel, need [num_tokens, 2*d] format
        input_2d = torch.cat([gate_xpu, up_xpu], dim=1)
        
        # Verify contiguous
        assert gate_xpu.is_contiguous(), "gate not contiguous!"
        assert up_xpu.is_contiguous(), "up not contiguous!"
        assert input_2d.is_contiguous(), "input_2d not contiguous!"
        
        # Test 1: PyTorch SiLU on contiguous tensor
        silu_fn = lambda x: torch.nn.functional.silu(x)
        measure_bandwidth(silu_fn, gate_xpu, "PyTorch F.silu() [contiguous]")
        
        # Test 2: PyTorch element-wise multiply
        mul_fn = lambda g: g * up_xpu
        measure_bandwidth(mul_fn, gate_xpu, "PyTorch multiply [contiguous]")
        
        # Test 3: PyTorch unfused (but contiguous inputs)
        def pytorch_silu_mul_contiguous(g):
            return torch.nn.functional.silu(g) * up_xpu
        measure_bandwidth(pytorch_silu_mul_contiguous, gate_xpu, 
                         "PyTorch F.silu(g) * u [contiguous, unfused]")
        
        # Test 4: Our custom fused kernel
        if HAS_CUSTOM:
            custom_fn = lambda x: torch.ops.silu_and_mul_xpu.silu_and_mul(x)
            # Bandwidth calculation for custom kernel
            def measure_custom(x):
                out = custom_fn(x)
                return out
            
            # Warmup
            for _ in range(20):
                _ = custom_fn(input_2d)
            torch.xpu.synchronize()
            
            # Timed
            times = []
            for _ in range(100):
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                out = custom_fn(input_2d)
                end.record()
                torch.xpu.synchronize()
                times.append(start.elapsed_time(end))
            
            min_ms = min(times)
            # Input: [N, 2*d], Output: [N, d]
            total_bytes = (num_tokens * 2 * d * 4 + num_tokens * d * 4) / 1e9
            bw_gbps = total_bytes / (min_ms / 1000.0)
            pct = (bw_gbps / 530.0) * 100
            print(f"{'Our custom fused kernel':50s} | {min_ms:7.3f} ms | {bw_gbps:7.1f} GB/s | {pct:5.1f}%")
    
    print("=" * 110)
    print("\nIf PyTorch F.silu still shows low bandwidth, it's genuinely a stack limitation.")

if __name__ == "__main__":
    if not torch.xpu.is_available():
        print("ERROR: XPU not available!")
        sys.exit(1)
    
    test_fixed()
