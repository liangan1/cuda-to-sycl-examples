"""
Step 4: Accuracy Test for Rotary Embedding
Following SKILLS.md test pattern
Coverage: 16+ test cases (4 batch sizes × 4 head configs)
"""

import torch
import numpy as np
import math
import sys

# Import custom extension
try:
    import rotary_embedding_xpu
except ImportError:
    print("ERROR: rotary_embedding_xpu extension not found")
    print("Build with: cd torch_ext && python setup.py install")
    sys.exit(1)


def generate_cos_sin_cache_numpy(max_position, head_size, base=10000.0):
    """Generate cos/sin cache using NumPy (reference implementation)"""
    embed_dim = head_size // 2
    cache = np.zeros((max_position, head_size), dtype=np.float32)
    
    for pos in range(max_position):
        for i in range(embed_dim):
            freq = 1.0 / (base ** (2.0 * i / head_size))
            angle = pos * freq
            cache[pos, i] = np.cos(angle)
            cache[pos, i + embed_dim] = np.sin(angle)
    
    return cache


def rotary_embedding_reference(query, key, positions, cos_sin_cache):
    """
    Reference implementation of rotary position embedding
    
    Args:
        query: [num_tokens, num_heads, head_size]
        key: [num_tokens, num_kv_heads, head_size]
        positions: [num_tokens]
        cos_sin_cache: [max_position, head_size]
    
    Returns:
        query_rotated, key_rotated (NumPy arrays)
    """
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key.shape[1]
    embed_dim = head_size // 2
    
    query_out = np.copy(query)
    key_out = np.copy(key)
    
    for token_idx in range(num_tokens):
        pos = positions[token_idx]
        
        # Get cos/sin for this position
        cos_vals = cos_sin_cache[pos, :embed_dim]
        sin_vals = cos_sin_cache[pos, embed_dim:]
        
        # Rotate query
        for head in range(num_heads):
            q0 = query[token_idx, head, :embed_dim]
            q1 = query[token_idx, head, embed_dim:]
            
            query_out[token_idx, head, :embed_dim] = q0 * cos_vals - q1 * sin_vals
            query_out[token_idx, head, embed_dim:] = q0 * sin_vals + q1 * cos_vals
        
        # Rotate key
        for head in range(num_kv_heads):
            k0 = key[token_idx, head, :embed_dim]
            k1 = key[token_idx, head, embed_dim:]
            
            key_out[token_idx, head, :embed_dim] = k0 * cos_vals - k1 * sin_vals
            key_out[token_idx, head, embed_dim:] = k0 * sin_vals + k1 * cos_vals
    
    return query_out, key_out


def test_rotary_embedding(num_tokens, num_heads, num_kv_heads, head_size, max_position=2048, device='xpu'):
    """
    Test rotary embedding for a specific configuration
    
    Args:
        num_tokens: Number of tokens
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_size: Head dimension
        max_position: Maximum position in cache
        device: Device to run on
    
    Returns:
        bool: True if test passes
    """
    # Generate random inputs
    query_cpu = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float32)
    key_cpu = torch.randn(num_tokens, num_kv_heads, head_size, dtype=torch.float32)
    positions_cpu = torch.randint(0, max_position, (num_tokens,), dtype=torch.int64)
    
    # Generate cos/sin cache
    cos_sin_cache_np = generate_cos_sin_cache_numpy(max_position, head_size)
    cos_sin_cache_cpu = torch.from_numpy(cos_sin_cache_np).float()
    
    # Compute reference result
    query_ref, key_ref = rotary_embedding_reference(
        query_cpu.numpy(),
        key_cpu.numpy(),
        positions_cpu.numpy(),
        cos_sin_cache_np
    )
    
    # Run on XPU
    query_xpu = query_cpu.to(device).contiguous()
    key_xpu = key_cpu.to(device).contiguous()
    positions_xpu = positions_cpu.to(device)
    cos_sin_cache_xpu = cos_sin_cache_cpu.to(device)
    
    # Call custom kernel (inplace modification)
    torch.ops.rotary_embedding_xpu.rotary_embedding(
        query_xpu, key_xpu, positions_xpu, cos_sin_cache_xpu
    )
    
    # Compare results
    query_result = query_xpu.cpu().numpy()
    key_result = key_xpu.cpu().numpy()
    
    query_match = np.allclose(query_result, query_ref, rtol=1e-4, atol=1e-5)
    key_match = np.allclose(key_result, key_ref, rtol=1e-4, atol=1e-5)
    
    if query_match and key_match:
        print(f"✅ PASS: num_tokens={num_tokens}, num_heads={num_heads}, "
              f"num_kv_heads={num_kv_heads}, head_size={head_size}")
        return True
    else:
        print(f"❌ FAIL: num_tokens={num_tokens}, num_heads={num_heads}, "
              f"num_kv_heads={num_kv_heads}, head_size={head_size}")
        
        if not query_match:
            query_diff = np.abs(query_result - query_ref)
            print(f"  Query max diff: {query_diff.max():.6e}")
        
        if not key_match:
            key_diff = np.abs(key_result - key_ref)
            print(f"  Key max diff: {key_diff.max():.6e}")
        
        return False


def main():
    print("="*80)
    print("Step 4: Rotary Embedding Accuracy Test")
    print("="*80)
    
    if not torch.xpu.is_available():
        print("ERROR: XPU device not available")
        return False
    
    device = torch.device('xpu')
    print(f"Device: {torch.xpu.get_device_name(0)}\n")
    
    # Test configurations (16+ cases)
    test_configs = [
        # (num_tokens, num_heads, num_kv_heads, head_size)
        # LLaMA-style configurations
        (1, 32, 32, 128),      # Single token, LLaMA-7B
        (4, 32, 32, 128),      # Small batch
        (8, 32, 32, 128),      # Medium batch
        (16, 32, 32, 128),     # Larger batch
        
        # GQA (Grouped Query Attention) configurations
        (1, 32, 8, 128),       # Single token, GQA
        (4, 32, 8, 128),       # Small batch, GQA
        (8, 32, 8, 128),       # Medium batch, GQA
        (16, 32, 8, 128),      # Larger batch, GQA
        
        # Different head sizes
        (4, 32, 32, 64),       # Smaller head
        (4, 32, 32, 256),      # Larger head (if memory allows)
        (4, 40, 40, 128),      # Different head count
        (4, 48, 48, 128),      # Another head count
        
        # Edge cases
        (1, 1, 1, 64),         # Minimal config
        (2, 8, 4, 128),        # Small GQA
        (32, 16, 16, 128),     # Many tokens
        (64, 8, 8, 64),        # Many tokens, small heads
    ]
    
    passed = 0
    failed = 0
    
    for config in test_configs:
        try:
            if test_rotary_embedding(*config, device='xpu'):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ EXCEPTION: {config} - {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"Results: {passed}/{len(test_configs)} tests passed")
    
    if failed == 0:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        return True
    else:
        print(f"❌ {failed} TESTS FAILED")
        print("="*80)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
