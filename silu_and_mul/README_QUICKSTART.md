# Quick Start Guide

This is the fastest way to get started with the SiLU-and-Mul PyTorch custom op.

## Prerequisites

```bash
# Check oneAPI is available
source /opt/intel/oneapi/setvars.sh

# Check PyTorch XPU
python -c "import torch; print(f'XPU available: {torch.xpu.is_available()}')"
```

## Build & Test (3 minutes)

```bash
# 1. Build PyTorch extension (1 min)
cd torch_ext
python setup.py install
cd ..

# 2. Quick accuracy test (30 sec)
python test_accuracy.py

# 3. Quick benchmark - small shapes only (1 min)
python bench_xpu_perf.py --config <(echo '{
    "cases": [{
        "arg_type": "default",
        "dtype": ["float32"],
        "batch_size": [1024, 8192, 65536],
        "dim_size": [1024]
    }]
}') --iters 50 --warmup 10
```

## Example Usage

```python
import torch
import silu_and_mul_xpu

# Create input on XPU
batch_size, hidden_dim = 8192, 4096
input_xpu = torch.randn(batch_size, 2 * hidden_dim, device='xpu')

# Run custom op
output = torch.ops.silu_and_mul_xpu.silu_and_mul(input_xpu)

# Verify shape
print(f"Input:  {input_xpu.shape}")   # [8192, 8192]
print(f"Output: {output.shape}")      # [8192, 4096]
```

## Expected Results

- **Accuracy**: All tests pass with `max_diff < 1e-6`
- **Performance**: >89% of HBM peak for large batches (>32K)

## Next Steps

- See [README.md](README.md) for full documentation
- Run full benchmark: `python bench_xpu_perf.py` (takes ~2 min)
