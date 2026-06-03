import torch
import torch.nn.functional as F


def ref_impl(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[1] // 2
    return F.silu(x[:, :d]) * x[:, d:]


def main() -> None:
    import silu_and_mul_verify_xpu  # noqa: F401

    assert torch.xpu.is_available(), 'XPU not available'
    torch.manual_seed(0)

    cases = [
        (1, 256),
        (8, 1024),
        (16, 4096),
        (32, 11008),
    ]

    max_abs = 0.0
    max_rel = 0.0
    for b, d in cases:
        x = torch.randn(b, 2 * d, device='xpu', dtype=torch.float32)
        y_ref = ref_impl(x)
        y = torch.ops.silu_and_mul_verify_xpu.silu_and_mul(x)
        ok = torch.allclose(y, y_ref, rtol=1e-5, atol=1e-6)
        abs_err = (y - y_ref).abs().max().item()
        rel_err = ((y - y_ref).abs() / (y_ref.abs() + 1e-8)).max().item()
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        print(f'case b={b}, d={d}: allclose={ok}, max_abs={abs_err:.3e}, max_rel={rel_err:.3e}')
        if not ok:
            raise RuntimeError('accuracy check failed')

    print('\n[Accuracy]')
    print(f'cases: {len(cases)}')
    print(f'max_abs_error: {max_abs:.6e}')
    print(f'max_rel_error: {max_rel:.6e}')
    print('status: PASS')


if __name__ == '__main__':
    main()
