from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.sae import (  # noqa: E402
    BilinearGatedSAE,
    BilinearJumpReLUSAE,
    build_sae_from_config,
    infer_sae_type,
    MatrixSAE,
)


def _check_output(name: str, sae, x: torch.Tensor, n_features: int) -> None:
    out = sae(x)
    assert out.reconstruction.shape == x.shape, (
        f"{name}: reconstruction shape {out.reconstruction.shape} != input {x.shape}"
    )
    assert out.coefficients.shape == (x.shape[0], n_features), (
        f"{name}: coefficients shape {out.coefficients.shape} != (batch, n_features)"
    )
    assert out.reconstruction.dtype == x.dtype, f"{name}: dtype mismatch"
    assert torch.isfinite(out.loss), f"{name}: loss not finite"
    assert torch.isfinite(out.mse), f"{name}: mse not finite"
    assert torch.isfinite(out.aux_loss), f"{name}: aux_loss not finite"

    aux = sae.aux_loss(x)
    assert aux.ndim == 0, f"{name}: aux_loss(x) should be scalar"
    aux.backward()

    any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in sae.parameters())
    assert any_grad, f"{name}: aux_loss produced no gradients"

    sae.zero_grad(set_to_none=True)

    out = sae(x)
    out.loss.backward()
    any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in sae.parameters())
    assert any_grad, f"{name}: forward loss produced no gradients"


def test_sae_variants_smoke() -> None:
    torch.manual_seed(0)
    d_k, d_v = 8, 6
    n_features = 32
    batch = 4

    x = torch.randn(batch, d_k, d_v, requires_grad=False)

    gated = BilinearGatedSAE(
        d_k=d_k, d_v=d_v, n_features=n_features, k=4, rank=1,
        lambda_sparsity=1e-3, lambda_aux=1.0,
    )
    _check_output("BilinearGatedSAE", gated, x, n_features)
    print("[pass] BilinearGatedSAE forward+aux_loss+backward")

    # Wider bandwidth + larger init_threshold so the rectangle kernel covers
    # some pre-activations at random init; default h=1e-3 has zero gradient
    # almost surely.
    jrelu = BilinearJumpReLUSAE(
        d_k=d_k, d_v=d_v, n_features=n_features, k=4, rank=1,
        lambda_sparsity=1e-3, bandwidth=1.0, init_threshold=0.1,
    )
    _check_output("BilinearJumpReLUSAE", jrelu, x, n_features)
    print("[pass] BilinearJumpReLUSAE forward+aux_loss+backward")

    gated_cfg = {
        "sae_type": "bilinear_gated",
        "d_k": d_k,
        "d_v": d_v,
        "n_features": n_features,
        "k": 4,
        "rank": 1,
    }
    built = build_sae_from_config(gated_cfg)
    assert isinstance(built, BilinearGatedSAE), f"expected BilinearGatedSAE, got {type(built)}"
    _ = built(x)
    print("[pass] build_sae_from_config(sae_type='bilinear_gated')")

    jrelu_cfg = {
        "sae_type": "bilinear_jumprelu",
        "d_k": d_k,
        "d_v": d_v,
        "n_features": n_features,
        "k": 4,
        "rank": 1,
    }
    built = build_sae_from_config(jrelu_cfg)
    assert isinstance(built, BilinearJumpReLUSAE), f"expected BilinearJumpReLUSAE, got {type(built)}"
    _ = built(x)
    print("[pass] build_sae_from_config(sae_type='bilinear_jumprelu')")

    gated_sd = gated.state_dict()
    assert infer_sae_type(state_dict=gated_sd) == "bilinear_gated", (
        f"state_dict inference failed for gated, got {infer_sae_type(state_dict=gated_sd)}"
    )
    print("[pass] infer_sae_type(state_dict) -> bilinear_gated")

    jrelu_sd = jrelu.state_dict()
    assert infer_sae_type(state_dict=jrelu_sd) == "bilinear_jumprelu", (
        f"state_dict inference failed for jumprelu, got {infer_sae_type(state_dict=jrelu_sd)}"
    )
    print("[pass] infer_sae_type(state_dict) -> bilinear_jumprelu")

    gated2 = BilinearGatedSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=4, rank=1)
    gated2.load_state_dict(gated_sd)
    print("[pass] BilinearGatedSAE.load_state_dict round-trip")

    jrelu2 = BilinearJumpReLUSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=4, rank=1)
    jrelu2.load_state_dict(jrelu_sd)
    print("[pass] BilinearJumpReLUSAE.load_state_dict round-trip")

    print("\nALL SMOKE TESTS PASSED")


def test_topk_larger_than_feature_count_is_clamped() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, 2)
    sae = MatrixSAE(d_k=4, d_v=2, n_features=5, k=32, rank=1)

    out = sae(x)

    assert out.coefficients.shape == (3, 5)
    assert out.reconstruction.shape == x.shape
    assert torch.isfinite(out.loss)


if __name__ == "__main__":
    test_sae_variants_smoke()
