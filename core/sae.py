from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import ConfigValue, SAEConfig, SAEOutput

__all__ = [
    "SAEOutput",
    "FlatSAE",
    "MatrixSAE",
    "BilinearMatrixSAE",
    "BilinearEncoderFlatSAE",
    "BilinearGatedSAE",
    "BilinearJumpReLUSAE",
    "infer_sae_type",
    "infer_n_features",
    "infer_rank",
    "infer_matrix_dims",
    "build_sae_from_config",
]

_SUPPORTED_SAE_TYPES = {
    "flat",
    "rank1",
    "bilinear",
    "bilinear_tied",
    "bilinear_flat",
    "bilinear_gated",
    "bilinear_jumprelu",
}
_RANK_STATE_KEYS = ("V", "W", "V_dec", "W_dec", "V_enc", "W_enc")
_MATRIX_STATE_KEYS = ("V", "W")
_BILINEAR_STATE_KEYS = ("V_dec", "W_dec", "V_enc", "W_enc")
ConfigLike = SAEConfig | Mapping[str, ConfigValue]


def _config_str(config: ConfigLike | None, key: str, default: str = "") -> str:
    value = None if config is None else cast(ConfigValue | None, config.get(key))
    return default if value is None else str(value)


def _config_int(config: ConfigLike | None, key: str, default: int) -> int:
    value = None if config is None else cast(ConfigValue | None, config.get(key))
    return default if value is None else int(value)


def _state_tensor(state_dict: Mapping[str, torch.Tensor] | None, key: str) -> torch.Tensor | None:
    if state_dict is None:
        return None
    return state_dict.get(key)


def _upgrade_rank_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    compatible_state = dict(state_dict)
    for key in keys:
        tensor = compatible_state.get(key)
        if tensor is not None and tensor.ndim == 2:
            compatible_state[key] = tensor.unsqueeze(1)
    return compatible_state


def _topk_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(x)
    topk_vals, topk_idx = torch.topk(x, k=min(k, x.shape[-1]), dim=-1)
    topk_vals = F.relu(topk_vals)
    out = torch.zeros_like(x)
    out.scatter_(-1, topk_idx, topk_vals)
    return out


def _batchtopk_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    batch_size = x.shape[0]
    total_k = k * batch_size
    flat = x.reshape(-1)
    flat_relu = F.relu(flat)
    topk_vals, topk_idx = torch.topk(flat_relu, k=min(total_k, flat.numel()))
    out = torch.zeros_like(flat)
    out.scatter_(0, topk_idx, topk_vals)
    return out.reshape_as(x)


class BaseSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_features: int,
        k: int = 32,
        k_aux: int = 256,
        dead_threshold: int = 100,
        use_batchtopk: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.k_aux = k_aux
        self.dead_threshold = dead_threshold
        self.use_batchtopk = use_batchtopk

        self.encoder = nn.Linear(d_in, n_features)
        self.register_buffer("steps_since_active", torch.zeros(n_features, dtype=torch.long))
        self.steps_since_active = cast(torch.Tensor, self.steps_since_active)

    def _encode(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.encoder(x_flat)
        if self.use_batchtopk:
            return _batchtopk_activation(pre, k=self.k), pre
        return _topk_activation(pre, k=self.k), pre

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = self._flatten(x)
        coeffs, _ = self._encode(x_flat)
        return coeffs

    def _decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.d_in)

    def _unflatten(self, x_flat: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
        return x_flat.reshape(ref_shape)

    def _update_dead_tracking(self, coeffs: torch.Tensor) -> torch.Tensor:
        active = (coeffs.abs() > 0).any(dim=0)
        self.steps_since_active[active] = 0
        self.steps_since_active[~active] += 1
        return self.steps_since_active >= self.dead_threshold

    def _compute_aux_loss(
        self,
        pre: torch.Tensor,
        dead_mask: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if dead_mask.sum() == 0 or self.k_aux == 0:
            return torch.tensor(0.0, device=pre.device)

        dead_pre = pre.clone()
        dead_pre[:, ~dead_mask] = -float("inf")
        k_use = min(self.k_aux, int(dead_mask.sum().item()))
        aux_coeffs = _topk_activation(dead_pre, k=k_use)
        aux_recon = self._flatten(self._decode(aux_coeffs))
        return ((aux_recon - residual) ** 2).mean()

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_flat = self._flatten(x)
        coeffs, pre = self._encode(x_flat)
        recon = self._decode(coeffs)
        recon_flat = self._flatten(recon)

        mse = ((recon_flat - x_flat) ** 2).mean()

        dead_mask = self._update_dead_tracking(coeffs)
        residual = (x_flat - recon_flat.detach())
        aux = self._compute_aux_loss(pre, dead_mask, residual)

        return SAEOutput(
            reconstruction=self._unflatten(recon, x.shape),
            coefficients=coeffs,
            loss=mse + aux,
            mse=mse,
            aux_loss=aux,
            n_dead=int(dead_mask.sum().item()),
        )

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def resample_dead_features(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_flat = self._flatten(x_batch)
        dead_mask = self.steps_since_active >= self.dead_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return torch.empty(0, dtype=torch.long)

        coeffs, _ = self._encode(x_flat)
        recon_flat = self._flatten(self._decode(coeffs))
        per_sample_loss = ((recon_flat - x_flat) ** 2).sum(dim=-1)

        loss_sum = per_sample_loss.sum()
        if not torch.isfinite(loss_sum) or loss_sum.item() <= 0:
            probs = torch.full_like(per_sample_loss, 1.0 / max(len(per_sample_loss), 1))
        else:
            probs = per_sample_loss / loss_sum
        indices = torch.multinomial(probs, num_samples=min(n_dead, len(probs)), replacement=True)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:indices.shape[0]]

        alive_mask = ~dead_mask
        avg_alive_norm = self.encoder.weight.data[alive_mask].norm(dim=-1).mean() if alive_mask.any() else 1.0

        self._resample_into(x_batch, indices, dead_indices, avg_alive_norm)
        return dead_indices.detach().cpu()

    def _resample_into(
        self,
        x_batch: torch.Tensor,
        sample_indices: torch.Tensor,
        dead_indices: torch.Tensor,
        avg_alive_norm: float | torch.Tensor,
    ) -> None:
        raise NotImplementedError


class FlatSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        expansion_factor: int = 8,
        k: int = 32,
        k_aux: int = 256,
        dead_threshold: int = 100,
        n_features: int | None = None,
        use_batchtopk: bool = False,
    ):
        if n_features is None:
            n_features = d_in * expansion_factor
        super().__init__(d_in, n_features, k, k_aux, dead_threshold, use_batchtopk=use_batchtopk)
        self.decoder = nn.Linear(n_features, d_in, bias=True)
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def _decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self.decoder(coeffs)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def _resample_into(self, x_batch, sample_indices, dead_indices, avg_alive_norm):
        x_flat = self._flatten(x_batch)
        for i, di in enumerate(dead_indices):
            direction = F.normalize(x_flat[sample_indices[i]], dim=0)
            self.encoder.weight[di].copy_(direction * avg_alive_norm * 0.2)
            self.encoder.bias[di].zero_()
            self.decoder.weight[:, di].copy_(direction)
            self.steps_since_active[di] = 0


class MatrixSAE(BaseSAE):
    def __init__(
        self,
        d_k: int,
        d_v: int,
        expansion_factor: int = 8,
        k: int = 32,
        k_aux: int = 256,
        dead_threshold: int = 100,
        rank: int = 1,
        n_features: int | None = None,
        use_batchtopk: bool = False,
    ):
        d_in = d_k * d_v
        _n_features = n_features if n_features is not None else d_in * expansion_factor
        super().__init__(d_in, _n_features, k, k_aux, dead_threshold, use_batchtopk=use_batchtopk)
        self.d_k = d_k
        self.d_v = d_v
        self.rank = rank

        self.V = nn.Parameter(torch.randn(self.n_features, rank, d_k))
        self.W = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.bias = nn.Parameter(torch.zeros(d_k, d_v))

        with torch.no_grad():
            self.V.copy_(F.normalize(self.V, dim=-1))
            self.W.copy_(F.normalize(self.W, dim=-1))

    def _decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,irk,irv->bkv", coeffs, self.V, self.W) + self.bias

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(x.shape[0], self.d_in)
        return x.reshape(-1, self.d_in)

    def _unflatten(self, x_flat: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
        return x_flat.reshape(ref_shape)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.V.copy_(F.normalize(self.V, dim=-1))
        self.W.copy_(F.normalize(self.W, dim=-1))

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        compatible_state = _upgrade_rank_state_dict(state_dict, _MATRIX_STATE_KEYS)
        return super().load_state_dict(compatible_state, strict=strict, assign=assign)

    def _resample_into(self, x_batch, sample_indices, dead_indices, avg_alive_norm):
        x_flat = self._flatten(x_batch)
        for i, di in enumerate(dead_indices):
            sampled = x_batch[sample_indices[i]]
            U, S, Vt = torch.linalg.svd(sampled, full_matrices=False)
            r = min(self.rank, U.shape[1], Vt.shape[0])
            for j in range(r):
                self.V[di, j].copy_(F.normalize(U[:, j], dim=0))
                self.W[di, j].copy_(F.normalize(Vt[j, :], dim=0))
            for j in range(r, self.rank):
                self.V[di, j].zero_()
                self.W[di, j].zero_()

            flat_dir = F.normalize(x_flat[sample_indices[i]], dim=0)
            self.encoder.weight[di].copy_(flat_dir * avg_alive_norm * 0.2)
            self.encoder.bias[di].zero_()
            self.steps_since_active[di] = 0


class BilinearMatrixSAE(nn.Module):
    def __init__(
        self,
        d_k: int,
        d_v: int,
        expansion_factor: int = 8,
        k: int = 32,
        k_aux: int = 256,
        dead_threshold: int = 100,
        tied: bool = False,
        rank: int = 1,
        n_features: int | None = None,
        use_batchtopk: bool = False,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_k * d_v
        self.n_features = n_features if n_features is not None else d_k * d_v * expansion_factor
        self.k = k
        self.k_aux = k_aux
        self.dead_threshold = dead_threshold
        self.tied = tied
        self.rank = rank
        self.use_batchtopk = use_batchtopk

        if tied:
            self.V_enc = None
            self.W_enc = None
        else:
            self.V_enc = nn.Parameter(torch.randn(self.n_features, rank, d_k))
            self.W_enc = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))

        self.V_dec = nn.Parameter(torch.randn(self.n_features, rank, d_k))
        self.W_dec = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.bias = nn.Parameter(torch.zeros(d_k, d_v))

        self.register_buffer("steps_since_active", torch.zeros(self.n_features, dtype=torch.long))
        self.steps_since_active = cast(torch.Tensor, self.steps_since_active)

        with torch.no_grad():
            self.V_dec.copy_(F.normalize(self.V_dec, dim=-1))
            self.W_dec.copy_(F.normalize(self.W_dec, dim=-1))
            if not tied and self.V_enc is not None and self.W_enc is not None:
                self.V_enc.copy_(F.normalize(self.V_enc, dim=-1))
                self.W_enc.copy_(F.normalize(self.W_enc, dim=-1))

    @property
    def _v_enc(self) -> torch.Tensor:
        if self.tied:
            return self.V_dec
        assert self.V_enc is not None
        return self.V_enc

    @property
    def _w_enc(self) -> torch.Tensor:
        if self.tied:
            return self.W_dec
        assert self.W_enc is not None
        return self.W_enc

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = torch.einsum("irk,bkv,irv->bi", self._v_enc, x, self._w_enc) + self.b_enc
        if self.use_batchtopk:
            return _batchtopk_activation(pre, k=self.k), pre
        return _topk_activation(pre, k=self.k), pre

    def _decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,irk,irv->bkv", coeffs, self.V_dec, self.W_dec) + self.bias

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(x.shape[0], self.d_in)
        return x.reshape(-1, self.d_in)

    def _unflatten(self, x_flat: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
        return x_flat.reshape(ref_shape)

    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.reshape(x.shape[0], self.d_k, self.d_v)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        coeffs, _ = self._encode(self._ensure_3d(x))
        return coeffs

    def _update_dead_tracking(self, coeffs: torch.Tensor) -> torch.Tensor:
        active = (coeffs.abs() > 0).any(dim=0)
        self.steps_since_active[active] = 0
        self.steps_since_active[~active] += 1
        return self.steps_since_active >= self.dead_threshold

    def _compute_aux_loss(
        self,
        pre: torch.Tensor,
        dead_mask: torch.Tensor,
        residual_3d: torch.Tensor,
    ) -> torch.Tensor:
        if dead_mask.sum() == 0 or self.k_aux == 0:
            return torch.tensor(0.0, device=pre.device)
        dead_pre = pre.clone()
        dead_pre[:, ~dead_mask] = -float("inf")
        k_use = min(self.k_aux, int(dead_mask.sum().item()))
        aux_coeffs = _topk_activation(dead_pre, k=k_use)
        aux_recon = self._decode(aux_coeffs)
        return ((self._flatten(aux_recon) - self._flatten(residual_3d)) ** 2).mean()

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_3d = self._ensure_3d(x)
        x_flat = self._flatten(x_3d)
        coeffs, pre = self._encode(x_3d)
        recon = self._decode(coeffs)
        recon_flat = self._flatten(recon)

        mse = ((recon_flat - x_flat) ** 2).mean()

        dead_mask = self._update_dead_tracking(coeffs)
        residual = x_3d - recon.detach()
        aux = self._compute_aux_loss(pre, dead_mask, residual)

        return SAEOutput(
            reconstruction=self._unflatten(recon_flat, x.shape),
            coefficients=coeffs,
            loss=mse + aux,
            mse=mse,
            aux_loss=aux,
            n_dead=int(dead_mask.sum().item()),
        )

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        compatible_state = _upgrade_rank_state_dict(state_dict, _BILINEAR_STATE_KEYS)
        return super().load_state_dict(compatible_state, strict=strict, assign=assign)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.V_dec.copy_(F.normalize(self.V_dec, dim=-1))
        self.W_dec.copy_(F.normalize(self.W_dec, dim=-1))

    @torch.no_grad()
    def resample_dead_features(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_3d = self._ensure_3d(x_batch)
        x_flat = self._flatten(x_3d)
        dead_mask = self.steps_since_active >= self.dead_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return torch.empty(0, dtype=torch.long)

        coeffs, _ = self._encode(x_3d)
        recon_flat = self._flatten(self._decode(coeffs))
        per_sample_loss = ((recon_flat - x_flat) ** 2).sum(dim=-1)

        loss_sum = per_sample_loss.sum()
        if not torch.isfinite(loss_sum) or loss_sum.item() <= 0:
            probs = torch.full_like(per_sample_loss, 1.0 / max(len(per_sample_loss), 1))
        else:
            probs = per_sample_loss / loss_sum
        indices = torch.multinomial(probs, num_samples=min(n_dead, len(probs)), replacement=True)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:indices.shape[0]]

        alive_mask = ~dead_mask
        v_enc = self._v_enc
        avg_alive_norm = v_enc[alive_mask].norm(dim=-1).mean() if alive_mask.any() else 1.0
        untied_v_enc = None if self.tied else self.V_enc
        untied_w_enc = None if self.tied else self.W_enc
        if not self.tied:
            assert untied_v_enc is not None and untied_w_enc is not None

        for i, di in enumerate(dead_indices):
            sampled = x_3d[indices[i]]
            U, S, Vt = torch.linalg.svd(sampled, full_matrices=False)
            r = min(self.rank, U.shape[1], Vt.shape[0])

            for j in range(r):
                v_dir = F.normalize(U[:, j], dim=0)
                w_dir = F.normalize(Vt[j, :], dim=0)
                self.V_dec[di, j].copy_(v_dir)
                self.W_dec[di, j].copy_(w_dir)
                if untied_v_enc is not None and untied_w_enc is not None:
                    untied_v_enc[di, j].copy_(v_dir * avg_alive_norm * 0.2)
                    untied_w_enc[di, j].copy_(w_dir * avg_alive_norm * 0.2)
            for j in range(r, self.rank):
                self.V_dec[di, j].zero_()
                self.W_dec[di, j].zero_()
                if untied_v_enc is not None and untied_w_enc is not None:
                    untied_v_enc[di, j].zero_()
                    untied_w_enc[di, j].zero_()

            self.b_enc[di].zero_()
            self.steps_since_active[di] = 0

        return dead_indices.detach().cpu()


class BilinearEncoderFlatSAE(nn.Module):
    def __init__(
        self,
        d_k: int,
        d_v: int,
        expansion_factor: int = 8,
        k: int = 32,
        k_aux: int = 256,
        dead_threshold: int = 100,
        rank: int = 1,
        n_features: int | None = None,
        use_batchtopk: bool = False,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_k * d_v
        self.n_features = n_features if n_features is not None else d_k * d_v * expansion_factor
        self.k = k
        self.k_aux = k_aux
        self.dead_threshold = dead_threshold
        self.rank = rank
        self.use_batchtopk = use_batchtopk

        self.V_enc = nn.Parameter(torch.randn(self.n_features, rank, d_k))
        self.W_enc = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))

        self.decoder = nn.Linear(self.n_features, self.d_in, bias=True)

        self.register_buffer("steps_since_active", torch.zeros(self.n_features, dtype=torch.long))
        self.steps_since_active = cast(torch.Tensor, self.steps_since_active)

        with torch.no_grad():
            self.V_enc.copy_(F.normalize(self.V_enc, dim=-1))
            self.W_enc.copy_(F.normalize(self.W_enc, dim=-1))
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.reshape(x.shape[0], self.d_k, self.d_v)
        return x

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(x.shape[0], self.d_in)
        return x.reshape(-1, self.d_in)

    def _unflatten(self, x_flat: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
        return x_flat.reshape(ref_shape)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = torch.einsum("irk,bkv,irv->bi", self.V_enc, x, self.W_enc) + self.b_enc
        if self.use_batchtopk:
            return _batchtopk_activation(pre, k=self.k), pre
        return _topk_activation(pre, k=self.k), pre

    def _decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self.decoder(coeffs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        coeffs, _ = self._encode(self._ensure_3d(x))
        return coeffs

    def _update_dead_tracking(self, coeffs: torch.Tensor) -> torch.Tensor:
        active = (coeffs.abs() > 0).any(dim=0)
        self.steps_since_active[active] = 0
        self.steps_since_active[~active] += 1
        return self.steps_since_active >= self.dead_threshold

    def _compute_aux_loss(
        self,
        pre: torch.Tensor,
        dead_mask: torch.Tensor,
        residual_flat: torch.Tensor,
    ) -> torch.Tensor:
        if dead_mask.sum() == 0 or self.k_aux == 0:
            return torch.tensor(0.0, device=pre.device)
        dead_pre = pre.clone()
        dead_pre[:, ~dead_mask] = -float("inf")
        k_use = min(self.k_aux, int(dead_mask.sum().item()))
        aux_coeffs = _topk_activation(dead_pre, k=k_use)
        aux_recon = self._decode(aux_coeffs)
        return ((aux_recon - residual_flat) ** 2).mean()

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_3d = self._ensure_3d(x)
        x_flat = self._flatten(x_3d)
        coeffs, pre = self._encode(x_3d)
        recon_flat = self._decode(coeffs)

        mse = ((recon_flat - x_flat) ** 2).mean()

        dead_mask = self._update_dead_tracking(coeffs)
        residual_flat = x_flat - recon_flat.detach()
        aux = self._compute_aux_loss(pre, dead_mask, residual_flat)

        return SAEOutput(
            reconstruction=self._unflatten(recon_flat, x.shape),
            coefficients=coeffs,
            loss=mse + aux,
            mse=mse,
            aux_loss=aux,
            n_dead=int(dead_mask.sum().item()),
        )

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        compatible_state = _upgrade_rank_state_dict(state_dict, ("V_enc", "W_enc"))
        return super().load_state_dict(compatible_state, strict=strict, assign=assign)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    @torch.no_grad()
    def resample_dead_features(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_3d = self._ensure_3d(x_batch)
        x_flat = self._flatten(x_3d)
        dead_mask = self.steps_since_active >= self.dead_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return torch.empty(0, dtype=torch.long)

        coeffs, _ = self._encode(x_3d)
        recon_flat = self._decode(coeffs)
        per_sample_loss = ((recon_flat - x_flat) ** 2).sum(dim=-1)

        loss_sum = per_sample_loss.sum()
        if not torch.isfinite(loss_sum) or loss_sum.item() <= 0:
            probs = torch.full_like(per_sample_loss, 1.0 / max(len(per_sample_loss), 1))
        else:
            probs = per_sample_loss / loss_sum
        indices = torch.multinomial(probs, num_samples=min(n_dead, len(probs)), replacement=True)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:indices.shape[0]]

        alive_mask = ~dead_mask
        avg_alive_norm = self.V_enc[alive_mask].norm(dim=-1).mean() if alive_mask.any() else 1.0

        for i, di in enumerate(dead_indices):
            sampled = x_3d[indices[i]]
            U, S, Vt = torch.linalg.svd(sampled, full_matrices=False)
            r = min(self.rank, U.shape[1], Vt.shape[0])

            for j in range(r):
                self.V_enc[di, j].copy_(F.normalize(U[:, j], dim=0) * avg_alive_norm * 0.2)
                self.W_enc[di, j].copy_(F.normalize(Vt[j, :], dim=0) * avg_alive_norm * 0.2)
            for j in range(r, self.rank):
                self.V_enc[di, j].zero_()
                self.W_enc[di, j].zero_()

            direction = F.normalize(x_flat[indices[i]], dim=0)
            self.decoder.weight[:, di].copy_(direction)

            self.b_enc[di].zero_()
            self.steps_since_active[di] = 0

        return dead_indices.detach().cpu()


class _GatedIndicatorSTE(torch.autograd.Function):
    """STE for the Gated SAE hard indicator.

    Without this, MSE has zero gradient through the indicator and L1 alone drives
    gate_pre monotonically negative, killing features. STE pretends the indicator
    is identity on the gate_pre path. Rajamanoharan et al. (2024).
    """

    @staticmethod
    def forward(ctx, gate_pre: torch.Tensor, mag_relu: torch.Tensor) -> torch.Tensor:
        mask = (gate_pre > 0).to(mag_relu.dtype)
        ctx.save_for_backward(mask, mag_relu)
        return mask * mag_relu

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        mask, mag_relu = ctx.saved_tensors
        grad_gate_pre = grad_output * mag_relu
        grad_mag_relu = grad_output * mask
        return grad_gate_pre, grad_mag_relu


class _JumpReLUSTE(torch.autograd.Function):
    """STE for JumpReLU activation with rectangle kernel on theta.

    Forward: y = z if z > theta else 0. Gradient flows to z via 1[z > theta]
    and to theta via -theta/h * rect((z - theta)/h). Companion _L0HeavisideSTE
    pushes theta when y=0. Rajamanoharan et al. (2024) Appendix C.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        theta = log_threshold.exp()
        mask = z > theta
        ctx.save_for_backward(z, theta, mask)
        ctx.bandwidth = bandwidth
        return torch.where(mask, z, torch.zeros_like(z))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        z, theta, mask = ctx.saved_tensors
        h = ctx.bandwidth

        grad_z = grad_output * mask.to(grad_output.dtype)

        diff = (z - theta).abs()
        rect = (diff <= 0.5 * h).to(grad_output.dtype)
        grad_theta_z = -(theta / h) * rect * grad_output
        grad_log_threshold = (grad_theta_z.sum(dim=0)) * theta
        return grad_z, grad_log_threshold, None


class _L0HeavisideSTE(torch.autograd.Function):
    """STE for H(z - theta) in the L0 penalty.

    Gradient routes only to theta (rectangle kernel); z gets none so L0 doesn't
    perturb pre-activations.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, log_threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        theta = log_threshold.exp()
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        z, theta = ctx.saved_tensors
        h = ctx.bandwidth
        diff = (z - theta).abs()
        rect = (diff <= 0.5 * h).to(grad_output.dtype)
        grad_theta = -(1.0 / h) * rect * grad_output
        grad_log_threshold = grad_theta.sum(dim=0) * theta
        return None, grad_log_threshold, None


class BilinearGatedSAE(nn.Module):
    """Gated bilinear SAE with rank-r decoder. Rajamanoharan et al. (2024).

    Shared bilinear pre-act W = V_enc outer W_enc drives two paths:
      gate_pre = W x + b_gate
      mag_pre  = exp(r_mag) * W x + b_mag
      coeffs   = 1[gate_pre > 0] * ReLU(mag_pre)

    Training loss splits into mse (in `output.loss`) and the sparsity +
    frozen-decoder auxiliary term (`aux_loss(x)`).
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        expansion_factor: int = 8,
        k: int = 32,
        k_aux: int = 256,
        dead_threshold: int = 100,
        rank: int = 1,
        n_features: int | None = None,
        use_batchtopk: bool = False,
        lambda_sparsity: float = 1e-3,
        lambda_aux: float = 1.0,
        gate_tau: float = 0.1,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_k * d_v
        self.n_features = n_features if n_features is not None else d_k * d_v * expansion_factor
        self.k = k
        self.k_aux = k_aux
        self.dead_threshold = dead_threshold
        self.rank = rank
        # Finite tau replaces the paper's hard indicator: sigmoid never saturates
        # to 0, so MSE gradient keeps reaching every b_gate. tau=0.1 is a constant
        # default; annealing not implemented.
        self.gate_tau = gate_tau
        # Stored for API parity; Gated SAE replaces TopK so the flag is unused in _encode.
        self.use_batchtopk = use_batchtopk
        self.lambda_sparsity = lambda_sparsity
        self.lambda_aux = lambda_aux

        self.V_enc = nn.Parameter(torch.randn(self.n_features, rank, d_k))
        self.W_enc = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.b_gate = nn.Parameter(torch.zeros(self.n_features))
        self.b_mag = nn.Parameter(torch.zeros(self.n_features))
        # Paper's W_mag = exp(r_mag) * W_gate parameterization keeps the magnitude scale strictly positive.
        self.r_mag = nn.Parameter(torch.zeros(self.n_features))

        self.V_dec = nn.Parameter(torch.randn(self.n_features, rank, d_k))
        self.W_dec = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.bias = nn.Parameter(torch.zeros(d_k, d_v))

        self.register_buffer("steps_since_active", torch.zeros(self.n_features, dtype=torch.long))
        self.steps_since_active = cast(torch.Tensor, self.steps_since_active)

        with torch.no_grad():
            self.V_enc.copy_(F.normalize(self.V_enc, dim=-1))
            self.W_enc.copy_(F.normalize(self.W_enc, dim=-1))
            self.V_dec.copy_(F.normalize(self.V_dec, dim=-1))
            self.W_dec.copy_(F.normalize(self.W_dec, dim=-1))

    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.reshape(x.shape[0], self.d_k, self.d_v)
        return x

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(x.shape[0], self.d_in)
        return x.reshape(-1, self.d_in)

    def _unflatten(self, x_flat: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
        return x_flat.reshape(ref_shape)

    def _shared_preact(self, x_3d: torch.Tensor) -> torch.Tensor:
        return torch.einsum("irk,bkv,irv->bi", self.V_enc, x_3d, self.W_enc)

    def _encode(self, x_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self._shared_preact(x_3d)
        gate_pre = shared + self.b_gate
        mag_pre = self.r_mag.exp() * shared + self.b_mag

        # Soft sigmoid replaces the hard indicator: sigmoid never saturates so MSE
        # always reaches b_gate. L1 on gate_soft (bounded in (0,1)) is scale-normalized.
        gate_soft = torch.sigmoid(gate_pre / self.gate_tau)
        coeffs = gate_soft * F.relu(mag_pre)
        return coeffs, gate_soft, mag_pre

    def _decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,irk,irv->bkv", coeffs, self.V_dec, self.W_dec) + self.bias

    def _decode_frozen(self, coeffs: torch.Tensor) -> torch.Tensor:
        # L_aux must only update gate-path weights, so freeze decoder params.
        V = self.V_dec.detach()
        W = self.W_dec.detach()
        b = self.bias.detach()
        return torch.einsum("bi,irk,irv->bkv", coeffs, V, W) + b

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        coeffs, _, _ = self._encode(self._ensure_3d(x))
        return coeffs

    def _update_dead_tracking(self, coeffs: torch.Tensor) -> torch.Tensor:
        active = (coeffs.abs() > 0).any(dim=0)
        self.steps_since_active[active] = 0
        self.steps_since_active[~active] += 1
        return self.steps_since_active >= self.dead_threshold

    def aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_3d = self._ensure_3d(x)
        _, gate_soft, _ = self._encode(x_3d)

        sparsity = gate_soft.sum(dim=-1).mean()
        recon_gate = self._decode_frozen(gate_soft)
        recon_mse = ((self._flatten(recon_gate) - self._flatten(x_3d)) ** 2).mean()
        return self.lambda_sparsity * sparsity + self.lambda_aux * recon_mse

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_3d = self._ensure_3d(x)
        x_flat = self._flatten(x_3d)
        coeffs, gate_soft, _ = self._encode(x_3d)
        recon = self._decode(coeffs)
        recon_flat = self._flatten(recon)

        mse = ((recon_flat - x_flat) ** 2).mean()

        # Aux assembled inline to skip a second forward pass.
        sparsity = gate_soft.sum(dim=-1).mean()
        recon_gate = self._decode_frozen(gate_soft)
        recon_gate_mse = ((self._flatten(recon_gate) - x_flat) ** 2).mean()
        aux = self.lambda_sparsity * sparsity + self.lambda_aux * recon_gate_mse

        dead_mask = self._update_dead_tracking(coeffs)

        return SAEOutput(
            reconstruction=self._unflatten(recon_flat, x.shape),
            coefficients=coeffs,
            loss=mse + aux,
            mse=mse,
            aux_loss=aux,
            n_dead=int(dead_mask.sum().item()),
        )

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        compatible_state = _upgrade_rank_state_dict(state_dict, ("V_enc", "W_enc", "V_dec", "W_dec"))
        return super().load_state_dict(compatible_state, strict=strict, assign=assign)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.V_dec.copy_(F.normalize(self.V_dec, dim=-1))
        self.W_dec.copy_(F.normalize(self.W_dec, dim=-1))

    @torch.no_grad()
    def resample_dead_features(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_3d = self._ensure_3d(x_batch)
        x_flat = self._flatten(x_3d)
        dead_mask = self.steps_since_active >= self.dead_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return torch.empty(0, dtype=torch.long)

        coeffs, _, _ = self._encode(x_3d)
        recon_flat = self._flatten(self._decode(coeffs))
        per_sample_loss = ((recon_flat - x_flat) ** 2).sum(dim=-1)

        loss_sum = per_sample_loss.sum()
        if not torch.isfinite(loss_sum) or loss_sum.item() <= 0:
            probs = torch.full_like(per_sample_loss, 1.0 / max(len(per_sample_loss), 1))
        else:
            probs = per_sample_loss / loss_sum
        indices = torch.multinomial(probs, num_samples=min(n_dead, len(probs)), replacement=True)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][: indices.shape[0]]

        alive_mask = ~dead_mask
        avg_alive_norm = (
            self.V_enc[alive_mask].norm(dim=-1).mean() if alive_mask.any() else torch.tensor(1.0, device=x_batch.device)
        )

        for i, di in enumerate(dead_indices):
            sampled = x_3d[indices[i]]
            U, _S, Vt = torch.linalg.svd(sampled, full_matrices=False)
            r = min(self.rank, U.shape[1], Vt.shape[0])
            for j in range(r):
                v_dir = F.normalize(U[:, j], dim=0)
                w_dir = F.normalize(Vt[j, :], dim=0)
                self.V_dec[di, j].copy_(v_dir)
                self.W_dec[di, j].copy_(w_dir)
                self.V_enc[di, j].copy_(v_dir * avg_alive_norm * 0.2)
                self.W_enc[di, j].copy_(w_dir * avg_alive_norm * 0.2)
            for j in range(r, self.rank):
                self.V_dec[di, j].zero_()
                self.W_dec[di, j].zero_()
                self.V_enc[di, j].zero_()
                self.W_enc[di, j].zero_()

            self.b_gate[di].zero_()
            self.b_mag[di].zero_()
            self.r_mag[di].zero_()
            self.steps_since_active[di] = 0

        return dead_indices.detach().cpu()


class BilinearJumpReLUSAE(nn.Module):
    """JumpReLU SAE on a rank-r bilinear decoder. Rajamanoharan et al. (2024).

      z = V_enc outer W_enc applied to x, plus b_enc
      theta = exp(log_threshold)  (learnable, per-feature)
      a = z * H(z - theta)

    Sparsity term is L0 (not L1): gradient routes to thresholds via the
    Heaviside STE, pushing theta into the z-distribution tail.
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        expansion_factor: int = 8,
        k: int = 32,
        k_aux: int = 256,
        dead_threshold: int = 100,
        rank: int = 1,
        n_features: int | None = None,
        use_batchtopk: bool = False,
        lambda_sparsity: float = 1e-3,
        bandwidth: float = 1e-3,
        init_threshold: float = 1e-3,
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_in = d_k * d_v
        self.n_features = n_features if n_features is not None else d_k * d_v * expansion_factor
        self.k = k
        self.k_aux = k_aux
        self.dead_threshold = dead_threshold
        self.rank = rank
        # Stored for API parity; JumpReLU replaces TopK so the flag is unused.
        self.use_batchtopk = use_batchtopk
        self.lambda_sparsity = lambda_sparsity
        self.bandwidth = bandwidth

        self.V_enc = nn.Parameter(torch.randn(self.n_features, rank, d_k))
        self.W_enc = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))

        # Log-space threshold stays strictly positive under gradient updates.
        log_init = float(torch.log(torch.tensor(init_threshold)))
        self.log_threshold = nn.Parameter(torch.full((self.n_features,), log_init))

        self.V_dec = nn.Parameter(torch.randn(self.n_features, rank, d_k))
        self.W_dec = nn.Parameter(torch.randn(self.n_features, rank, d_v))
        self.bias = nn.Parameter(torch.zeros(d_k, d_v))

        self.register_buffer("steps_since_active", torch.zeros(self.n_features, dtype=torch.long))
        self.steps_since_active = cast(torch.Tensor, self.steps_since_active)

        with torch.no_grad():
            self.V_enc.copy_(F.normalize(self.V_enc, dim=-1))
            self.W_enc.copy_(F.normalize(self.W_enc, dim=-1))
            self.V_dec.copy_(F.normalize(self.V_dec, dim=-1))
            self.W_dec.copy_(F.normalize(self.W_dec, dim=-1))

    def _ensure_3d(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.reshape(x.shape[0], self.d_k, self.d_v)
        return x

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.reshape(x.shape[0], self.d_in)
        return x.reshape(-1, self.d_in)

    def _unflatten(self, x_flat: torch.Tensor, ref_shape: torch.Size) -> torch.Tensor:
        return x_flat.reshape(ref_shape)

    def _preact(self, x_3d: torch.Tensor) -> torch.Tensor:
        return torch.einsum("irk,bkv,irv->bi", self.V_enc, x_3d, self.W_enc) + self.b_enc

    def _encode(self, x_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self._preact(x_3d)
        coeffs = _JumpReLUSTE.apply(z, self.log_threshold, self.bandwidth)
        return coeffs, z

    def _decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,irk,irv->bkv", coeffs, self.V_dec, self.W_dec) + self.bias

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        coeffs, _ = self._encode(self._ensure_3d(x))
        return coeffs

    def _update_dead_tracking(self, coeffs: torch.Tensor) -> torch.Tensor:
        active = (coeffs.abs() > 0).any(dim=0)
        self.steps_since_active[active] = 0
        self.steps_since_active[~active] += 1
        return self.steps_since_active >= self.dead_threshold

    def _l0_penalty(self, z: torch.Tensor) -> torch.Tensor:
        # Heaviside STE routes gradient to log_threshold even when z <= theta.
        indicator = _L0HeavisideSTE.apply(z, self.log_threshold, self.bandwidth)
        return indicator.sum(dim=-1).mean()

    def aux_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_3d = self._ensure_3d(x)
        _, z = self._encode(x_3d)
        return self.lambda_sparsity * self._l0_penalty(z)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_3d = self._ensure_3d(x)
        x_flat = self._flatten(x_3d)
        coeffs, z = self._encode(x_3d)
        recon = self._decode(coeffs)
        recon_flat = self._flatten(recon)

        mse = ((recon_flat - x_flat) ** 2).mean()
        aux = self.lambda_sparsity * self._l0_penalty(z)

        dead_mask = self._update_dead_tracking(coeffs)

        return SAEOutput(
            reconstruction=self._unflatten(recon_flat, x.shape),
            coefficients=coeffs,
            loss=mse + aux,
            mse=mse,
            aux_loss=aux,
            n_dead=int(dead_mask.sum().item()),
        )

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False):
        compatible_state = _upgrade_rank_state_dict(state_dict, ("V_enc", "W_enc", "V_dec", "W_dec"))
        return super().load_state_dict(compatible_state, strict=strict, assign=assign)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.V_dec.copy_(F.normalize(self.V_dec, dim=-1))
        self.W_dec.copy_(F.normalize(self.W_dec, dim=-1))

    @torch.no_grad()
    def resample_dead_features(self, x_batch: torch.Tensor) -> torch.Tensor:
        x_3d = self._ensure_3d(x_batch)
        x_flat = self._flatten(x_3d)
        dead_mask = self.steps_since_active >= self.dead_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return torch.empty(0, dtype=torch.long)

        coeffs, _ = self._encode(x_3d)
        recon_flat = self._flatten(self._decode(coeffs))
        per_sample_loss = ((recon_flat - x_flat) ** 2).sum(dim=-1)

        loss_sum = per_sample_loss.sum()
        if not torch.isfinite(loss_sum) or loss_sum.item() <= 0:
            probs = torch.full_like(per_sample_loss, 1.0 / max(len(per_sample_loss), 1))
        else:
            probs = per_sample_loss / loss_sum
        indices = torch.multinomial(probs, num_samples=min(n_dead, len(probs)), replacement=True)
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][: indices.shape[0]]

        alive_mask = ~dead_mask
        avg_alive_norm = (
            self.V_enc[alive_mask].norm(dim=-1).mean() if alive_mask.any() else torch.tensor(1.0, device=x_batch.device)
        )

        # Reset dead thresholds toward alive mean so they don't stay stuck high.
        init_log_threshold = self.log_threshold[alive_mask].mean() if alive_mask.any() else self.log_threshold.mean()

        for i, di in enumerate(dead_indices):
            sampled = x_3d[indices[i]]
            U, _S, Vt = torch.linalg.svd(sampled, full_matrices=False)
            r = min(self.rank, U.shape[1], Vt.shape[0])
            for j in range(r):
                v_dir = F.normalize(U[:, j], dim=0)
                w_dir = F.normalize(Vt[j, :], dim=0)
                self.V_dec[di, j].copy_(v_dir)
                self.W_dec[di, j].copy_(w_dir)
                self.V_enc[di, j].copy_(v_dir * avg_alive_norm * 0.2)
                self.W_enc[di, j].copy_(w_dir * avg_alive_norm * 0.2)
            for j in range(r, self.rank):
                self.V_dec[di, j].zero_()
                self.W_dec[di, j].zero_()
                self.V_enc[di, j].zero_()
                self.W_enc[di, j].zero_()

            self.b_enc[di].zero_()
            self.log_threshold[di] = init_log_threshold
            self.steps_since_active[di] = 0

        return dead_indices.detach().cpu()


def infer_sae_type(
    config: ConfigLike | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> str:
    cfg = config or {}
    sd = state_dict or {}
    sae_type = _config_str(cfg, "sae_type")
    if sae_type in _SUPPORTED_SAE_TYPES:
        return sae_type
    if "V_enc" in sd and "decoder.weight" in sd and "V_dec" not in sd:
        return "bilinear_flat"
    if "log_threshold" in sd and "V_dec" in sd:
        return "bilinear_jumprelu"
    if "b_gate" in sd and "r_mag" in sd and "V_dec" in sd:
        return "bilinear_gated"
    if "V_dec" in sd or "W_dec" in sd:
        return "bilinear_tied" if "V_enc" not in sd else "bilinear"
    if "V" in sd or "W" in sd:
        return "rank1"
    return "flat"


def infer_n_features(
    config: ConfigLike | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> int | None:
    cfg = config or {}
    nf = cfg.get("n_features")
    if nf is not None:
        return _config_int(cfg, "n_features", 16384)

    sd = state_dict or {}
    for key in ("encoder.weight", *_RANK_STATE_KEYS, "b_enc"):
        tensor = _state_tensor(sd, key)
        if tensor is not None:
            return int(tensor.shape[0])

    decoder_weight = sd.get("decoder.weight")
    if torch.is_tensor(decoder_weight) and decoder_weight.ndim >= 2:
        return int(decoder_weight.shape[1])
    return None


def infer_rank(
    config: ConfigLike | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> int:
    cfg = config or {}
    rank = cfg.get("rank")
    if rank is not None:
        return _config_int(cfg, "rank", 1)

    sd = state_dict or {}
    for key in _RANK_STATE_KEYS:
        tensor = _state_tensor(sd, key)
        if tensor is not None:
            return int(tensor.shape[1]) if tensor.ndim == 3 else 1
    return 1


def infer_matrix_dims(
    config: ConfigLike | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    *,
    default_d_k: int = 128,
    default_d_v: int = 128,
) -> tuple[int, int]:
    cfg = config or {}
    sd = state_dict or {}

    d_k = cast(ConfigValue | None, cfg.get("d_k"))
    d_v = cast(ConfigValue | None, cfg.get("d_v"))
    if d_k is None:
        for key in ("V_dec", "V", "V_enc"):
            tensor = _state_tensor(sd, key)
            if tensor is not None:
                d_k = int(tensor.shape[-1])
                break
    if d_v is None:
        for key in ("W_dec", "W", "W_enc"):
            tensor = _state_tensor(sd, key)
            if tensor is not None:
                d_v = int(tensor.shape[-1])
                break

    return int(d_k or default_d_k), int(d_v or default_d_v)


def build_sae_from_config(
    config: ConfigLike | None = None,
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    default_d_k: int = 128,
    default_d_v: int = 128,
) -> FlatSAE | MatrixSAE | BilinearMatrixSAE | BilinearEncoderFlatSAE | BilinearGatedSAE | BilinearJumpReLUSAE:
    cfg = config or {}
    sd = state_dict or {}
    sae_type = infer_sae_type(cfg, sd)
    n_features = infer_n_features(cfg, sd)
    expansion_factor = _config_int(cfg, "expansion_factor", 4)
    k = _config_int(cfg, "k", 32)
    rank = infer_rank(cfg, sd)
    use_batchtopk = bool(cfg.get("use_batchtopk", False))
    d_k, d_v = infer_matrix_dims(cfg, sd, default_d_k=default_d_k, default_d_v=default_d_v)

    decoder_weight = sd.get("decoder.weight")
    inferred_d_in = int(decoder_weight.shape[0]) if torch.is_tensor(decoder_weight) and decoder_weight.ndim >= 2 else d_k * d_v
    d_in = _config_int(cfg, "d_in", inferred_d_in)

    if sae_type == "flat":
        return FlatSAE(
            d_in=d_in,
            n_features=n_features,
            expansion_factor=expansion_factor,
            k=k,
            use_batchtopk=use_batchtopk,
        )

    if sae_type == "bilinear_flat":
        return BilinearEncoderFlatSAE(
            d_k=d_k,
            d_v=d_v,
            n_features=n_features,
            expansion_factor=expansion_factor,
            k=k,
            rank=rank,
            use_batchtopk=use_batchtopk,
        )

    if sae_type in {"bilinear", "bilinear_tied"}:
        tied = sae_type == "bilinear_tied" or ("V_dec" in sd and "V_enc" not in sd)
        return BilinearMatrixSAE(
            d_k=d_k,
            d_v=d_v,
            n_features=n_features,
            expansion_factor=expansion_factor,
            k=k,
            tied=tied,
            rank=rank,
            use_batchtopk=use_batchtopk,
        )

    if sae_type == "bilinear_gated":
        lambda_sparsity = float(cfg.get("lambda_sparsity", 1e-3) or 1e-3)
        lambda_aux = float(cfg.get("lambda_aux", 1.0) or 1.0)
        return BilinearGatedSAE(
            d_k=d_k,
            d_v=d_v,
            n_features=n_features,
            expansion_factor=expansion_factor,
            k=k,
            rank=rank,
            use_batchtopk=use_batchtopk,
            lambda_sparsity=lambda_sparsity,
            lambda_aux=lambda_aux,
        )

    if sae_type == "bilinear_jumprelu":
        lambda_sparsity = float(cfg.get("lambda_sparsity", 1e-3) or 1e-3)
        bandwidth = float(cfg.get("bandwidth", 1e-3) or 1e-3)
        init_threshold = float(cfg.get("init_threshold", 1e-3) or 1e-3)
        return BilinearJumpReLUSAE(
            d_k=d_k,
            d_v=d_v,
            n_features=n_features,
            expansion_factor=expansion_factor,
            k=k,
            rank=rank,
            use_batchtopk=use_batchtopk,
            lambda_sparsity=lambda_sparsity,
            bandwidth=bandwidth,
            init_threshold=init_threshold,
        )

    return MatrixSAE(
        d_k=d_k,
        d_v=d_v,
        n_features=n_features,
        expansion_factor=expansion_factor,
        k=k,
        rank=rank,
        use_batchtopk=use_batchtopk,
    )
