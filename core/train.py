#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.sae import (
    BilinearEncoderFlatSAE,
    BilinearGatedSAE,
    BilinearJumpReLUSAE,
    BilinearMatrixSAE,
    FlatSAE,
    MatrixSAE,
)
from core.split_utils import make_train_val_subsets
from core.types import SAECheckpoint, SAEConfig, TrainResult

TrainableSAE = (
    FlatSAE
    | MatrixSAE
    | BilinearMatrixSAE
    | BilinearEncoderFlatSAE
    | BilinearGatedSAE
    | BilinearJumpReLUSAE
)
_wandb = None


class GDNStateDataset(Dataset):
    def __init__(self, path: str):
        self.data = np.load(path, mmap_mode="r")
    def __len__(self) -> int:
        return self.data.shape[0]
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx].astype(np.float32))


def get_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def _log(log_file, record: dict[str, float | int], step: int) -> None:
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()
    if _wandb is not None:
        _wandb.log(record, step=step)


@torch.no_grad()
def evaluate(model: TrainableSAE, loader: DataLoader, device: torch.device, is_flat: bool) -> dict[str, float]:
    was_training = model.training
    model.eval()
    saved_dead_state = model.steps_since_active.clone() if hasattr(model, "steps_since_active") else None
    totals = {"mse": 0.0, "l0": 0.0}
    feat_ever_active: torch.Tensor | None = None
    n = 0
    for batch in loader:
        batch = batch.to(device)
        bs = batch.shape[0]
        out = model(batch.reshape(bs, -1) if is_flat else batch)
        x_flat, recon_flat = batch.reshape(bs, -1), out.reconstruction.reshape(bs, -1)
        totals["mse"] += F.mse_loss(recon_flat, x_flat).item() * bs
        totals["l0"] += (out.coefficients != 0).float().sum().item()
        batch_active = (out.coefficients.abs() > 0).any(dim=0)
        if feat_ever_active is None:
            feat_ever_active = batch_active
        else:
            feat_ever_active = feat_ever_active | batch_active
        n += bs
    if saved_dead_state is not None:
        model.steps_since_active.copy_(saved_dead_state)
    model.train(was_training)
    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg["dead"] = float((~feat_ever_active).sum().item()) if feat_ever_active is not None else 0.0
    return avg


def _save(
    model: TrainableSAE,
    path: str,
    *,
    config: SAEConfig | None = None,
    epoch: int | None = None,
    step: int | None = None,
    val_mse: float | None = None,
    best_val_mse: float | None = None,
) -> None:
    payload: SAECheckpoint = {"model_state_dict": model.state_dict()}
    if config is not None:
        payload["config"] = config
    if epoch is not None:
        payload["epoch"] = epoch
    if step is not None:
        payload["step"] = step
    if val_mse is not None:
        payload["val_mse"] = val_mse
    if best_val_mse is not None:
        payload["best_val_mse"] = best_val_mse
    torch.save(payload, path)


def _clear_optimizer_state_slice(
    optimizer: torch.optim.Optimizer, param: torch.Tensor,
    indices: torch.Tensor, dim: int = 0,
) -> None:
    state = optimizer.state.get(param)
    if not state:
        return
    idx = indices.to(device=param.device, dtype=torch.long)
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        buf = state.get(key)
        if not torch.is_tensor(buf) or buf.ndim == 0 or buf.shape != param.shape:
            continue
        buf.index_fill_(dim, idx, 0)


def _clear_resampled_optimizer_state(
    model: TrainableSAE, optimizer: torch.optim.Optimizer, indices: torch.Tensor,
) -> None:
    if indices.numel() == 0:
        return
    if isinstance(model, FlatSAE):
        _clear_optimizer_state_slice(optimizer, model.encoder.weight, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.encoder.bias, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.decoder.weight, indices, dim=1)
    elif isinstance(model, MatrixSAE):
        _clear_optimizer_state_slice(optimizer, model.encoder.weight, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.encoder.bias, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.V, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W, indices, dim=0)
    elif isinstance(model, BilinearEncoderFlatSAE):
        _clear_optimizer_state_slice(optimizer, model.decoder.weight, indices, dim=1)
        _clear_optimizer_state_slice(optimizer, model.b_enc, indices, dim=0)
        if model.V_enc is not None:
            _clear_optimizer_state_slice(optimizer, model.V_enc, indices, dim=0)
        if model.W_enc is not None:
            _clear_optimizer_state_slice(optimizer, model.W_enc, indices, dim=0)
    elif isinstance(model, BilinearMatrixSAE):
        _clear_optimizer_state_slice(optimizer, model.V_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.b_enc, indices, dim=0)
        if not model.tied and model.V_enc is not None and model.W_enc is not None:
            _clear_optimizer_state_slice(optimizer, model.V_enc, indices, dim=0)
            _clear_optimizer_state_slice(optimizer, model.W_enc, indices, dim=0)
    elif isinstance(model, BilinearGatedSAE):
        _clear_optimizer_state_slice(optimizer, model.V_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.V_enc, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W_enc, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.b_gate, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.b_mag, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.r_mag, indices, dim=0)
    elif isinstance(model, BilinearJumpReLUSAE):
        _clear_optimizer_state_slice(optimizer, model.V_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W_dec, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.V_enc, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.W_enc, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.b_enc, indices, dim=0)
        _clear_optimizer_state_slice(optimizer, model.log_threshold, indices, dim=0)


def train(
    sae_type: str,
    data_dir: str,
    layer: int,
    head: int,
    n_features: int,
    k: int = 32,
    lr: float = 3e-4,
    lr_min: float = 3e-5,
    batch_size: int = 256,
    epochs: int = 20,
    warmup_steps: int = 50,
    norm_every: int = 100,
    resample_every: int = 250,
    log_every: int = 50,
    output_dir: str = "checkpoints",
    use_wandb: bool = False,
    seed: int = 42,
    rank: int = 1,
    use_batchtopk: bool = False,
    gated_lambda_sparsity: float = 1.0,
    gated_lambda_aux: float = 1.0,
    jumprelu_lambda_sparsity: float = 1e-3,
    jumprelu_bandwidth: float = 1e-3,
    jumprelu_init_threshold: float = 1e-3,
    jumprelu_bandwidth_schedule: str = "constant",
    jumprelu_bandwidth_final: float = 1e-5,
) -> TrainResult:
    global _wandb
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(data_dir, f"layer_{layer}", f"head_{head}.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"State data not found: {data_path}")
    dataset = GDNStateDataset(data_path)
    if len(dataset) == 0:
        raise ValueError(f"Empty dataset at {data_path}")
    d_k, d_v = dataset.data.shape[1], dataset.data.shape[2]
    d_in = d_k * d_v
    expansion = n_features // d_in if n_features >= d_in else 0

    train_set, val_set = make_train_val_subsets(dataset, train_fraction=0.8, seed=seed)
    n_train = len(train_set)

    def make_loader(dataset_split: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    train_loader = make_loader(train_set, True)
    val_loader = make_loader(val_set, False)

    is_flat = sae_type == "flat"
    if is_flat:
        model: TrainableSAE = FlatSAE(d_in=d_in, n_features=n_features, k=k, use_batchtopk=use_batchtopk)
    elif sae_type == "bilinear":
        model = BilinearMatrixSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank, use_batchtopk=use_batchtopk)
    elif sae_type == "bilinear_tied":
        model = BilinearMatrixSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, tied=True, rank=rank, use_batchtopk=use_batchtopk)
    elif sae_type == "bilinear_flat":
        model = BilinearEncoderFlatSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank, use_batchtopk=use_batchtopk)
    elif sae_type == "bilinear_gated":
        model = BilinearGatedSAE(
            d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank,
            use_batchtopk=use_batchtopk,
            lambda_sparsity=gated_lambda_sparsity,
            lambda_aux=gated_lambda_aux,
        )
    elif sae_type == "bilinear_jumprelu":
        model = BilinearJumpReLUSAE(
            d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank,
            use_batchtopk=use_batchtopk,
            lambda_sparsity=jumprelu_lambda_sparsity,
            bandwidth=jumprelu_bandwidth,
            init_threshold=jumprelu_init_threshold,
        )
    else:
        model = MatrixSAE(d_k=d_k, d_v=d_v, n_features=n_features, k=k, rank=rank, use_batchtopk=use_batchtopk)
    uses_aux_loss = sae_type in {"bilinear_gated", "bilinear_jumprelu"}
    anneal_bandwidth = (
        sae_type == "bilinear_jumprelu" and jumprelu_bandwidth_schedule == "cosine_to_zero"
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    total_steps = epochs * len(train_loader)
    warmup_steps = min(warmup_steps, total_steps // 4)
    resample_every = min(resample_every, max(total_steps // 4, 1))
    print(f"{sae_type} | features={n_features} k={k} params={n_params:,} | "
          f"train={n_train} val={len(dataset)-n_train} | {device}")
    print(f"  total_steps={total_steps} warmup={warmup_steps} resample_every={resample_every}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__) or ".",
        check=False,
    )
    _sha = result.stdout.strip()
    config: SAEConfig = {
        "sae_type": sae_type,
        "layer": layer,
        "head": head,
        "n_features": n_features,
        "d_k": d_k,
        "d_v": d_v,
        "d_in": d_k * d_v,
        "expansion_factor": expansion,
        "k": k,
        "rank": rank,
        "use_batchtopk": use_batchtopk,
        "seed": seed,
        "lr": lr,
        "lr_min": lr_min,
        "batch_size": batch_size,
        "epochs": epochs,
        "total_steps": total_steps,
        "n_params": n_params,
        "device": str(device),
        "code_sha": _sha,
    }
    if sae_type == "bilinear_gated":
        config["lambda_sparsity"] = gated_lambda_sparsity  # type: ignore[typeddict-unknown-key]
        config["lambda_aux"] = gated_lambda_aux  # type: ignore[typeddict-unknown-key]
    if sae_type == "bilinear_jumprelu":
        config["lambda_sparsity"] = jumprelu_lambda_sparsity  # type: ignore[typeddict-unknown-key]
        config["bandwidth"] = jumprelu_bandwidth  # type: ignore[typeddict-unknown-key]
        config["init_threshold"] = jumprelu_init_threshold  # type: ignore[typeddict-unknown-key]
        config["bandwidth_schedule"] = jumprelu_bandwidth_schedule  # type: ignore[typeddict-unknown-key]
        config["bandwidth_final"] = jumprelu_bandwidth_final  # type: ignore[typeddict-unknown-key]
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log_file = open(os.path.join(output_dir, "train_log.jsonl"), "a")

    if use_wandb:
        import wandb
        _wandb = wandb
        _wandb.init(project="matrix-sae", config=config, name=f"{sae_type}_L{layer}_H{head}")

    best_val_mse, last_val_mse, step = float("inf"), float("inf"), 0
    start_epoch = 0
    t_start = time.time()

    ckpt_path = os.path.join(output_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt: SAECheckpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        start_epoch = ckpt["epoch"] + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        print(f"Resuming from step {step} / epoch {start_epoch} (best_val_mse={best_val_mse:.4e})")

    for epoch in range(start_epoch, epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_t in pbar:
            batch_t = batch_t.to(device, non_blocking=True)
            bs = batch_t.shape[0]
            if anneal_bandwidth:
                progress = step / max(total_steps - 1, 1)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                new_bw = jumprelu_bandwidth_final + (jumprelu_bandwidth - jumprelu_bandwidth_final) * cosine
                model.bandwidth = float(new_bw)
            out = model(batch_t.reshape(bs, -1) if is_flat else batch_t)
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()

            lr_now = get_lr(step, warmup_steps, total_steps, lr, lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            optimizer.step()

            if step % norm_every == 0:
                model.normalize_decoder()
            if step > 0 and step % resample_every == 0:
                resampled = model.resample_dead_features(batch_t)
                if resampled.numel() > 0:
                    _clear_resampled_optimizer_state(model, optimizer, resampled)
                    print(f"  Resampled {int(resampled.numel())} dead features at step {step}")

            if step % log_every == 0:
                with torch.no_grad():
                    x_flat = batch_t.reshape(bs, -1)
                    r_flat = out.reconstruction.reshape(bs, -1)
                    mse = F.mse_loss(r_flat, x_flat).item()
                    nmse = mse / (x_flat ** 2).mean().clamp(min=1e-12).item()
                    ev = 1.0 - mse / x_flat.var().clamp(min=1e-12).item()
                    l0 = (out.coefficients != 0).float().sum(dim=-1).mean().item()
                m = dict(step=step, epoch=epoch, lr=lr_now, loss=out.loss.item(),
                         mse=mse, nmse=nmse, explained_var=ev, l0=l0)
                if uses_aux_loss:
                    m["aux_loss"] = float(out.aux_loss.item())
                    m["total_loss"] = float(out.loss.item())
                if anneal_bandwidth:
                    m["bandwidth"] = float(model.bandwidth)
                pbar.set_postfix(loss=f"{m['loss']:.4e}", nmse=f"{nmse:.4f}",
                                 ev=f"{ev:.4f}", l0=f"{l0:.0f}", lr=f"{lr_now:.2e}")
                _log(log_file, m, step)
            step += 1

        val = evaluate(model, val_loader, device, is_flat)
        last_val_mse = val["mse"]
        vr = {f"val_{vk}": vv for vk, vv in val.items()}
        vr.update(epoch=epoch, step=step)
        _log(log_file, vr, step)
        print(f"  Val MSE={val['mse']:.4e}  L0={val['l0']:.1f}  Dead={val['dead']:.0f}")

        if val["mse"] < best_val_mse:
            best_val_mse = val["mse"]
            _save(
                model,
                os.path.join(output_dir, "best.pt"),
                epoch=epoch,
                step=step,
                val_mse=best_val_mse,
                config=config,
            )
            print(f"  Saved best (val_mse={best_val_mse:.4e})")

        ckpt: SAECheckpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step, "epoch": epoch,
            "best_val_mse": best_val_mse,
            "config": config,
        }
        torch.save(ckpt, ckpt_path)

    final_val_mse = last_val_mse if (epochs > 0 and start_epoch < epochs) else best_val_mse
    _save(
        model,
        os.path.join(output_dir, "final.pt"),
        epoch=epochs - 1,
        step=step,
        val_mse=final_val_mse,
        config=config,
    )
    log_file.close()
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    if _wandb is not None:
        _wandb.finish()
        _wandb = None
    print(f"Saved final model to {output_dir}/final.pt")

    model_dead = int((model.steps_since_active >= model.dead_threshold).sum().item()) if hasattr(model, "steps_since_active") else 0
    result: TrainResult = {
        "sae_type": sae_type, "layer": layer, "head": head,
        "expansion_factor": expansion, "k": k, "rank": rank, "seed": seed,
        "code_sha": _sha, "n_features": n_features,
        "n_samples": len(dataset), "best_mse": best_val_mse,
        "final_mse": final_val_mse, "final_n_dead": model_dead,
        "total_time_s": round(time.time() - t_start, 1),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sae_type",
        choices=[
            "flat",
            "rank1",
            "bilinear",
            "bilinear_tied",
            "bilinear_flat",
            "bilinear_gated",
            "bilinear_jumprelu",
        ],
        required=True,
    )
    p.add_argument("--data_dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", required=True, help="int or 'all'")
    p.add_argument("--n_features", default="16384", help="int or expansion like '2x'")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_min", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--norm_every", type=int, default=100)
    p.add_argument("--resample_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--seed", type=int, default=42, help="Random seed for weight init + train/val split")
    p.add_argument("--batchtopk", action="store_true", help="Use BatchTopK sparsity instead of per-sample TopK")
    p.add_argument("--gated_lambda_sparsity", type=float, default=1.0,
                   help="Gated SAE sparsity coefficient on ReLU(gate_pre) L1 (sae_type=bilinear_gated)")
    p.add_argument("--gated_lambda_aux", type=float, default=1.0,
                   help="Gated SAE auxiliary frozen-decoder reconstruction weight (sae_type=bilinear_gated)")
    p.add_argument("--jumprelu_lambda_sparsity", type=float, default=1e-3,
                   help="JumpReLU L0 penalty coefficient (sae_type=bilinear_jumprelu)")
    p.add_argument("--jumprelu_bandwidth", type=float, default=1e-3,
                   help="JumpReLU initial STE bandwidth h (sae_type=bilinear_jumprelu)")
    p.add_argument("--jumprelu_init_threshold", type=float, default=1e-3,
                   help="JumpReLU initial per-feature threshold theta (sae_type=bilinear_jumprelu)")
    p.add_argument("--jumprelu_bandwidth_schedule",
                   choices=["constant", "cosine_to_zero"], default="constant",
                   help="Anneal JumpReLU bandwidth: constant or cosine to jumprelu_bandwidth_final")
    p.add_argument("--jumprelu_bandwidth_final", type=float, default=1e-5,
                   help="Final bandwidth value when schedule=cosine_to_zero")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    nf = args.n_features
    if nf.endswith("x"):
        _dd = Path(args.data_dir) / f"layer_{args.layer}"
        _candidates = sorted(_dd.glob("head_*.npy"))
        if _candidates:
            _arr = np.load(str(_candidates[0]), mmap_mode="r")
            if _arr.ndim == 3:
                input_dim = int(_arr.shape[1]) * int(_arr.shape[2])
            elif _arr.ndim == 2:
                input_dim = int(_arr.shape[1])
            else:
                input_dim = 128 * 128
            del _arr
        else:
            input_dim = 128 * 128
        n_features = int(input_dim * float(nf[:-1]))
    else:
        n_features = int(nf)

    if args.head == "all":
        ld = os.path.join(args.data_dir, f"layer_{args.layer}")
        heads = sorted(int(f.stem.split("_")[1]) for f in Path(ld).glob("head_*.npy"))
        print(f"Training {len(heads)} heads: {heads}")
    else:
        heads = [int(args.head)]

    for head in heads:
        print(f"\n{'='*60}\nlayer={args.layer} head={head} type={args.sae_type}\n{'='*60}")
        train(
            sae_type=args.sae_type, data_dir=args.data_dir, layer=args.layer,
            head=head, n_features=n_features, k=args.k, lr=args.lr,
            lr_min=args.lr_min, batch_size=args.batch_size, epochs=args.epochs,
            warmup_steps=args.warmup_steps, norm_every=args.norm_every,
            resample_every=args.resample_every, log_every=args.log_every,
            output_dir=os.path.join(args.output_dir, f"{args.sae_type}_L{args.layer}_H{head}"),
            use_wandb=args.wandb, rank=args.rank,
            use_batchtopk=args.batchtopk,
            seed=args.seed,
            gated_lambda_sparsity=args.gated_lambda_sparsity,
            gated_lambda_aux=args.gated_lambda_aux,
            jumprelu_lambda_sparsity=args.jumprelu_lambda_sparsity,
            jumprelu_bandwidth=args.jumprelu_bandwidth,
            jumprelu_init_threshold=args.jumprelu_init_threshold,
            jumprelu_bandwidth_schedule=args.jumprelu_bandwidth_schedule,
            jumprelu_bandwidth_final=args.jumprelu_bandwidth_final,
        )


if __name__ == "__main__":
    main()
