from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import TypedDict

import torch

ConfigValue = str | int | float | bool | None


@dataclass
class SAEOutput:
    reconstruction: torch.Tensor
    coefficients: torch.Tensor
    loss: torch.Tensor
    mse: torch.Tensor
    aux_loss: torch.Tensor
    n_dead: int


class SAEConfig(TypedDict, total=False):
    sae_type: str
    layer: int
    head: int
    n_features: int
    d_k: int
    d_v: int
    d_in: int
    expansion_factor: int
    k: int
    rank: int
    use_batchtopk: bool
    seed: int
    lr: float
    lr_min: float
    batch_size: int
    epochs: int
    total_steps: int
    n_params: int
    device: str
    code_sha: str


class SAECheckpoint(TypedDict, total=False):
    model_state_dict: Mapping[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    config: SAEConfig
    step: int
    epoch: int
    val_mse: float
    best_val_mse: float


class TrainResult(TypedDict):
    sae_type: str
    layer: int
    head: int
    expansion_factor: int
    k: int
    rank: int
    seed: int
    code_sha: str
    n_features: int
    n_samples: int
    best_mse: float
    final_mse: float
    final_n_dead: int
    total_time_s: float


class PromptRecord(TypedDict):
    source: str
    tokens: list[int]


class PromptPool(TypedDict):
    prompts: list[PromptRecord]


class FlagshipFeatureRecord(TypedDict):
    feature_id: int
    taxonomy: str
    layer: int
    head: int


class FlagshipFeatureFile(TypedDict):
    features: list[FlagshipFeatureRecord]


class H7FeatureResult(TypedDict):
    feature_id: int
    taxonomy: str
    layer: int
    head: int
    n_train: int
    n_holdout: int
    effect_train_median: float
    effect_train_ci_low: float
    effect_train_ci_high: float
    effect_holdout_median: float
    effect_holdout_ci_low: float
    effect_holdout_ci_high: float
    transfer_ratio: float
    transfer_ratio_ci_low: float
    transfer_ratio_ci_high: float
    wilcoxon_holdout_vs_zero_p: float
    wilcoxon_holdout_vs_train_p: float
    passes_transfer_threshold: bool
    per_prompt_effect_train: list[float]
    per_prompt_effect_holdout: list[float]


class H7FamilyResult(TypedDict):
    n_features: int
    transfer_ratio_min: float
    family_alpha: float
    bonferroni_alpha: float
    n_pass_threshold: int
    wilcoxon_paired_holdout_vs_train_p: float
    bonferroni_n_significant: int


class H7ResultsPayload(TypedDict):
    schema_version: int
    experiment: str
    exp_name: str
    seed: int
    model_name: str
    layer: int
    head: int
    ref_t: int
    ref_s: int
    epsilon_c: float
    top_k: int
    baseline_conf_threshold: float
    n_holdout_prompts_requested: int
    source_filter: str
    bootstrap_resamples: int
    transfer_ratio_min: float
    family_alpha: float
    per_feature: list[H7FeatureResult]
    family: H7FamilyResult
