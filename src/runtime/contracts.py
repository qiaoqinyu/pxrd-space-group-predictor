"""Stable prediction-modality identifiers."""

from __future__ import annotations

import torch

PREDICTION_MODALITIES = ("pxrd_smiles", "smiles", "pxrd")


def normalize_prediction_modality(value: str) -> str:
    modality = str(value)
    if modality not in PREDICTION_MODALITIES:
        raise ValueError(
            f"unsupported prediction modality {modality!r}; expected one of {PREDICTION_MODALITIES}"
        )
    return modality


def uses_pxrd(value: str) -> bool:
    return normalize_prediction_modality(value) in {"pxrd_smiles", "pxrd"}


def uses_molecular_embedding(value: str) -> bool:
    return normalize_prediction_modality(value) in {"pxrd_smiles", "smiles"}


def validate_classification_tensors(
    logits: torch.Tensor, labels: torch.Tensor | None = None
) -> None:
    if logits.ndim != 2 or min(logits.shape) <= 0:
        raise ValueError("logits must have non-empty shape [N, C]")
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain finite floating-point values")
    if labels is None:
        return
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("logits and labels have incompatible shapes")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("labels must contain integer class indices")
    if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError("labels must be in [0, num_classes)")
