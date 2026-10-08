"""Checkpoint serialization with an explicit model configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.xraypro import ModelConfig, XRayProModel

CHECKPOINT_SCHEMA_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: XRayProModel,
    epoch: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[XRayProModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    config = ModelConfig.from_dict(dict(payload.get("model_config") or {}))
    model = XRayProModel(config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload
