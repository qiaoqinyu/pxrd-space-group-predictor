"""Small runtime helpers shared by command-line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.arrays import ArrayBundle
from src.models.xraypro import ModelConfig


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def model_config_from_bundle(
    path: Path | None,
    bundle: ArrayBundle,
    *,
    prediction_modality: str,
    base_config: ModelConfig | None = None,
) -> ModelConfig:
    values: dict[str, Any] = base_config.to_dict() if base_config is not None else {}
    if path is not None:
        supplied = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("model configuration root must be an object")
        values.update(supplied)
    configured_modality = values.get("prediction_modality")
    if configured_modality is not None and str(configured_modality) != prediction_modality:
        raise ValueError(
            "prediction modality differs between the CLI and model configuration"
        )
    values["prediction_modality"] = prediction_modality
    arrays = bundle.arrays
    inferred = {
        "pxrd_length": int(arrays["pxrd"].shape[1]) if "pxrd" in arrays else None,
        "smiles_embedding_dim": int(arrays["smiles_embedding"].shape[1])
        if "smiles_embedding" in arrays
        else None,
        "atom_dim": int(arrays["atom_fea"].shape[2]) if "atom_fea" in arrays else None,
        "nbr_dim": int(arrays["nbr_fea"].shape[3]) if "nbr_fea" in arrays else None,
    }
    for key, value in inferred.items():
        if value is not None:
            if key in values and int(values[key]) != value:
                raise ValueError(
                    f"model configuration {key}={values[key]} does not match data value {value}"
                )
            values[key] = value
    if "num_classes" not in values:
        if values.get("class_space_groups") is not None:
            values["num_classes"] = len(values["class_space_groups"])
        elif "labels" in arrays:
            values["num_classes"] = int(arrays["labels"].max()) + 1
    config = ModelConfig.from_dict(values)
    validate_bundle_for_model(bundle, config)
    return config


def validate_bundle_for_model(bundle: ArrayBundle, config: ModelConfig) -> None:
    arrays = bundle.arrays
    dimensions = {
        "pxrd": config.pxrd_length,
        "smiles_embedding": config.smiles_embedding_dim,
        "atom_fea": config.atom_dim,
        "nbr_fea": config.nbr_dim,
    }
    for key, expected in dimensions.items():
        if key in arrays and arrays[key].shape[-1] != expected:
            raise ValueError(f"{key} feature dimension does not match model value {expected}")
    if "labels" in arrays and (arrays["labels"] >= config.num_classes).any():
        raise ValueError("labels must be in [0, num_classes)")


def display_space_group(group: int | None) -> str:
    """Others is an aggregate label, not a concrete space group."""
    return "Others" if group is None else str(group)
