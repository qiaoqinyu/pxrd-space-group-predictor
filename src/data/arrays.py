"""Array-based data contract for training, evaluation and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.pxrd import augment_csd_profile
from src.runtime.contracts import uses_molecular_embedding, uses_pxrd


GRAPH_KEYS = ("atom_fea", "nbr_fea", "nbr_fea_idx", "atom_mask")


@dataclass(frozen=True)
class ArrayBundle:
    arrays: dict[str, np.ndarray]
    size: int
    has_labels: bool
    has_graph: bool


class ArrayDataset(Dataset):
    def __init__(self, bundle: ArrayBundle, *, augment_csd: bool = False, seed: int = 240427) -> None:
        self.bundle = bundle
        self.augment_csd = augment_csd
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.bundle.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {}
        for key, values in self.bundle.arrays.items():
            value = values[index]
            if key == "sample_id":
                item[key] = str(value)
            elif key in {"labels", "nbr_fea_idx", "row_index"}:
                item[key] = torch.as_tensor(value, dtype=torch.long)
            elif key == "atom_mask":
                item[key] = torch.as_tensor(value, dtype=torch.bool)
            else:
                item[key] = torch.as_tensor(value, dtype=torch.float32)
        if "labels" in item:
            item["label"] = item.pop("labels")
        if self.augment_csd and "pxrd" in item:
            item["pxrd"] = augment_csd_profile(
                item["pxrd"], seed=self.seed, epoch=self.epoch, index=index,
                row_index=int(item.get("row_index", index)))
        return item


def _check_first_dimension(arrays: dict[str, np.ndarray]) -> int:
    if any(value.ndim == 0 or 0 in value.shape for value in arrays.values()):
        raise ValueError("arrays must have a row axis and no empty dimensions")
    sizes = {key: int(value.shape[0]) for key, value in arrays.items()}
    if not sizes:
        raise ValueError("array bundle is empty")
    if len(set(sizes.values())) != 1:
        raise ValueError(f"arrays have inconsistent row counts: {sizes}")
    size = next(iter(sizes.values()))
    if size <= 0:
        raise ValueError("array bundle contains zero rows")
    return size


def load_array_bundle(
    path: str | Path,
    *,
    prediction_modality: str,
    require_labels: bool,
    require_graph: bool,
) -> ArrayBundle:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    required: set[str] = set()
    if uses_pxrd(prediction_modality):
        required.add("pxrd")
    if uses_molecular_embedding(prediction_modality):
        required.add("smiles_embedding")
    if require_labels:
        required.add("labels")
    graph_presence = [key in arrays for key in GRAPH_KEYS]
    if any(graph_presence) and not all(graph_presence):
        missing_graph = [key for key in GRAPH_KEYS if key not in arrays]
        raise ValueError(f"partial graph tensors; missing {missing_graph}")
    if require_graph:
        required.update(GRAPH_KEYS)
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"missing required arrays: {missing}")

    size = _check_first_dimension(arrays)
    for key in ("pxrd", "smiles_embedding", "atom_fea", "nbr_fea"):
        if key in arrays:
            values = arrays[key]
            if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError(f"{key} must contain finite real numeric values")
            if np.any(np.abs(values.astype(np.float64)) > np.finfo(np.float32).max):
                raise ValueError(f"{key} contains values outside the float32 range")
    if "row_index" in arrays:
        rows = arrays["row_index"]
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0):
            raise ValueError("row_index must contain non-negative integer row identities")
    if "sample_id" in arrays and arrays["sample_id"].ndim != 1:
        raise ValueError("sample_id must have shape [N]")
    if "pxrd" in arrays and arrays["pxrd"].ndim != 2:
        raise ValueError("pxrd must have shape [N, L]")
    if "smiles_embedding" in arrays and arrays["smiles_embedding"].ndim != 2:
        raise ValueError("smiles_embedding must have shape [N, D]")
    if "labels" in arrays:
        labels = arrays["labels"]
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be a one-dimensional integer array")
        if np.any(labels < 0):
            raise ValueError("labels must be non-negative class indices")
    if all(graph_presence):
        if arrays["atom_fea"].ndim != 3:
            raise ValueError("atom_fea must have shape [N, A, F]")
        if arrays["nbr_fea"].ndim != 4:
            raise ValueError("nbr_fea must have shape [N, A, K, E]")
        if arrays["nbr_fea"].shape[:2] != arrays["atom_fea"].shape[:2]:
            raise ValueError("atom_fea and nbr_fea atom axes must match")
        if arrays["nbr_fea_idx"].shape != arrays["nbr_fea"].shape[:-1]:
            raise ValueError("nbr_fea_idx shape must match nbr_fea without its feature axis")
        if arrays["atom_mask"].shape != arrays["atom_fea"].shape[:2]:
            raise ValueError("atom_mask shape must match atom_fea atom axes")
        indices = arrays["nbr_fea_idx"]
        mask = arrays["atom_mask"]
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("nbr_fea_idx must contain integer indices")
        if np.any(indices < 0) or np.any(indices >= arrays["atom_fea"].shape[1]):
            raise ValueError("nbr_fea_idx must be in [0, atom_count)")
        if mask.dtype != np.bool_ or not mask.any(axis=1).all():
            raise ValueError("atom_mask must be boolean with at least one real atom per graph")
        neighbors_are_real = mask[np.arange(size)[:, None, None], indices]
        if np.any(mask[:, :, None] & ~neighbors_are_real):
            raise ValueError("real atoms must not reference padded atoms as neighbors")
    return ArrayBundle(
        arrays=arrays,
        size=size,
        has_labels="labels" in arrays,
        has_graph=all(graph_presence),
    )
