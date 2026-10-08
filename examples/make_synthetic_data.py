#!/usr/bin/env python3
"""Create synthetic bundles for software smoke tests only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_bundle(path: Path, *, size: int, seed: int, labels: bool, graph: bool) -> None:
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {
        "pxrd": rng.random((size, 60), dtype=np.float32),
        "smiles_embedding": rng.normal(size=(size, 8)).astype(np.float32),
        "sample_id": np.asarray([f"synthetic-{seed}-{index}" for index in range(size)]),
    }
    if labels:
        arrays["labels"] = np.arange(size, dtype=np.int64) % 3
    if graph:
        arrays.update(
            {
                "atom_fea": rng.normal(size=(size, 3, 4)).astype(np.float32),
                "nbr_fea": rng.normal(size=(size, 3, 2, 5)).astype(np.float32),
                "nbr_fea_idx": rng.integers(0, 3, size=(size, 3, 2), dtype=np.int64),
                "atom_mask": np.ones((size, 3), dtype=bool),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    make_bundle(args.output_dir / "train.npz", size=6, seed=1, labels=True, graph=True)
    make_bundle(args.output_dir / "val.npz", size=3, seed=2, labels=True, graph=True)
    make_bundle(args.output_dir / "test.npz", size=3, seed=3, labels=True, graph=False)
    make_bundle(args.output_dir / "inference.npz", size=2, seed=4, labels=False, graph=False)
    reference = np.linspace(0, 1, 60, dtype=np.float32)
    candidates = np.stack([reference, np.roll(reference, 2), reference[::-1]])
    np.save(args.output_dir / "reference.npy", reference)
    np.save(args.output_dir / "candidates.npy", candidates)
    (args.output_dir / "model.json").write_text(
        "{\n"
        '  "pxrd_length": 60,\n'
        '  "smiles_embedding_dim": 8,\n'
        '  "num_classes": 3,\n'
        '  "atom_dim": 4,\n'
        '  "nbr_dim": 5,\n'
        '  "hidden_dim": 8,\n'
        '  "projection_dim": 4,\n'
        '  "graph_layers": 1,\n'
        '  "dropout": 0.0,\n'
        '  "prediction_modality": "pxrd_smiles"\n'
        "}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
