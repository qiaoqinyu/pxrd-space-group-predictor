#!/usr/bin/env python3
"""Run graph-free XRayPro inference from PXRD and molecular embeddings."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.arrays import ArrayDataset, load_array_bundle
from src.runtime.checkpoints import load_checkpoint
from src.runtime.contracts import validate_classification_tensors
from src.runtime.io import display_space_group, move_batch, resolve_device, validate_bundle_for_model


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    bundle = load_array_bundle(
        args.data,
        prediction_modality=model.config.prediction_modality,
        require_labels=False,
        require_graph=False,
    )
    validate_bundle_for_model(bundle, model.config)
    groups = model.config.class_space_groups
    loader = DataLoader(ArrayDataset(bundle), batch_size=args.batch_size, shuffle=False)
    rows: list[list[object]] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)["multi_logits"]
            validate_classification_tensors(logits)
            probabilities = torch.softmax(logits, dim=1).cpu()
            order = torch.argsort(logits.cpu(), dim=1, descending=True, stable=True)
            ids = batch.get("sample_id")
            for index in range(probabilities.shape[0]):
                sample_id = str(ids[index]) if ids is not None else str(offset + index)
                top = order[index, : args.top_k]
                rows.append(
                    [
                        sample_id,
                        ";".join(str(int(value)) for value in top),
                        ";".join(f"{float(probabilities[index, value]):.8g}" for value in top),
                    ]
                )
                if groups is not None:
                    rows[-1].append(";".join(display_space_group(groups[int(value)]) for value in top))
            offset += probabilities.shape[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["sample_id", "predicted_class_indices", "probabilities"]
        if groups is not None:
            header.append("predicted_space_groups")
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
