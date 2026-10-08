#!/usr/bin/env python3
"""Evaluate an XRayPro checkpoint on a labeled NPZ bundle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.arrays import ArrayDataset, load_array_bundle
from src.evaluation.metrics import topk_summary, classification_details
from src.runtime.checkpoints import load_checkpoint
from src.runtime.io import display_space_group, move_batch, resolve_device, validate_bundle_for_model


def run(args: argparse.Namespace) -> dict[str, float]:
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    bundle = load_array_bundle(
        args.data,
        prediction_modality=model.config.prediction_modality,
        require_labels=True,
        require_graph=False,
    )
    validate_bundle_for_model(bundle, model.config)
    groups = model.config.class_space_groups
    loader = DataLoader(ArrayDataset(bundle), batch_size=args.batch_size, shuffle=False)
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits_all.append(model(batch)["multi_logits"].cpu())
            labels_all.append(batch["label"].cpu())
            ids = batch.get("sample_id")
            if ids is None:
                start = len(sample_ids)
                sample_ids.extend(str(index) for index in range(start, start + len(batch["label"])))
            else:
                sample_ids.extend(str(value) for value in ids)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    metrics = topk_summary(logits, labels)
    details = classification_details(logits, labels)
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classification_details.json").write_text(
        json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["sample_id", "label", "pred_top1", "pred_top3"]
        if groups is not None:
            header.extend(["label_space_group", "pred_top1_space_group", "pred_top3_space_groups"])
        writer.writerow(header)
        for index in range(labels.numel()):
            row = [
                sample_ids[index],
                int(labels[index]),
                int(order[index, 0]),
                ";".join(str(int(value)) for value in order[index, :3]),
            ]
            if groups is not None:
                row.extend([
                    display_space_group(groups[int(labels[index])]),
                    display_space_group(groups[int(order[index, 0])]),
                    ";".join(display_space_group(groups[int(value)]) for value in order[index, :3]),
                ])
            writer.writerow(row)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
