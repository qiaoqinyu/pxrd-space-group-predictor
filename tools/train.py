#!/usr/bin/env python3
"""Train the graph-aligned XRayPro space-group classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.arrays import ArrayDataset, load_array_bundle
from src.evaluation.metrics import topk_summary
from src.models.xraypro import XRayProModel
from src.runtime.checkpoints import load_checkpoint, save_checkpoint
from src.runtime.io import (
    model_config_from_bundle,
    move_batch,
    resolve_device,
    validate_bundle_for_model,
)
from src.training.losses import LossConfig, classification_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: XRayProModel,
    loader: DataLoader,
    device: torch.device,
    loss_config: LossConfig,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch)
            losses = classification_loss(output, batch["label"], loss_config)
            count = int(batch["label"].numel())
            total_loss += float(losses["loss"].item()) * count
            total += count
            logits_all.append(output["multi_logits"].cpu())
            labels_all.append(batch["label"].cpu())
    metrics = topk_summary(torch.cat(logits_all), torch.cat(labels_all))
    metrics["loss"] = total_loss / total
    return metrics


def make_optimizer(model: XRayProModel, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_prefixes = ("multi_fuse.", "multi_proj.", "multi_head.", "graph_head.", "graph_proj.")
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(head_prefixes)
    ]
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(head_prefixes)
    ]
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.learning_rate},
            {
                "params": head_parameters,
                "lr": args.head_learning_rate or args.learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    require_graph = not args.no_graph_supervision
    train_bundle = load_array_bundle(
        args.train,
        prediction_modality=args.prediction_modality,
        require_labels=True,
        require_graph=require_graph,
    )
    val_bundle = load_array_bundle(
        args.val,
        prediction_modality=args.prediction_modality,
        require_labels=True,
        require_graph=require_graph,
    )
    initialized = None
    if args.init_checkpoint is not None:
        initialized, _ = load_checkpoint(args.init_checkpoint, device=device)
    config = model_config_from_bundle(
        args.config,
        train_bundle,
        prediction_modality=args.prediction_modality,
        base_config=initialized.config if initialized is not None else None,
    )
    validate_bundle_for_model(val_bundle, config)
    model = XRayProModel(config).to(device)
    if initialized is not None:
        if initialized.config != model.config:
            raise ValueError("initial checkpoint model configuration does not match")
        model.load_state_dict(initialized.state_dict(), strict=True)

    train_loader = DataLoader(
        ArrayDataset(train_bundle, augment_csd=args.stage == "csd-pretrain", seed=args.seed),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        ArrayDataset(val_bundle), batch_size=args.eval_batch_size, shuffle=False
    )
    loss_config = LossConfig(
        multimodal_ce_weight=args.multimodal_ce_weight,
        graph_ce_weight=0.0 if args.no_graph_supervision else args.graph_ce_weight,
        alignment_weight=0.0 if args.no_graph_supervision else args.alignment_weight,
        barlow_lambda=args.barlow_lambda,
    )
    optimizer = make_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=args.scheduler_patience, min_lr=5e-7
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best: dict[str, float] | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loader.dataset.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = classification_loss(output, batch["label"], loss_config)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            count = int(batch["label"].numel())
            train_loss += float(losses["loss"].item()) * count
            train_count += count
        val_metrics = evaluate(model, val_loader, device, loss_config)
        scheduler.step(val_metrics["top3"])
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss / train_count,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)
        is_better = best is None or val_metrics[args.selection_metric] > best[args.selection_metric] or (
            val_metrics[args.selection_metric] == best[args.selection_metric] and val_metrics["loss"] < best["loss"]
        )
        if is_better:
            best = val_metrics
            best_epoch = epoch
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                epoch=epoch,
                metrics=val_metrics,
                optimizer=optimizer,
            )
        if epoch - best_epoch >= args.early_stop_patience:
            break
    summary = {
        "status": "ok",
        "stage": args.stage,
        "selection_metric": args.selection_metric,
        "scheduler_metric": "top3",
        "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "best_epoch": best_epoch,
        "best": best,
        "model_config": config.to_dict(),
        "loss_config": loss_config.__dict__,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("csd-pretrain", "experimental"), default="experimental")
    parser.add_argument("--selection-metric", choices=("top1", "top3"))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--prediction-modality",
        choices=("pxrd_smiles", "smiles", "pxrd"),
        default="pxrd_smiles",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=240427)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--multimodal-ce-weight", type=float, default=0.2)
    parser.add_argument("--graph-ce-weight", type=float, default=0.2)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--barlow-lambda", type=float, default=0.005)
    parser.add_argument("--no-graph-supervision", action="store_true")
    args = parser.parse_args()
    defaults = ({"epochs": 150, "batch_size": 128, "eval_batch_size": 256,
                 "learning_rate": 1e-4, "head_learning_rate": 1e-4,
                 "selection_metric": "top1"} if args.stage == "csd-pretrain" else
                {"epochs": 120, "batch_size": 32, "eval_batch_size": 128,
                 "learning_rate": 3e-6, "head_learning_rate": 1e-4,
                 "selection_metric": "top3"})
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ("epochs", "batch_size", "eval_batch_size", "early_stop_patience",
                 "learning_rate", "head_learning_rate", "gradient_clip"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
