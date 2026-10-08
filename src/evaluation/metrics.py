"""Classification metrics used by the public evaluation interface."""

from __future__ import annotations

import numpy as np
import torch

from src.runtime.contracts import validate_classification_tensors


def topk_summary(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    validate_classification_tensors(logits, labels)
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    labels_column = labels[:, None]
    result: dict[str, float] = {"n": float(labels.numel())}
    labels_np = labels.detach().cpu().numpy()
    for requested_k in (1, 3, 5):
        k = min(requested_k, logits.shape[1])
        hits = order[:, :k].eq(labels_column).any(dim=1)
        result[f"top{requested_k}"] = float(hits.double().mean().item())
        per_class = []
        hits_np = hits.detach().cpu().numpy()
        for label in np.unique(labels_np):
            per_class.append(float(hits_np[labels_np == label].mean()))
        result[f"balanced_top{requested_k}"] = float(np.mean(per_class))
    confusion = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
    np.add.at(confusion, (labels_np, order[:, 0].cpu().numpy()), 1)
    support = confusion.sum(axis=1)
    denominator = support + confusion.sum(axis=0)
    f1 = np.divide(2 * np.diag(confusion), denominator,
                   out=np.zeros(logits.shape[1]), where=denominator > 0)
    result["macro_f1"] = float(f1[support > 0].mean())
    result["active_classes"] = int((support > 0).sum())
    return result


def classification_details(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Per-class recalls, confusion matrices and ordinary TOP-1 through TOP-10."""
    validate_classification_tensors(logits, labels)
    order = torch.argsort(logits, dim=1, descending=True, stable=True).cpu().numpy()
    truth = labels.cpu().numpy()
    classes = logits.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, order[:, 0]), 1)
    support = confusion.sum(axis=1)
    normalized = np.divide(confusion, support[:, None],
                           out=np.zeros_like(confusion, dtype=float), where=support[:, None] > 0)
    recalls = {}
    for k in (1, 3, 5):
        hits = (order[:, :min(k, classes)] == truth[:, None]).any(axis=1)
        recalls[k] = [float(hits[truth == c].mean()) if support[c] else None
                      for c in range(classes)]
    denominator = support + confusion.sum(axis=0)
    f1 = np.divide(2 * np.diag(confusion), denominator, out=np.zeros(classes), where=denominator > 0)
    return {
        "per_class": [{"class_index": c, "support": int(support[c]),
                       **{f"top{k}": recalls[k][c] for k in (1, 3, 5)},
                       "f1": float(f1[c]) if support[c] else None} for c in range(classes)],
        "confusion": confusion.tolist(), "confusion_row_normalized": normalized.tolist(),
        "rank_coverage": {str(k): float((order[:, :k] == truth[:, None]).any(axis=1).mean())
                          for k in range(1, min(10, classes) + 1)},
        "balanced_denominator": "classes with positive test support",
    }


def paired_seed_summary(baseline: dict[int, float], treatment: dict[int, float]) -> dict:
    """Apply the SI-6 five-pair decision rule to unrounded recall fractions."""
    if len(baseline) != 5 or set(baseline) != set(treatment):
        raise ValueError("exactly five matching seed IDs are required")
    seeds = sorted(baseline)
    base = np.asarray([baseline[s] for s in seeds], dtype=float)
    treated = np.asarray([treatment[s] for s in seeds], dtype=float)
    if not (np.isfinite(base).all() and np.isfinite(treated).all()) or np.any(
            (base < 0) | (base > 1) | (treated < 0) | (treated > 1)):
        raise ValueError("recalls must be finite fractions in [0, 1]")
    gains = treated - base
    mean_gain = float(gains.mean())
    return {"seeds": seeds, "paired_gains": gains.tolist(),
            "baseline_mean": float(base.mean()), "baseline_sample_std": float(base.std(ddof=1)),
            "treatment_mean": float(treated.mean()), "treatment_sample_std": float(treated.std(ddof=1)),
            "mean_gain": mean_gain, "gain_sample_std": float(gains.std(ddof=1)),
            "positive_pairs": int((gains > 0).sum()),
            "criterion_passed": bool(mean_gain >= 0.02 and (gains > 0).sum() >= 4)}
