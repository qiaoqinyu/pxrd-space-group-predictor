"""Training objectives for graph-aligned multimodal classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.runtime.contracts import validate_classification_tensors


@dataclass(frozen=True)
class LossConfig:
    multimodal_ce_weight: float = 0.2
    graph_ce_weight: float = 0.2
    alignment_weight: float = 1.0
    barlow_lambda: float = 0.005
    label_smoothing: float = 0.0


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"expected a square matrix, got {tuple(matrix.shape)}")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


def barlow_alignment_loss(
    graph_projection: torch.Tensor,
    multimodal_projection: torch.Tensor,
    *,
    coefficient: float = 0.005,
) -> torch.Tensor:
    if graph_projection.shape != multimodal_projection.shape:
        raise ValueError("graph and multimodal projections must have equal shapes")
    if graph_projection.shape[0] < 2:
        return F.mse_loss(
            F.normalize(graph_projection, dim=-1),
            F.normalize(multimodal_projection, dim=-1),
        )
    graph_projection = (graph_projection - graph_projection.mean(0)) / graph_projection.std(
        0, unbiased=False
    ).clamp_min(1e-6)
    multimodal_projection = (
        multimodal_projection - multimodal_projection.mean(0)
    ) / multimodal_projection.std(0, unbiased=False).clamp_min(1e-6)
    correlation = graph_projection.T @ multimodal_projection / graph_projection.shape[0]
    diagonal_loss = torch.diagonal(correlation).sub(1).pow(2).sum()
    redundancy_loss = off_diagonal(correlation).pow(2).sum()
    return diagonal_loss + float(coefficient) * redundancy_loss


def classification_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    config: LossConfig,
) -> dict[str, torch.Tensor]:
    validate_classification_tensors(output["multi_logits"], labels)
    multimodal_ce = F.cross_entropy(
        output["multi_logits"], labels, label_smoothing=config.label_smoothing
    )
    zero = multimodal_ce.new_zeros(())
    graph_ce = zero
    alignment = zero
    if config.graph_ce_weight > 0 or config.alignment_weight > 0:
        missing = {"graph_logits", "graph_proj", "multi_proj"}.difference(output)
        if missing:
            raise ValueError(
                "graph supervision is enabled but graph outputs are missing: "
                + ", ".join(sorted(missing))
            )
        validate_classification_tensors(output["graph_logits"], labels)
        graph_ce = F.cross_entropy(
            output["graph_logits"], labels, label_smoothing=config.label_smoothing
        )
        alignment = barlow_alignment_loss(
            output["graph_proj"],
            output["multi_proj"],
            coefficient=config.barlow_lambda,
        )
    total = (
        config.multimodal_ce_weight * multimodal_ce
        + config.graph_ce_weight * graph_ce
        + config.alignment_weight * alignment
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("non-finite classification or alignment loss")
    return {
        "loss": total,
        "multimodal_ce": multimodal_ce,
        "graph_ce": graph_ce,
        "alignment": alignment,
    }
