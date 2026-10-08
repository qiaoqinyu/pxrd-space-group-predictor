"""Graph-aligned PXRD and molecular-embedding classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.branches import PxrdEncoder
from src.runtime.contracts import (
    normalize_prediction_modality,
    uses_molecular_embedding,
    uses_pxrd,
)


@dataclass(frozen=True)
class ModelConfig:
    pxrd_length: int = 3400
    smiles_embedding_dim: int = 768
    num_classes: int = 11
    atom_dim: int = 92
    nbr_dim: int = 41
    hidden_dim: int = 128
    projection_dim: int = 64
    graph_layers: int = 2
    dropout: float = 0.1
    prediction_modality: str = "pxrd_smiles"
    class_space_groups: list[int | None] | None = None

    def __post_init__(self) -> None:
        normalize_prediction_modality(self.prediction_modality)
        for field_name in (
            "pxrd_length",
            "smiles_embedding_dim",
            "num_classes",
            "atom_dim",
            "nbr_dim",
            "hidden_dim",
            "projection_dim",
            "graph_layers",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.class_space_groups is not None:
            groups = self.class_space_groups
            if not isinstance(groups, list) or len(groups) != self.num_classes:
                raise ValueError("class_space_groups must list one space-group number or null per class")
            if any(group is not None and (type(group) is not int or not 1 <= group <= 230) for group in groups):
                raise ValueError("space-group numbers must be integers in [1, 230]; null denotes Others")
            if len(set(groups)) != len(groups):
                raise ValueError("class_space_groups must not contain duplicates")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(f"unknown model configuration keys: {unknown}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CrystalGraphEncoder(nn.Module):
    """Lightweight graph teacher used for training-time supervision."""

    def __init__(self, atom_dim: int, nbr_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.convolutions = nn.ModuleList(
            [nn.Linear(hidden_dim * 2 + nbr_dim, hidden_dim * 2) for _ in range(layers)]
        )
        self.normalizations = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layers)]
        )

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        if atom_fea.ndim != 3 or nbr_fea.ndim != 4:
            raise ValueError("graph features must have shapes [N, A, F] and [N, A, K, E]")
        if nbr_fea.shape[:2] != atom_fea.shape[:2] or nbr_fea_idx.shape != nbr_fea.shape[:-1]:
            raise ValueError("graph atom and neighbor axes must agree")
        if atom_mask.shape != atom_fea.shape[:2] or atom_mask.dtype != torch.bool:
            raise ValueError("atom_mask must be boolean with shape [N, A]")
        if min(atom_fea.shape) <= 0 or min(nbr_fea.shape) <= 0 or not bool(atom_mask.any(dim=1).all()):
            raise ValueError("each graph must contain atoms and neighbors with at least one real atom")
        if nbr_fea_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("neighbor indices must be integers")
        if bool(((nbr_fea_idx < 0) | (nbr_fea_idx >= atom_fea.shape[1])).any()):
            raise ValueError("neighbor indices must be in [0, atom_count)")
        hidden = self.atom_projection(atom_fea)
        batch_size, atom_count, neighbor_count = nbr_fea_idx.shape
        batch_index = torch.arange(batch_size, device=hidden.device).view(-1, 1, 1)
        batch_index = batch_index.expand(-1, atom_count, neighbor_count)
        if bool((atom_mask.unsqueeze(-1) & ~atom_mask[batch_index, nbr_fea_idx]).any()):
            raise ValueError("real atoms must not reference padded atoms as neighbors")
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            neighbor_hidden = hidden[batch_index, nbr_fea_idx]
            center_hidden = hidden.unsqueeze(2).expand_as(neighbor_hidden)
            message = convolution(
                torch.cat([center_hidden, neighbor_hidden, nbr_fea], dim=-1)
            )
            gate, core = message.chunk(2, dim=-1)
            update = torch.sigmoid(gate) * F.softplus(core)
            hidden = F.softplus(normalization(hidden + update.sum(dim=2)))
            hidden = hidden * atom_mask.unsqueeze(-1)
        denominator = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        return (hidden * atom_mask.unsqueeze(-1)).sum(dim=1) / denominator


class XRayProModel(nn.Module):
    """Formal prediction pathway with an optional training-only graph teacher."""

    GRAPH_KEYS = ("atom_fea", "nbr_fea", "nbr_fea_idx", "atom_mask")

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.prediction_modality = normalize_prediction_modality(config.prediction_modality)
        self.graph = CrystalGraphEncoder(
            atom_dim=config.atom_dim,
            nbr_dim=config.nbr_dim,
            hidden_dim=config.hidden_dim,
            layers=config.graph_layers,
        )
        self.pxrd: nn.Module | None = None
        if uses_pxrd(self.prediction_modality):
            self.pxrd = PxrdEncoder(
                input_length=config.pxrd_length,
                output_dim=config.hidden_dim,
                dropout=config.dropout,
            )
        self.smiles: nn.Module | None = None
        if uses_molecular_embedding(self.prediction_modality):
            self.smiles = nn.Sequential(
                nn.LayerNorm(config.smiles_embedding_dim),
                nn.Linear(config.smiles_embedding_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        input_dim = config.hidden_dim * (
            int(uses_pxrd(self.prediction_modality))
            + int(uses_molecular_embedding(self.prediction_modality))
        )
        self.multi_fuse = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.graph_proj = nn.Linear(config.hidden_dim, config.projection_dim)
        self.multi_proj = nn.Linear(config.hidden_dim, config.projection_dim)
        self.multi_head = nn.Linear(config.hidden_dim, config.num_classes)
        self.graph_head = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prediction_features: list[torch.Tensor] = []
        if uses_pxrd(self.prediction_modality):
            if self.pxrd is None or "pxrd" not in batch:
                raise ValueError("PXRD input is required by the configured prediction modality")
            prediction_features.append(self.pxrd(batch["pxrd"]))
        if uses_molecular_embedding(self.prediction_modality):
            if self.smiles is None or "smiles_embedding" not in batch:
                raise ValueError(
                    "molecular embedding input is required by the configured prediction modality"
                )
            prediction_features.append(self.smiles(batch["smiles_embedding"]))
        fused = prediction_features[0] if len(prediction_features) == 1 else torch.cat(
            prediction_features, dim=-1
        )
        multimodal = self.multi_fuse(fused)
        output = {
            "multi_z": multimodal,
            "multi_proj": self.multi_proj(multimodal),
            "multi_logits": self.multi_head(multimodal),
        }
        graph_presence = [key in batch for key in self.GRAPH_KEYS]
        if any(graph_presence) and not all(graph_presence):
            missing = [key for key in self.GRAPH_KEYS if key not in batch]
            raise ValueError(f"partial graph input; missing {missing}")
        if all(graph_presence):
            graph = self.graph(
                batch["atom_fea"],
                batch["nbr_fea"],
                batch["nbr_fea_idx"],
                batch["atom_mask"],
            )
            output.update(
                {
                    "graph_z": graph,
                    "graph_proj": self.graph_proj(graph),
                    "graph_logits": self.graph_head(graph),
                }
            )
        return output
