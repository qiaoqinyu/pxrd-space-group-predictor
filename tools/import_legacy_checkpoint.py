#!/usr/bin/env python3
"""Convert a verified historical SE-Res/simple-graph checkpoint to schema 1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.xraypro import ModelConfig, XRayProModel
from src.runtime.checkpoints import CHECKPOINT_SCHEMA_VERSION


def convert(source: Path, config_path: Path, output: Path, expected_sha256: str,
            provenance_label: str):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("source checkpoint SHA256 differs from the verified value")
    if not provenance_label.strip():
        raise ValueError("provide an explicit model provenance label")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    if config.class_space_groups is None:
        raise ValueError("config must include the verified class-index-to-space-group mapping")
    if config.prediction_modality != "pxrd_smiles":
        raise ValueError("legacy importer supports only PXRD + embedding inference")
    if config.pxrd_length != 3400:
        raise ValueError("legacy importer requires the verified 3400-point classification grid")
    args = payload.get("args", {})
    required = {"pxrd_encoder": "seres", "graph_encoder": "simple", "smiles_mode": "embedding",
                "hidden_dim": config.hidden_dim, "projection_dim": config.projection_dim,
                "graph_layers": config.graph_layers, "dropout": config.dropout}
    if any(args.get(key) != value for key, value in required.items()):
        raise ValueError("unsupported legacy architecture or inconsistent model configuration")
    if args.get("prediction_modality", "pxrd_smiles") != "pxrd_smiles":
        raise ValueError("legacy prediction modality must be pxrd_smiles")
    model = XRayProModel(config)
    replacements = {"graph.atom_proj.": "graph.atom_projection.",
                    "graph.convs.": "graph.convolutions.",
                    "graph.norms.": "graph.normalizations."}
    state = {}
    for key, value in payload["model_state"].items():
        target = key
        for old, new in replacements.items():
            if key.startswith(old):
                target = new + key[len(old):]
                break
        if target in state:
            raise ValueError(f"duplicate state key after conversion: {target}")
        state[target] = value
    model.load_state_dict(state, strict=True)
    converted = {"schema_version": CHECKPOINT_SCHEMA_VERSION,
                 "model_config": config.to_dict(), "model_state": state,
                 "epoch": int(payload["epoch"]), "metrics": {},
                 "provenance": {"source_sha256": digest, "label": provenance_label,
                                "conversion": "graph key renaming only; no parameter changes"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(converted, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--provenance-label", required=True)
    args = parser.parse_args()
    convert(args.source, args.config, args.output, args.expected_sha256, args.provenance_label)
