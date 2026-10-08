#!/usr/bin/env python3
"""Encode a CSV of SMILES strings with the paper's ChemBERTa representation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "DeepChem/ChemBERTa-100M-MLM"
MODEL_REVISION = "f5c45f44d3061f0346888f5c09db17ec1146d29d"


def read_records(path: Path) -> tuple[list[str], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "smiles" not in rows[0]:
        raise ValueError("input CSV must contain at least one row and a smiles column")
    identifiers = [str(row.get("sample_id", index)) for index, row in enumerate(rows)]
    smiles = [str(row["smiles"]) for row in rows]
    if any(not value.strip() for value in smiles):
        raise ValueError("SMILES values must be non-empty")
    return identifiers, smiles


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def run(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the 'embedding' optional dependencies") from exc
    identifiers, smiles = read_records(args.input)
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.revision, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        args.model_name, revision=args.revision, trust_remote_code=False
    ).to(device)
    model.eval()
    encoded: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(smiles), args.batch_size):
            tokens = tokenizer(
                smiles[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            output = model(**tokens)
            pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            encoded.append(pooled.cpu().to(torch.float32).numpy())
    embeddings = np.concatenate(encoded, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_id=np.asarray(identifiers),
        smiles_embedding=embeddings,
    )
    metadata = {
        "model_name": args.model_name,
        "revision": args.revision,
        "pooling": "attention-mask mean",
        "max_length": args.max_length,
        "embedding_shape": list(embeddings.shape),
        "dtype": "float32",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_length <= 0 or args.batch_size <= 0:
        parser.error("--max-length and --batch-size must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
