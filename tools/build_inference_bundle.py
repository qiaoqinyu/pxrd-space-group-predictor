#!/usr/bin/env python3
"""Join prepared PXRD and molecular embeddings by sample ID, never by row order."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.arrays import load_array_bundle
from src.models.xraypro import ModelConfig
from src.runtime.io import validate_bundle_for_model


def unique_ids(values, source):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "US":
        raise ValueError(f"{source}: sample IDs must be a one-dimensional string array")
    ids = values.tolist()
    if not ids or any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{source}: sample IDs must be non-empty and unique")
    return ids


def build_bundle(manifest: Path, embeddings: Path, config: Path, output: Path):
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"sample_id", "pxrd_path"}.issubset(reader.fieldnames or []):
            raise ValueError("manifest requires sample_id and pxrd_path columns")
        rows = list(reader)
    ids = unique_ids([row["sample_id"] for row in rows], "manifest")
    with np.load(embeddings, allow_pickle=False) as archive:
        embedding_ids = unique_ids(archive["sample_id"], "embeddings")
        vectors = archive["smiles_embedding"]
    if vectors.ndim != 2 or vectors.shape[0] != len(embedding_ids):
        raise ValueError("embedding rows must match embedding IDs")
    if set(ids) != set(embedding_ids):
        raise ValueError("manifest and embedding sample IDs must match exactly")
    lookup = {value: index for index, value in enumerate(embedding_ids)}
    profiles = []
    for row in rows:
        if not row.get("pxrd_path", "").strip():
            raise ValueError("pxrd_path must be non-empty")
        profile = np.load(manifest.parent / row["pxrd_path"], allow_pickle=False)
        if profile.ndim != 1:
            raise ValueError("each prepared PXRD file must contain one profile")
        profiles.append(profile)
    model_config = ModelConfig.from_dict(json.loads(config.read_text(encoding="utf-8")))
    with tempfile.TemporaryDirectory(prefix="pxrd-bundle-") as directory:
        temporary = Path(directory) / "bundle.npz"
        np.savez(temporary, sample_id=np.asarray(ids), pxrd=np.stack(profiles),
                 smiles_embedding=vectors[[lookup[value] for value in ids]])
        bundle = load_array_bundle(temporary, prediction_modality=model_config.prediction_modality,
                                   require_labels=False, require_graph=False)
        validate_bundle_for_model(bundle, model_config)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as destination, temporary.open("rb") as source:
            shutil.copyfileobj(source, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_bundle(args.manifest, args.embeddings, args.config, args.output)
