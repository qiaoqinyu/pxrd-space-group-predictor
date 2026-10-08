#!/usr/bin/env python3
"""Rank simulated candidate PXRD profiles by constrained DTW distance."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime.pxrd_dtw import score_candidates, screen_candidates


def run(args: argparse.Namespace) -> None:
    reference = np.load(args.reference, allow_pickle=False)
    candidates = np.load(args.candidates, allow_pickle=False)
    if reference.ndim != 1:
        raise ValueError("reference must be a one-dimensional NPY array")
    if candidates.ndim != 2 or candidates.shape[1] != reference.shape[0]:
        raise ValueError("candidates must have shape [N, reference_length]")
    start = args.start_index
    stop = args.stop_index if args.stop_index is not None else reference.shape[0]
    if not 0 <= start < stop <= reference.shape[0]:
        raise ValueError("invalid profile crop")
    energies_path = getattr(args, "energies", None)
    ids_path = getattr(args, "candidate_ids", None)
    if (energies_path is None) != (ids_path is None):
        raise ValueError("--energies and --candidate-ids must be supplied together")
    if energies_path is not None:
        rows = screen_candidates(
            reference[start:stop], candidates[:, start:stop],
            np.load(energies_path, allow_pickle=False), np.load(ids_path, allow_pickle=False),
            energy_limit=args.energy_limit, top_count=args.top_count,
            pooling_factor=args.pooling_factor, band_points=args.band_points)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    scores = score_candidates(
        reference[start:stop],
        candidates[:, start:stop],
        pooling_factor=args.pooling_factor,
        band_points=args.band_points,
    )
    order = np.argsort(scores, kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "candidate_index", "dtw_distance"])
        for rank, candidate_index in enumerate(order, start=1):
            writer.writerow([rank, int(candidate_index), f"{float(scores[candidate_index]):.17g}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energies", type=Path, help="Comparable per-molecule energies (.npy).")
    parser.add_argument("--candidate-ids", type=Path, help="Unique string identifiers (.npy).")
    parser.add_argument("--energy-limit", type=int, default=200)
    parser.add_argument("--top-count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--pooling-factor", type=int, default=5)
    parser.add_argument(
        "--band-points",
        type=int,
        default=30,
        help="Warping half-width in pooled-profile index units.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
