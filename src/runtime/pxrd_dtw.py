"""Constrained DTW for PXRD candidate ranking."""

from __future__ import annotations

import math

import numpy as np


def normalize_max100(values: np.ndarray, *, subtract_minimum: bool = False) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    if output.size == 0 or not np.isfinite(output).all():
        raise ValueError("PXRD profiles must be non-empty and finite")
    output = np.clip(output, 0.0, None)
    if subtract_minimum:
        output -= np.min(output, axis=-1, keepdims=True)
    output = np.clip(output, 0.0, None)
    maximum = np.max(output, axis=-1, keepdims=True)
    if np.any(maximum <= 0) or not np.isfinite(maximum).all():
        raise ValueError("PXRD profile has no finite positive signal after preprocessing")
    return np.divide(output, maximum, out=np.zeros_like(output), where=maximum > 0) * 100.0


def max_pool_rows(values: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError("pooling factor must be positive")
    usable = values.shape[-1] // factor * factor
    if usable < factor:
        raise ValueError("profile has fewer points than one pooling bin")
    reshaped = values[..., :usable].reshape(*values.shape[:-1], usable // factor, factor)
    return reshaped.max(axis=-1)


def prepare_profile_pair(
    reference: np.ndarray,
    candidate: np.ndarray,
    pooling_factor: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the paper protocol before constrained DTW.

    The observed profile receives minimum-baseline subtraction; candidate
    profiles do not. Both profiles are max-normalized with linear intensities,
    max-pooled and L2-normalized.
    """

    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim != 1 or candidate.ndim != 1 or reference.shape != candidate.shape:
        raise ValueError("reference and candidate must be equal-length one-dimensional arrays")
    prepared_reference = normalize_max100(reference, subtract_minimum=True)
    prepared_candidate = normalize_max100(candidate, subtract_minimum=False)
    prepared_reference = max_pool_rows(prepared_reference[None, :], pooling_factor)[0]
    prepared_candidate = max_pool_rows(prepared_candidate[None, :], pooling_factor)[0]
    reference_norm = float(np.linalg.norm(prepared_reference))
    candidate_norm = float(np.linalg.norm(prepared_candidate))
    if not (0 < reference_norm < np.inf and 0 < candidate_norm < np.inf):
        raise ValueError("PXRD profile has no finite positive signal after pooling")
    prepared_reference = prepared_reference / reference_norm
    prepared_candidate = prepared_candidate / candidate_norm
    return prepared_reference, prepared_candidate


def constrained_dtw_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
    band_points: int,
) -> float:
    # Euclidean local cost in one dimension is absolute difference.
    # dtaidistance window=31 permits |i-j| <= 30; no endpoint relaxation.
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or candidate.shape != reference.shape:
        raise ValueError("DTW inputs must be equal-length one-dimensional arrays")
    if reference.size == 0 or not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        raise ValueError("DTW inputs must be non-empty and finite")
    if band_points < 0:
        raise ValueError("band_points must be non-negative")
    length = reference.size
    previous = np.full(length + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for row in range(1, length + 1):
        current = np.full(length + 1, np.inf, dtype=np.float64)
        start = max(1, row - band_points)
        stop = min(length, row + band_points)
        for column in range(start, stop + 1):
            cost = abs(float(reference[row - 1] - candidate[column - 1]))
            current[column] = cost + min(
                previous[column], current[column - 1], previous[column - 1]
            )
        previous = current
    return float(previous[length] / math.sqrt(length))


def score_candidates(
    reference: np.ndarray,
    candidates: np.ndarray,
    *,
    pooling_factor: int = 5,
    band_points: int = 30,
) -> np.ndarray:
    candidates = np.asarray(candidates)
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("candidates must have non-empty shape [N, L]")
    scores = np.empty(candidates.shape[0], dtype=np.float64)
    for index, candidate in enumerate(candidates):
        prepared_reference, prepared_candidate = prepare_profile_pair(
            reference, candidate, pooling_factor=pooling_factor
        )
        scores[index] = constrained_dtw_distance(
            prepared_reference, prepared_candidate, band_points=band_points
        )
    return scores


def screen_candidates(reference, candidates, energies, identifiers, *, energy_limit=200,
                      top_count=10, pooling_factor=5, band_points=30):
    """Screen already deduplicated candidates pooled over selected space groups.

    This routine never uses packing labels or inserts an external reference.
    Energies must be comparable within this one molecular system.
    """
    candidates = np.asarray(candidates)
    energies = np.asarray(energies, dtype=float)
    identifiers = np.asarray(identifiers)
    if candidates.ndim != 2 or energies.shape != (len(candidates),) or identifiers.shape != energies.shape:
        raise ValueError("profiles, energies and identifiers must have matching row counts")
    if energies.size == 0 or not np.isfinite(energies).all():
        raise ValueError("energies must be non-empty and finite")
    if identifiers.dtype.kind not in "US" or any(not value for value in identifiers.tolist()):
        raise ValueError("candidate identifiers must be non-empty strings")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("candidate identifiers must be unique")
    if type(energy_limit) is not int or type(top_count) is not int or not 0 < top_count <= energy_limit:
        raise ValueError("budgets must satisfy 0 < top_count <= energy_limit")
    kept = np.lexsort((identifiers, energies))[:energy_limit]
    scores = score_candidates(reference, candidates[kept], pooling_factor=pooling_factor,
                              band_points=band_points)
    order = np.lexsort((identifiers[kept], energies[kept], scores))
    return [{"rank": rank, "candidate_index": int(kept[i]),
             "candidate_id": str(identifiers[kept[i]]), "energy": float(energies[kept[i]]),
             "energy_rank": int(i) + 1, "dtw_distance": float(scores[i]),
             "selected": rank <= top_count} for rank, i in enumerate(order, 1)]
