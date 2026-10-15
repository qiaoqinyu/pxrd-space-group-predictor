"""Classification preprocessing and CSD-only training perturbations (SI-3)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import percentile_filter

CLASSIFICATION_GRID = 2.0 + np.arange(3400, dtype=np.float64) * 0.02


def preprocess_experimental(theta: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Interpolate, subtract a rolling fifth percentile, and scale to max 100.

    Return linear intensities, not the log1p storage representation used by
    some historical data caches. Do not use this baseline correction for DTW.
    """
    theta = np.asarray(theta, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if theta.ndim != 1 or theta.size < 2 or intensity.shape != theta.shape:
        raise ValueError("theta and intensity must be equal-length vectors with >=2 points")
    if not (np.isfinite(theta).all() and np.isfinite(intensity).all()):
        raise ValueError("experimental coordinates and intensities must be finite")
    order = np.argsort(theta, kind="stable")
    theta, intensity = theta[order], np.maximum(intensity[order], 0.0)
    if np.any(np.diff(theta) <= 0):
        raise ValueError("experimental angular coordinates must be unique")
    profile = np.interp(CLASSIFICATION_GRID, theta, intensity, left=0.0, right=0.0)
    baseline = percentile_filter(profile, percentile=5, size=601, mode="nearest")
    profile = np.maximum(profile - baseline, 0.0)
    if not np.any(profile > 0):
        raise ValueError("experimental PXRD has no positive signal after preprocessing; "
                         "check the measured angular range and intensities")
    profile *= 100.0 / profile.max()
    return profile.astype(np.float32)


def augment_csd_profile(profile: torch.Tensor, *, seed: int, epoch: int,
                        index: int, row_index: int) -> torch.Tensor:
    """Apply the original stateless CSD perturbation, preserving padded support."""
    profile = profile.clone()
    if profile.ndim != 1 or not bool(torch.isfinite(profile).all()) or bool((profile < 0).any()):
        raise ValueError("CSD profile must be a finite non-negative vector")

    if profile.numel() == 0 or not bool((profile > 0).any()):
        raise ValueError("CSD PXRD must contain positive signal before augmentation")

    def generator(offset):
        return torch.Generator().manual_seed(
            seed + epoch * 10_000_019 + row_index * 1_000_003 + index + offset)

    def rescale(values, original):
        active = torch.nonzero(original != 0).flatten()
        if active.numel():
            values[:int(active[0])] = 0
            values[int(active[-1]) + 1:] = 0
        if original.max() > 0 and values.max() > 0:
            values = values * (original.max() / values.max())
        return values.clamp_min(0)

    noise_rng = generator(31_337)
    if torch.rand((), generator=noise_rng) < 0.5:
        noise = torch.randn(profile.shape, generator=noise_rng, dtype=profile.dtype)
        profile = rescale((profile + noise * 0.002 * profile.max().clamp_min(torch.finfo(profile.dtype).eps)).clamp_min(0), profile)
    background_rng = generator(47_819)
    if torch.rand((), generator=background_rng) < 0.5:
        maximum = profile.max().clamp_min(torch.finfo(profile.dtype).eps)
        constant = float(torch.rand((), generator=background_rng).item()) * 0.005 * maximum
        air = float(torch.rand((), generator=background_rng).item()) * 0.010 * maximum
        position = torch.linspace(0, 1, profile.numel(), dtype=profile.dtype)
        background = torch.zeros_like(profile) + constant
        background = background + air * torch.exp(-3.0 * position)
        profile = rescale((profile + background).clamp_min(0), profile)
    return profile
