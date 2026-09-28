"""Pipeline 1: global 70-by-70 pixel hierarchy."""

from __future__ import annotations

import numpy as np

from .bloom import Block


def _blocks(length: int, size: int, prefix: str, level: str, partial: bool) -> list[Block]:
    stops = range(0, length, size) if partial else range(0, length - size + 1, size)
    return [Block(start, min(start + size, length), f"{prefix}_{i:04d}", level) for i, start in enumerate(stops)]


def layout(mode: str = "thesis") -> list[Block]:
    """Return the original 165-filter or corrected 167-filter hierarchy."""
    if mode not in {"original", "thesis"}:
        raise ValueError("mode must be 'original' or 'thesis'")
    keep_partial = mode == "thesis"
    result = _blocks(4900, 32, "pixel", "lowest", keep_partial)
    result += _blocks(4900, 416, "middle", "middle", keep_partial)
    result.append(Block(0, 4900, "top_complete", "top"))
    expected = 167 if keep_partial else 165
    if len(result) != expected:
        raise RuntimeError(f"P1 layout produced {len(result)} filters, expected {expected}")
    return result


def fit_mean_face(fit_pixels: np.ndarray) -> np.ndarray:
    """Fit one mean intensity per pixel using fit identities only."""
    fit_pixels = np.asarray(fit_pixels)
    if fit_pixels.ndim != 2 or fit_pixels.shape[1] != 4900:
        raise ValueError("Expected an N-by-4,900 pixel matrix")
    return fit_pixels.astype(np.float64).mean(axis=0)


def encode(pixels: np.ndarray, mean_face: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels)
    mean_face = np.asarray(mean_face)
    if pixels.ndim != 2 or pixels.shape[1] != 4900 or mean_face.shape != (4900,):
        raise ValueError("P1 inputs do not have the frozen 4,900-pixel geometry")
    return pixels > mean_face[None, :]

