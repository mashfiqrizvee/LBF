"""Small reusable diagnostics retained because they explain the final result."""

from __future__ import annotations

import numpy as np

from .bloom import Block


def independent_block_survival(disagreement_rate: float, block_length: int) -> float:
    """Model prediction (1 - e/P)^l; this is not a measured firing rate."""
    return float((1.0 - disagreement_rate) ** block_length)


def constant_positions(bits: np.ndarray) -> np.ndarray:
    """Boolean mask of positions that never change across the supplied templates."""
    bits = np.asarray(bits, dtype=np.bool_)
    return np.all(bits == bits[:1], axis=0)


def constant_blocks(bits: np.ndarray, layout: list[Block]) -> list[str]:
    positions = constant_positions(bits)
    return [block.name for block in layout if bool(np.all(positions[block.start:block.end]))]


def observed_fire_rate(match_matrix: np.ndarray) -> float:
    """Mean measured fraction of filters firing across queries."""
    values = np.asarray(match_matrix, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("Expected query-by-filter matches")
    return float(values.mean())


def bit_disagreement(reference: np.ndarray, probes: np.ndarray) -> float:
    reference, probes = np.asarray(reference, dtype=np.bool_), np.asarray(probes, dtype=np.bool_)
    return float(np.mean(probes != reference))

