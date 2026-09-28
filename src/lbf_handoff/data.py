"""Dataset inventory and the frozen identity-disjoint protocol.

The most important protection in this project is not a model setting; it is
the split boundary.  This module reads explicit decisions from the manifest
instead of randomly reconstructing them.  That makes accidental evaluation
leakage both less likely and easier to audit.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from . import DATASETS


def read_samples(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 9_742:
        raise ValueError(f"Expected 9,742 photographs, found {len(rows):,}")
    return rows


def dataset_rows(rows: list[dict[str, str]], dataset: str) -> list[dict[str, str]]:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset}")
    selected = [row for row in rows if row["dataset"] == dataset]
    selected.sort(key=lambda row: int(row["dataset_index"]))
    return selected


def protocol_indices(
    rows: list[dict[str, str]], dataset: str, partition: str
) -> tuple[list[int], list[dict[str, object]]]:
    """Return enrollment indices and labelled queries for one frozen partition.

    Dataset indices address the bundled template arrays directly.  Failed
    preprocessing rows remain in the inventory for transparency but never
    enter enrollment or querying.
    """
    subset = dataset_rows(rows, dataset)
    enrollment_tag = f"{partition}_enrollment"
    genuine_tag = f"{partition}_genuine_query"
    impostor_tag = f"{partition}_impostor_query"
    enrollment = [
        int(row["dataset_index"]) for row in subset if row["protocol_use"] == enrollment_tag
    ]
    queries: list[dict[str, object]] = []
    # The accepted evidence groups identities alphabetically and photographs
    # by numeric capture order.  Keeping that order makes reproduced CSV files
    # directly diffable (for example, capture 9 precedes capture 10).
    ordered = sorted(
        subset,
        key=lambda row: (row["identity"], int(row["capture_order"]), row["relative_path"]),
    )
    for row in ordered:
        use = row["protocol_use"]
        if use not in {genuine_tag, impostor_tag}:
            continue
        queries.append(
            {
                "dataset_index": int(row["dataset_index"]),
                "identity": row["identity"],
                "relative_path": row["relative_path"],
                "label": int(use == genuine_tag),
            }
        )
    if not enrollment or {int(query["label"]) for query in queries} != {0, 1}:
        raise ValueError(f"Incomplete protocol for {dataset}/{partition}")
    return enrollment, queries


def validate_protocol(rows: list[dict[str, str]]) -> dict[str, object]:
    """Run the inexpensive checks that should precede every new experiment."""
    identities_by_partition: dict[str, set[tuple[str, str]]] = {}
    for partition in ("fit", "development", "evaluation"):
        identities_by_partition[partition] = {
            (row["dataset"], row["identity"])
            for row in rows
            if row["partition"] == partition
        }
    for left, right in (("fit", "development"), ("fit", "evaluation"), ("development", "evaluation")):
        overlap = identities_by_partition[left] & identities_by_partition[right]
        if overlap:
            raise ValueError(f"Identity leakage between {left} and {right}: {sorted(overlap)[:3]}")

    counts: dict[str, object] = {}
    for dataset in DATASETS:
        subset = dataset_rows(rows, dataset)
        counts[dataset] = {
            "images": len(subset),
            "identities": len({row["identity"] for row in subset}),
            "uses": dict(Counter(row["protocol_use"] for row in subset)),
        }
        # Dataset indices must be a dense address space for the frozen arrays.
        indices = [int(row["dataset_index"]) for row in subset]
        if indices != list(range(len(subset))):
            raise ValueError(f"Non-contiguous dataset indices for {dataset}")
    return counts


def load_packed_templates(path: Path) -> dict[str, np.ndarray]:
    """Load compact packed bits; unpacking is deferred until a pipeline is used."""
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def unpack_templates(packed: np.ndarray, bit_length: int) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1, bitorder="little")
    return bits[:, :bit_length].astype(np.bool_, copy=False)
