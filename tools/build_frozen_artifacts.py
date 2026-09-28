#!/usr/bin/env python3
"""One-time bridge from the thesis workspace into this compact handoff.

Ordinary users do not need this script: its outputs are already bundled.  It is
kept so the provenance of the manifest and packed templates is inspectable.
Run it only from the original thesis workspace because the archived evidence
paths intentionally are not dependencies of the normal handoff.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HANDOFF = Path(__file__).resolve().parents[1]
WORKSPACE = HANDOFF.parent
ARCHIVE = WORKSPACE / "superseded/2026-08-22-pre-cleanup/experimentation-full"
MAIN = ARCHIVE / "results/pipelines/20260813-main"
CONDITION = ARCHIVE / "results/conditions/20260813-fei-feret"
DATASETS = ("faces94", "faces95", "faces96", "fei", "feret")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def protocol_use(row: dict[str, str], successful: list[dict[str, str]], partition: str, role: str) -> str:
    if row["success"].lower() != "true":
        return "preprocessing_failed"
    if partition == "fit":
        return "fit_frontend"
    capture = int(row["capture_order"])
    if row["dataset"].startswith("faces"):
        if role == "imposter":
            return f"{partition}_impostor_query"
        ordered = sorted(successful, key=lambda item: (int(item["capture_order"]), item["relative_path"]))
        boundary = max(1, len(ordered) // 2)
        enrollment_indices = {int(item["dataset_index"]) for item in ordered[:boundary]}
        return f"{partition}_enrollment" if int(row["dataset_index"]) in enrollment_indices else f"{partition}_genuine_query"
    # The paired datasets contribute an identity only if both required
    # captures survived preprocessing.  This mirrors the accepted evaluator;
    # silently using half a pair would change both population and query count.
    complete_pair = {int(item["capture_order"]) for item in successful} == {0, 1}
    if not complete_pair:
        return "unused_incomplete_pair"
    if role == "enrolled" and capture == 0:
        return f"{partition}_enrollment"
    if capture == 1:
        return f"{partition}_{'genuine' if role == 'enrolled' else 'impostor'}_query"
    return "unused"


def main() -> None:
    output_rows: list[dict[str, object]] = []
    packed: dict[str, np.ndarray] = {}
    landmarks = json.loads((WORKSPACE / "data/cache/pipeline3/landmark_selection.json").read_text())["selected_landmarks"]
    for dataset in DATASETS:
        run = MAIN if dataset.startswith("faces") else CONDITION
        manifest = [row for row in read_csv(run / "manifest.csv") if row["dataset"] == dataset]
        preprocessing = {int(row["dataset_index"]): row for row in read_csv(run / "preprocessing.csv") if row["dataset"] == dataset}
        splits = {(row["dataset"], row["identity"]): (row["partition"], row["role"]) for row in read_csv(run / "identity_splits.csv")}
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in manifest:
            merged = {**row, **preprocessing[int(row["dataset_index"])]}
            if merged["success"].lower() == "true":
                grouped[row["identity"]].append(merged)
        for row in manifest:
            prep = preprocessing[int(row["dataset_index"])]
            partition, role = splits[(dataset, row["identity"])]
            merged = {**row, **prep}
            output_rows.append({
                "dataset": dataset, "dataset_index": row["dataset_index"], "identity": row["identity"],
                "capture_order": row["capture_order"], "relative_path": row["relative_path"], "sha256": row["sha256"],
                "partition": partition, "membership_role": role,
                "protocol_use": protocol_use(merged, grouped[row["identity"]], partition, role),
                "preprocessing_success": prep["success"], "preprocessing_result": prep["result"],
            })

        work = run / "work"
        p1 = np.load(work / f"{dataset}_p1_bits.npy", mmap_mode="r")
        p2 = np.load(work / f"{dataset}_p2_landmark_dct32_bits.npy", mmap_mode="r")[:, landmarks, :12].reshape(len(manifest), 576)
        p3_name = "FERET_pipeline3_bits.npy" if dataset == "feret" else f"{dataset}_pipeline3_bits.npy"
        p3 = np.load(WORKSPACE / "data/cache/pipeline3" / p3_name, mmap_mode="r")[:, landmarks, :].reshape(len(manifest), 576)
        for name, values in (("P1", p1), ("P2", p2), ("P3", p3)):
            packed[f"{dataset}_{name}"] = np.packbits(values, axis=1, bitorder="little")

    path = HANDOFF / "data/selected_samples.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    np.savez_compressed(HANDOFF / "frozen/templates.npz", **packed)
    print(f"Wrote {len(output_rows):,} manifest rows and {len(packed)} packed template arrays")


if __name__ == "__main__":
    main()
