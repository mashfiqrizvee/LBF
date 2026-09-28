"""Threshold selection, metrics, and exact final-experiment reproduction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import DATASETS, PIPELINES
from .bloom import BloomBank, encode_blocks
from .data import protocol_indices, unpack_templates
from .pipeline1 import layout as p1_layout
from .pipeline2 import layout as local_layout


DISPLAY = {"faces94": "Faces94", "faces95": "Faces95", "faces96": "Faces96", "fei": "FEI", "feret": "FERET"}


def select_threshold(scores: np.ndarray, labels: np.ndarray, maximum: int) -> int:
    """Maximize balanced accuracy; select the higher threshold on an exact tie."""
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    best_threshold, best_numerator = 0, -1
    for threshold in range(maximum + 2):
        accepted = scores >= threshold
        tp = int(np.sum(accepted & (labels == 1)))
        tn = int(np.sum(~accepted & (labels == 0)))
        numerator = tp * negatives + tn * positives
        if numerator > best_numerator or (numerator == best_numerator and threshold > best_threshold):
            best_threshold, best_numerator = threshold, numerator
    return best_threshold


def metric_row(scores: np.ndarray, labels: np.ndarray, threshold: int) -> dict[str, object]:
    decision = scores >= threshold
    tp = int(np.sum(decision & (labels == 1)))
    fn = int(np.sum(~decision & (labels == 1)))
    tn = int(np.sum(~decision & (labels == 0)))
    fp = int(np.sum(decision & (labels == 0)))
    return {
        "threshold": threshold,
        "auc": float(roc_auc_score(labels, scores)),
        "accuracy": float(np.mean(decision == (labels == 1))),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
        "correct": int(np.sum(decision == (labels == 1))),
        "total": len(labels),
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduce(
    samples: list[dict[str, str]], packed: dict[str, np.ndarray], landmarks: list[int],
    output: Path, probability: float = 0.01,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Rebuild development and evaluation banks from frozen final template bits."""
    layouts = {"P1": p1_layout("thesis"), "P2": local_layout(landmarks), "P3": local_layout(landmarks)}
    bit_lengths = {"P1": 4900, "P2": 576, "P3": 576}
    thresholds: dict[str, int] = {}
    final_metrics: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []

    for partition in ("development", "evaluation"):
        # Dataset-major ordering matches the accepted evidence files.  The
        # order has no mathematical effect, but stable CSV order makes a plain
        # diff useful during handoff review.
        for dataset in DATASETS:
            for pipeline in PIPELINES:
                enrollment, queries = protocol_indices(samples, dataset, partition)
                query_indices = [int(query["dataset_index"]) for query in queries]
                labels = np.asarray([int(query["label"]) for query in queries], dtype=np.int8)
                protocol = "several_enrollment_photographs" if dataset.startswith("faces") else "one_enrollment_photograph"
                bits = unpack_templates(packed[f"{dataset}_{pipeline}"], bit_lengths[pipeline])
                codes = encode_blocks(bits, layouts[pipeline])
                bank = BloomBank(layouts[pipeline], len(enrollment), probability)
                bank.enroll(codes, enrollment)
                scores = bank.query(codes, query_indices)
                key = f"{pipeline}|{dataset}"
                if partition == "development":
                    thresholds[key] = select_threshold(scores, labels, len(layouts[pipeline]))
                    continue
                threshold = thresholds[key]
                summary = metric_row(scores, labels, threshold)
                final_metrics.append({
                    "pipeline": pipeline, "dataset": DISPLAY[dataset], "partition": partition,
                    "protocol": protocol, "false_positive_probability": probability,
                    "filters": len(layouts[pipeline]), "template_bits": bit_lengths[pipeline],
                    "enrolled_templates": len(enrollment),
                    "genuine_queries": int(labels.sum()), "imposter_queries": int(len(labels) - labels.sum()),
                    **summary, "m": bank.m, "k": bank.k, "hbf_bank_storage_bytes": bank.storage_bytes,
                })
                for query, score in zip(queries, scores, strict=True):
                    decision = int(score >= threshold)
                    final_predictions.append({
                        "pipeline": pipeline, "dataset": DISPLAY[dataset], "partition": partition,
                        "protocol": protocol, "identity": query["identity"],
                        "relative_path": query["relative_path"], "label": query["label"],
                        "score": int(score), "threshold": threshold, "decision": decision,
                        "correct": int(decision == int(query["label"])),
                    })

    _write_csv(output / "final_metrics.csv", final_metrics)
    _write_csv(output / "final_predictions.csv", final_predictions)
    return final_metrics, final_predictions, thresholds
