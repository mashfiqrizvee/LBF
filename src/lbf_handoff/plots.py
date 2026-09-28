"""Render publication-ready plots from accepted tabular evidence."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


COLORS = {"P1": "#6b7c8c", "P2": "#119da4", "P3": "#f2a03d"}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def render(results: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metrics = _rows(results / "final_metrics.csv")
    predictions = _rows(results / "final_predictions.csv")

    datasets = ["Faces94", "Faces95", "Faces96", "FEI", "FERET"]
    x = np.arange(len(datasets))
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for offset, pipeline in zip((-0.24, 0.0, 0.24), ("P1", "P2", "P3"), strict=True):
        values = [float(next(row["auc"] for row in metrics if row["pipeline"] == pipeline and row["dataset"] == dataset)) for dataset in datasets]
        axis.bar(x + offset, values, width=0.23, label=pipeline, color=COLORS[pipeline])
    axis.set_xticks(x, datasets)
    axis.set_ylim(0.45, 1.02)
    axis.set_ylabel("ROC AUC")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "final_auc_comparison.png", dpi=220)
    plt.close(figure)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        grouped[(row["dataset"], row["pipeline"])].append(row)
    # The sixth panel is the pooled-query view used in the defense figure.  It
    # is computed here from the five datasets; no extra hand-entered curve is
    # stored in the handoff.
    roc_panels = datasets + ["Combined: pooled queries"]
    figure, axes = plt.subplots(3, 2, figsize=(10, 13))
    for axis, dataset in zip(axes.flat, roc_panels, strict=True):
        for pipeline in ("P1", "P2", "P3"):
            rows = (
                [row for source in datasets for row in grouped[(source, pipeline)]]
                if dataset.startswith("Combined")
                else grouped[(dataset, pipeline)]
            )
            labels = np.asarray([int(row["label"]) for row in rows])
            scores = np.asarray([int(row["score"]) for row in rows])
            fpr, tpr, _ = roc_curve(labels, scores)
            axis.step(fpr, tpr, where="post", label=f"{pipeline} ({auc(fpr, tpr):.4f})", color=COLORS[pipeline])
        axis.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
        axis.set_title(dataset)
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "final_roc_grid.png", dpi=220)
    plt.close(figure)
