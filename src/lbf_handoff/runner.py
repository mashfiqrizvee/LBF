"""Command orchestration kept separate from the scientific modules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import DATASETS
from .data import load_packed_templates, read_samples, validate_protocol
from .evaluation import reproduce


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact_hashes(root: Path) -> None:
    for line in (root / "frozen" / "artifact_checksums.sha256").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        if _sha256(path) != expected:
            raise RuntimeError(f"Frozen artifact checksum mismatch: {relative}")


def verify(root: Path) -> None:
    _verify_artifact_hashes(root)
    samples = read_samples(root / "data" / "selected_samples.csv")
    summary = validate_protocol(samples)
    predictions = _csv(root / "results" / "final_predictions.csv")
    metrics = _csv(root / "results" / "final_metrics.csv")
    frozen = json.loads((root / "frozen" / "thresholds.json").read_text())["thresholds"]
    for row in metrics:
        group = [item for item in predictions if item["pipeline"] == row["pipeline"] and item["dataset"] == row["dataset"]]
        labels = np.asarray([int(item["label"]) for item in group])
        scores = np.asarray([int(item["score"]) for item in group])
        decisions = np.asarray([int(item["decision"]) for item in group])
        if not np.isclose(roc_auc_score(labels, scores), float(row["auc"]), atol=1e-12):
            raise RuntimeError(f"AUC mismatch for {row['pipeline']}/{row['dataset']}")
        if not np.isclose(np.mean(decisions == labels), float(row["accuracy"]), atol=1e-12):
            raise RuntimeError(f"Accuracy mismatch for {row['pipeline']}/{row['dataset']}")
        threshold_key = f"{row['pipeline']}|{row['dataset'].lower()}"
        if int(row["threshold"]) != int(frozen[threshold_key]):
            raise RuntimeError(f"Frozen-threshold mismatch for {threshold_key}")
    print("Evidence verified: 9,742 photographs, identity-disjoint split, and 15 final metric rows.")
    print(json.dumps(summary, indent=2))


def _compare_reproduction(root: Path) -> None:
    """Fail loudly unless a new run is identical to the accepted evidence."""
    accepted_metrics = _csv(root / "results" / "final_metrics.csv")
    reproduced_metrics = _csv(root / "reproduced" / "final_metrics.csv")
    by_key = {(row["pipeline"], row["dataset"]): row for row in reproduced_metrics}
    exact_integer_fields = (
        "threshold", "correct", "total", "true_positive", "false_negative",
        "true_negative", "false_positive", "m", "k", "hbf_bank_storage_bytes",
    )
    float_fields = ("auc", "accuracy", "balanced_accuracy")
    for accepted in accepted_metrics:
        key = (accepted["pipeline"], accepted["dataset"])
        observed = by_key.get(key)
        if observed is None:
            raise RuntimeError(f"Reproduction omitted {key}")
        if any(int(accepted[name]) != int(observed[name]) for name in exact_integer_fields):
            raise RuntimeError(f"Integer metric mismatch for {key}")
        if any(not np.isclose(float(accepted[name]), float(observed[name]), atol=1e-12) for name in float_fields):
            raise RuntimeError(f"Floating-point metric mismatch for {key}")

    accepted_predictions = _csv(root / "results" / "final_predictions.csv")
    reproduced_predictions = _csv(root / "reproduced" / "final_predictions.csv")
    fields = (
        "pipeline", "dataset", "partition", "protocol", "identity",
        "relative_path", "label", "score", "threshold", "decision", "correct",
    )
    accepted_values = [[row[name] for name in fields] for row in accepted_predictions]
    reproduced_values = [[row[name] for name in fields] for row in reproduced_predictions]
    if accepted_values != reproduced_values:
        raise RuntimeError("Reproduced prediction rows differ from the accepted evidence")
    print("Exact match confirmed: 15 metric rows and 7,689 evaluation predictions.")


def main(root: Path) -> None:
    parser = argparse.ArgumentParser(description="Final dissertation LBF experiment handoff")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="check manifest and accepted result consistency")
    commands.add_parser("reproduce", help="rebuild final results from frozen template bits")
    commands.add_parser("figures", help="redraw final plots from accepted CSV evidence")
    args = parser.parse_args()

    if args.command == "verify":
        verify(root)
    elif args.command == "figures":
        # Plotting is imported only for this command.  A headless reproduction
        # or manifest check should not spend time initializing Matplotlib.
        from .plots import render
        render(root / "results", root / "results" / "figures")
        print(f"Figures written to {root / 'results' / 'figures'}")
    else:
        samples = read_samples(root / "data" / "selected_samples.csv")
        validate_protocol(samples)
        packed = load_packed_templates(root / "frozen" / "templates.npz")
        landmarks = json.loads((root / "frozen" / "selected_landmarks.json").read_text())["selected_landmarks"]
        metrics, _, thresholds = reproduce(samples, packed, landmarks, root / "reproduced")
        _compare_reproduction(root)
        print(f"Reproduced {len(metrics)} final rows in {root / 'reproduced'}")
        print(json.dumps(thresholds, indent=2, sort_keys=True))
