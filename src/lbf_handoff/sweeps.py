"""Helpers for controlled bit-depth, landmark-count, and Bloom-size sweeps."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def run_values(values: Iterable[object], experiment: Callable[[object], dict[str, object]]) -> list[dict[str, object]]:
    """Run one changed value at a time and attach it to each returned record.

    The caller owns the experiment function.  This intentionally small helper
    discourages hidden simultaneous changes to the protocol or frontend.
    """
    rows: list[dict[str, object]] = []
    for value in values:
        result = dict(experiment(value))
        rows.append({"sweep_value": value, **result})
    return rows


def best_development_value(rows: list[dict[str, object]], metric: str = "macro_auc") -> object:
    """Select on development only; callers apply the choice once to evaluation."""
    development = [row for row in rows if row.get("partition") == "development"]
    if not development:
        raise ValueError("No development rows were supplied")
    return max(development, key=lambda row: float(row[metric]))["sweep_value"]

