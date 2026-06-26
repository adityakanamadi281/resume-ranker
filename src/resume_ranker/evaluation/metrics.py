"""Lightweight ranking metrics used for audit reports."""

from __future__ import annotations

from typing import Dict, Iterable, List


def score_distribution(scores: Iterable[float]) -> Dict[str, float]:
    values = sorted(scores)
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0}
    return {
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
    }


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(max(round((len(sorted_values) - 1) * q), 0), len(sorted_values) - 1)
    return sorted_values[idx]
