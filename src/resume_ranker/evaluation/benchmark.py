"""Benchmark helpers for command-line and CI smoke runs."""

from __future__ import annotations

from typing import Dict


def format_stage_table(stage_seconds: Dict[str, float]) -> str:
    lines = ["Stage timings:"]
    for name, seconds in stage_seconds.items():
        lines.append(f"  - {name}: {seconds:.2f}s")
    return "\n".join(lines)
