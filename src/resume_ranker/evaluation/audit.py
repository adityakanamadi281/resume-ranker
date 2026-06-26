"""Generate a compact audit report for judges and operators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import orjson
import polars as pl

from resume_ranker.evaluation.metrics import score_distribution
from resume_ranker.infrastructure.artifact_store import ArtifactManifest


def build_audit_report(
    *,
    total_candidates: int,
    valid_candidates: int,
    honeypots: int,
    ranked_rows: pl.DataFrame,
    stage_seconds: Dict[str, float],
    manifest: ArtifactManifest,
    validation_errors: List[str],
) -> Dict[str, Any]:
    scores = ranked_rows.get_column("score").to_list() if "score" in ranked_rows.columns else []
    return {
        "runtime_seconds": round(sum(stage_seconds.values()), 3),
        "stage_seconds": {k: round(v, 3) for k, v in stage_seconds.items()},
        "candidate_counts": {
            "total": total_candidates,
            "valid_after_honeypot_filter": valid_candidates,
            "honeypots_excluded": honeypots,
            "honeypot_exclusion_rate": round(honeypots / total_candidates, 4)
            if total_candidates
            else 0.0,
        },
        "top_n": {
            "rows": ranked_rows.height,
            "score_distribution": score_distribution(scores),
        },
        "validation": {
            "passed": len(validation_errors) == 0,
            "errors": validation_errors,
        },
        "reproducibility": {
            "candidate_file_hash": manifest.candidate_file_hash,
            "jd_file_hash": manifest.jd_file_hash,
            "model_name": manifest.model_name,
            "embedding_mode": manifest.embedding_mode,
            "config_hash": manifest.config_hash,
        },
    }


def write_audit_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
