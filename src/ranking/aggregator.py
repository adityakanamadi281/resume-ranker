"""src/ranking/aggregator.py -- combine scores and produce final top-N."""

import logging
from typing import List, Tuple

from src.config import config
from src.exceptions import PipelineError

logger = logging.getLogger(__name__)


class Top100Aggregator:
    def __init__(self) -> None:
        pass

    def aggregate(self, scored_candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Sort by score descending, break ties deterministically by
        candidate_id ascending, and return exactly config.top_n
        candidates (or all of them if fewer than config.top_n are
        available)."""
        seen_ids = set()
        for candidate_id, _ in scored_candidates:
            if candidate_id in seen_ids:
                raise PipelineError(f"Duplicate candidate_id in scored_candidates: {candidate_id}")
            seen_ids.add(candidate_id)

        ranked = sorted(scored_candidates, key=lambda pair: (-pair[1], pair[0]))

        top_n = ranked[: config.top_n]

        scores = [s for _, s in top_n]
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            raise PipelineError("Aggregated scores are not non-increasing after sort.")

        expected_len = min(config.top_n, len(scored_candidates))
        if len(top_n) != expected_len:
            raise PipelineError(
                f"Aggregator produced {len(top_n)} candidates, expected {expected_len}."
            )

        logger.info(
            "Aggregated %d candidates -> top %d (top score=%.4f, bottom score=%.4f)",
            len(scored_candidates), len(top_n),
            top_n[0][1] if top_n else float("nan"),
            top_n[-1][1] if top_n else float("nan"),
        )

        return top_n
