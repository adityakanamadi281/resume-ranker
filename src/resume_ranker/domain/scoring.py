"""Pure ranking and scoring logic.

Combines five sub-scores (semantic, title, skill, behavioral,
experience) into one composite score. This module intentionally contains
no file-system, FAISS, pandas, or CLI concerns.
"""

import re
from typing import Dict, List, Set, Tuple

import numpy as np
from rapidfuzz import fuzz

from resume_ranker.config import AppConfig
from resume_ranker.domain.schema import Candidate
from resume_ranker.exceptions import PipelineError

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9]*(?:[+#.-][a-zA-Z0-9]+)*")


def extract_jd_keywords(jd_text: str) -> Set[str]:
    """Extract a lowercase keyword set from the JD text for title/skill
    matching. Generic tokenization -- no specific skill or title name is
    referenced here; whatever words actually appear in the JD become the
    keyword set."""
    tokens = _WORD_PATTERN.findall(jd_text.lower())
    return {t for t in tokens if len(t) > 1}


def _candidate_current_title_text(candidate: Candidate) -> str:
    if candidate.career_history:
        current = next((c for c in candidate.career_history if c.is_current), None)
        if current is not None:
            return f"{candidate.profile.headline} {current.title}".lower()
    return candidate.profile.headline.lower()


def compute_title_score(candidate: Candidate, jd_keywords: Set[str], config: AppConfig) -> float:
    title_text = _candidate_current_title_text(candidate)
    title_tokens = set(_WORD_PATTERN.findall(title_text))
    overlap = title_tokens & jd_keywords
    if not title_tokens:
        return config.title_score_no_match
    overlap_fraction = len(overlap) / len(title_tokens)
    if overlap_fraction >= config.score_max:
        return config.title_score_exact_match
    if overlap_fraction > config.score_min:
        return config.title_score_partial_match
    return config.title_score_no_match


def compute_skill_score(candidate: Candidate, jd_keywords: Set[str], config: AppConfig) -> float:
    if not candidate.skills:
        return config.score_min

    matches = 0.0
    jd_keyword_text = " ".join(sorted(jd_keywords))
    for skill in candidate.skills:
        skill_tokens = set(_WORD_PATTERN.findall(skill.name.lower()))
        fuzzy_match = fuzz.partial_ratio(skill.name.lower(), jd_keyword_text) >= 85
        if skill_tokens & jd_keywords or fuzzy_match:
            if skill.duration_months >= config.skill_min_duration_months or skill.endorsements > 0:
                matches += 1.0

    score = matches / config.skill_match_denominator
    if len(candidate.skills) > config.skill_stuffing_threshold:
        penalty = 1.0 - min(
            (len(candidate.skills) - config.skill_stuffing_threshold)
            * config.skill_stuffing_penalty_per_skill,
            config.skill_stuffing_max_penalty,
        )
        score *= penalty

    return min(score, config.score_max)


def compute_behavioral_score(features: Dict[str, float], config: AppConfig) -> float:
    weighted_sum = (
        config.behavioral_weight_open_to_work * features["open_to_work"]
        + config.behavioral_weight_notice_period * features["notice_period_score"]
        + config.behavioral_weight_response_rate * features["response_rate"]
        + config.behavioral_weight_experience_fit * features["experience_fit"]
        + config.behavioral_weight_location_fit * features["location_fit"]
        + config.behavioral_weight_profile_completeness * features["profile_completeness"]
        + config.behavioral_weight_verified * features["verified"]
        + config.behavioral_weight_recent_activity * features["recent_activity"]
    )
    return max(config.score_min, min(config.score_max, weighted_sum))


def compute_experience_score(candidate: Candidate, config: AppConfig) -> float:
    years = candidate.profile.years_of_experience
    if config.experience_ideal_min <= years <= config.experience_ideal_max:
        return config.score_max
    if config.experience_acceptable_min <= years <= config.experience_acceptable_max:
        if years < config.experience_ideal_min:
            span = config.experience_ideal_min - config.experience_acceptable_min
            if span <= 0:
                return config.score_max
            return (years - config.experience_acceptable_min) / span
        span = config.experience_acceptable_max - config.experience_ideal_max
        if span <= 0:
            return config.score_max
        return config.score_max - (years - config.experience_ideal_max) / span
    return config.score_min


def has_tech_career(candidate: Candidate, config: AppConfig) -> bool:
    """Check if the candidate has held at least one tech-related role in their career history."""
    tech_keywords = set(config.tech_career_keywords)
    for entry in candidate.career_history:
        title = entry.title.lower()
        words = set(_WORD_PATTERN.findall(title))
        if words & tech_keywords:
            return True
    return False


def compute_composite_score(
    candidate: Candidate,
    semantic_score: float,
    jd_keywords: Set[str],
    behavioral_features: Dict[str, float],
    config: AppConfig,
) -> Dict[str, float]:
    """Compute every sub-score plus the final clamped composite score for
    one candidate. Returns a dict with all components so the reasoner can
    explain exactly which factors drove the rank."""
    title_score = compute_title_score(candidate, jd_keywords, config)
    skill_score = compute_skill_score(candidate, jd_keywords, config)
    behavioral_score = compute_behavioral_score(behavioral_features, config)
    experience_score = compute_experience_score(candidate, config)

    # Career Alignment: zero out scores of candidates without any tech background
    location_score = behavioral_features["location_fit"]

    if not has_tech_career(candidate, config):
        title_score = 0.0
        skill_score = 0.0
        experience_score = 0.0

    composite = compute_composite_scores_np(
        np.asarray([semantic_score], dtype=np.float32),
        np.asarray([skill_score], dtype=np.float32),
        np.asarray([behavioral_score], dtype=np.float32),
        np.asarray([experience_score], dtype=np.float32),
        np.asarray([location_score], dtype=np.float32),
        config,
    )

    return {
        "semantic_score": semantic_score,
        "title_score": title_score,
        "skill_score": skill_score,
        "behavioral_score": behavioral_score,
        "experience_score": experience_score,
        "location_score": location_score,
        "composite_score": float(composite[0]),
    }


def compute_composite_scores_np(
    semantic_scores: np.ndarray,
    skill_scores: np.ndarray,
    behavior_scores: np.ndarray,
    experience_scores: np.ndarray,
    location_scores: np.ndarray,
    config: AppConfig,
) -> np.ndarray:
    scores = (
        config.weight_semantic * semantic_scores
        + config.weight_skills * skill_scores
        + config.weight_behavior * behavior_scores
        + config.weight_experience * experience_scores
        + config.weight_location * location_scores
    )
    return np.clip(scores, config.score_min, config.score_max).astype(np.float32)  # type: ignore[no-any-return]


class TopNAggregator:
    """Deterministically sort scored candidates and return the top-N."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def aggregate_arrays(self, candidate_ids: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if candidate_ids.shape[0] != scores.shape[0]:
            raise PipelineError("candidate_ids and scores must have the same length.")
        if np.unique(candidate_ids).shape[0] != candidate_ids.shape[0]:
            raise PipelineError("Duplicate candidate_id in scored candidates.")

        order = np.lexsort((candidate_ids.astype(str), -scores))
        top_order = order[: self.config.top_n]
        top_ids = candidate_ids[top_order]
        top_scores = scores[top_order]
        if top_scores.size > 1 and bool((top_scores[:-1] < top_scores[1:]).any()):
            raise PipelineError("Aggregated scores are not non-increasing after sort.")
        return top_ids, top_scores

    def aggregate(self, scored_candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        seen_ids = set()
        for candidate_id, _ in scored_candidates:
            if candidate_id in seen_ids:
                raise PipelineError(f"Duplicate candidate_id in scored_candidates: {candidate_id}")
            seen_ids.add(candidate_id)

        ranked = sorted(scored_candidates, key=lambda pair: (-pair[1], pair[0]))
        top_n = ranked[: self.config.top_n]

        scores = [s for _, s in top_n]
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            raise PipelineError("Aggregated scores are not non-increasing after sort.")

        expected_len = min(self.config.top_n, len(scored_candidates))
        if len(top_n) != expected_len:
            raise PipelineError(
                f"Aggregator produced {len(top_n)} candidates, expected {expected_len}."
            )

        return top_n
