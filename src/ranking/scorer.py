"""src/ranking/scorer.py -- multi-factor hybrid scoring.

Combines five sub-scores (semantic, title, skill, behavioral,
experience) into one composite score, weighted entirely by config.
"""

import re
from typing import Dict, Set

from src.config import config
from src.schema import Candidate

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


def compute_title_score(candidate: Candidate, jd_keywords: Set[str]) -> float:
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


def compute_skill_score(candidate: Candidate, jd_keywords: Set[str]) -> float:
    matches = 0
    for skill in candidate.skills:
        skill_tokens = set(_WORD_PATTERN.findall(skill.name.lower()))
        if skill_tokens & jd_keywords:
            matches += 1
    return min(matches / config.skill_match_denominator, config.score_max)


def compute_behavioral_score(features: Dict[str, float]) -> float:
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


def compute_experience_score(candidate: Candidate) -> float:
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


def compute_composite_score(
    candidate: Candidate,
    semantic_score: float,
    jd_keywords: Set[str],
    behavioral_features: Dict[str, float],
) -> Dict[str, float]:
    """Compute every sub-score plus the final clamped composite score for
    one candidate. Returns a dict with all components so the reasoner can
    explain exactly which factors drove the rank."""
    title_score = compute_title_score(candidate, jd_keywords)
    skill_score = compute_skill_score(candidate, jd_keywords)
    behavioral_score = compute_behavioral_score(behavioral_features)
    experience_score = compute_experience_score(candidate)

    composite = (
        config.weight_semantic * semantic_score
        + config.weight_title * title_score
        + config.weight_skill * skill_score
        + config.weight_behavioral * behavioral_score
        + config.weight_experience * experience_score
    )
    composite = max(config.score_min, min(config.score_max, composite))

    return {
        "semantic_score": semantic_score,
        "title_score": title_score,
        "skill_score": skill_score,
        "behavioral_score": behavioral_score,
        "experience_score": experience_score,
        "composite_score": composite,
    }
