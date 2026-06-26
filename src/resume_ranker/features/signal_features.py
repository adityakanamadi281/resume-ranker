"""Convert RedrobSignals into normalized feature values.

All thresholds and normalization values are provided by AppConfig.
"""

from datetime import date
from typing import Dict

from resume_ranker.config import AppConfig
from resume_ranker.domain.schema import Candidate


def _open_to_work_score(candidate: Candidate, config: AppConfig) -> float:
    return (
        config.score_open_to_work_true
        if candidate.redrob_signals.open_to_work_flag
        else config.score_open_to_work_false
    )


def _notice_period_score(candidate: Candidate, config: AppConfig) -> float:
    days = candidate.redrob_signals.notice_period_days
    if days <= config.notice_period_ideal_days:
        return config.score_max
    if days <= config.notice_period_max_days:
        span = config.notice_period_max_days - config.notice_period_ideal_days
        if span <= 0:
            return config.score_max
        fraction_through = (days - config.notice_period_ideal_days) / span
        # Linear decay from score_max down to the midpoint between
        # score_max and score_min (i.e. 0.5 of the [min,max] range).
        midpoint = (config.score_max + config.score_min) / 2
        return config.score_max - fraction_through * (config.score_max - midpoint)
    return config.score_min


def _recent_activity_score(candidate: Candidate, today: date, config: AppConfig) -> float:
    days_inactive = (today - candidate.redrob_signals.last_active_date).days
    if days_inactive <= config.recent_activity_days_tier1:
        return config.recent_activity_score_tier1
    if days_inactive <= config.recent_activity_days_tier2:
        return config.recent_activity_score_tier2
    if days_inactive <= config.recent_activity_days_tier3:
        return config.recent_activity_score_tier3
    return config.recent_activity_score_tier4


def _response_speed_score(candidate: Candidate, config: AppConfig) -> float:
    hours = candidate.redrob_signals.avg_response_time_hours
    if hours <= config.response_time_ideal_hours:
        return config.score_max
    if hours <= 0:
        return config.score_max
    ratio = config.response_time_ideal_hours / hours
    return max(ratio, config.response_time_score_floor)


def _profile_views_score(candidate: Candidate, config: AppConfig) -> float:
    views = candidate.redrob_signals.profile_views_received_30d
    return min(views / config.profile_views_normalization, config.score_max)


def _saved_by_recruiters_score(candidate: Candidate, config: AppConfig) -> float:
    saved = candidate.redrob_signals.saved_by_recruiters_30d
    return min(saved / config.saved_by_recruiters_normalization, config.score_max)


def _skill_assessments_score(candidate: Candidate, config: AppConfig) -> float:
    scores = candidate.redrob_signals.skill_assessment_scores
    if not scores:
        return config.score_min
    avg = sum(scores.values()) / len(scores)
    return avg / config.skill_assessment_max_score


def _github_activity_score(candidate: Candidate, config: AppConfig) -> float:
    raw = candidate.redrob_signals.github_activity_score
    if raw < config.score_min:
        return config.score_min
    return max(config.score_min, raw) / config.github_activity_max_score


def _location_fit_score(candidate: Candidate, config: AppConfig) -> float:
    location = candidate.profile.location.lower()
    country = candidate.profile.country.lower()
    if any(city in location for city in config.tier1_cities):
        return config.location_score_tier1
    if country == config.location_country_india_name:
        return config.location_score_india_other
    return config.location_score_abroad


def _willing_to_relocate_score(candidate: Candidate, config: AppConfig) -> float:
    return (
        config.score_willing_to_relocate_true
        if candidate.redrob_signals.willing_to_relocate
        else config.score_willing_to_relocate_false
    )


def _verified_score(candidate: Candidate) -> float:
    sig = candidate.redrob_signals
    flags = [sig.verified_email, sig.verified_phone, sig.linkedin_connected]
    return sum(1.0 for f in flags if f) / len(flags)


def _experience_fit_score(candidate: Candidate, config: AppConfig) -> float:
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


def process(candidate: Candidate, today: date, config: AppConfig) -> Dict[str, float]:
    """Convert a candidate's redrob_signals (plus a couple of profile
    fields needed for location/experience fit) into a normalized feature
    dictionary, every value in [0.0, 1.0]."""
    return {
        "open_to_work": _open_to_work_score(candidate, config),
        "notice_period_score": _notice_period_score(candidate, config),
        "recent_activity": _recent_activity_score(candidate, today, config),
        "response_rate": candidate.redrob_signals.recruiter_response_rate,
        "response_speed": _response_speed_score(candidate, config),
        "interview_completion": candidate.redrob_signals.interview_completion_rate,
        "profile_views": _profile_views_score(candidate, config),
        "saved_by_recruiters": _saved_by_recruiters_score(candidate, config),
        "profile_completeness": candidate.redrob_signals.profile_completeness_score / 100.0,
        "skill_assessments": _skill_assessments_score(candidate, config),
        "github_activity": _github_activity_score(candidate, config),
        "location_fit": _location_fit_score(candidate, config),
        "willing_to_relocate": _willing_to_relocate_score(candidate, config),
        "verified": _verified_score(candidate),
        "experience_fit": _experience_fit_score(candidate, config),
    }
