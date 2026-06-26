"""Pure honeypot detection rules for impossible/artificial profiles.

This module is called before scoring. Honeypots are excluded entirely
from the ranking pipeline.
"""

from datetime import date
from typing import List, Optional, Tuple

from resume_ranker.config import AppConfig
from resume_ranker.domain.schema import Candidate


def _rule_duration_mismatch(candidate: Candidate, today: date, config: AppConfig) -> Optional[str]:
    """Rule 1: career_history.duration_months should match the actual
    number of months between start_date and end_date (or today, if the
    role is current), within a tolerance."""
    for entry in candidate.career_history:
        end = entry.end_date if entry.end_date is not None else today
        actual_months = (end.year - entry.start_date.year) * 12 + (end.month - entry.start_date.month)
        if abs(actual_months - entry.duration_months) > config.honeypot_duration_tolerance_months:
            return (
                f"duration mismatch at {entry.company}: stated "
                f"{entry.duration_months} months, actual dates imply "
                f"~{actual_months} months"
            )
    return None


def _rule_expert_zero_experience(candidate: Candidate) -> Optional[str]:
    """Rule 2: a skill marked 'expert' with zero months of use is
    internally inconsistent."""
    for skill in candidate.skills:
        if skill.proficiency == "expert" and skill.duration_months == 0:
            return f"skill '{skill.name}' marked expert with 0 months of experience"
    return None


def _rule_inverted_salary(candidate: Candidate) -> Optional[str]:
    """Rule 3: expected salary minimum should not exceed the maximum."""
    salary = candidate.redrob_signals.expected_salary_range_inr_lpa
    if salary.min > salary.max:
        return f"inverted salary range: min={salary.min} > max={salary.max}"
    return None


def _rule_job_before_education(candidate: Candidate) -> Optional[str]:
    """Rule 4: no career entry should start before the candidate's
    earliest education end_year."""
    if not candidate.education:
        return None
    earliest_education_end_year = min(e.end_year for e in candidate.education)
    for entry in candidate.career_history:
        if entry.start_date.year < earliest_education_end_year:
            return (
                f"career entry at {entry.company} starts in "
                f"{entry.start_date.year}, before earliest education end "
                f"year {earliest_education_end_year}"
            )
    return None


def _rule_high_completeness_low_connections(
    candidate: Candidate, config: AppConfig
) -> Optional[str]:
    """Rule 5: a near-perfect profile with almost no network connections
    is a common synthetic-profile artifact."""
    sig = candidate.redrob_signals
    completeness_fraction = sig.profile_completeness_score / 100.0
    if (
        completeness_fraction > config.honeypot_high_completeness_threshold
        and sig.connection_count < config.honeypot_low_connections_threshold
    ):
        return (
            f"profile completeness {completeness_fraction:.2f} is very high "
            f"but connection_count is only {sig.connection_count}"
        )
    return None


def _rule_impossible_timeline(candidate: Candidate, today: date) -> Optional[str]:
    """Rule 6: total career experience should not exceed the time elapsed
    since the candidate's earliest education graduation (plus a small
    grace period), since a candidate cannot have more work experience
    than time has passed since they could plausibly have entered the
    workforce."""
    if not candidate.education:
        return None
    earliest_grad_year = min(e.end_year for e in candidate.education)
    years_since_grad = today.year - earliest_grad_year
    total_experience_years = candidate.profile.years_of_experience
    if total_experience_years > years_since_grad + 1:
        return (
            f"{total_experience_years:.1f} years of stated experience exceeds "
            f"{years_since_grad} years since earliest graduation ({earliest_grad_year})"
        )
    return None


def _rule_education_hierarchy_inversion(candidate: Candidate) -> Optional[str]:
    """Rule 7: a higher-level degree (Master's, Ph.D.) should not be completed
    chronologically before a lower-level degree (Bachelor's)."""
    degree_levels = {
        "b.tech": 1, "b.s.": 1, "b.sc.": 1, "bachelor": 1, "ba": 1, "b.e.": 1, "be": 1,
        "b.com": 1, "b.b.a": 1, "bba": 1, "b.f.a": 1, "bfa": 1,
        "m.tech": 2, "m.s.": 2, "m.sc.": 2, "master": 2, "ma": 2, "mba": 2, "m.e.": 2, "me": 2,
        "m.com": 2, "m.phil": 2,
        "ph.d": 3, "phd": 3, "doctorate": 3,
    }

    def get_degree_level(degree_name: str) -> int:
        name = degree_name.lower().replace(".", "")
        for k, v in degree_levels.items():
            if k.replace(".", "") in name:
                return v
        return 0

    edu_list = []
    for edu in candidate.education:
        lvl = get_degree_level(edu.degree)
        if lvl > 0:
            edu_list.append((edu.degree, lvl, edu.end_year))

    for i in range(len(edu_list)):
        for j in range(len(edu_list)):
            if edu_list[i][1] > edu_list[j][1] and edu_list[i][2] < edu_list[j][2]:
                return (
                    f"education timeline inversion: completed higher degree '{edu_list[i][0]}' "
                    f"in {edu_list[i][2]} before lower degree '{edu_list[j][0]}' in {edu_list[j][2]}"
                )
    return None


def detect(candidate: Candidate, today: date, config: AppConfig) -> Tuple[bool, List[str]]:
    """Run every honeypot detection rule against `candidate`. Returns
    (is_honeypot, reasons) where reasons lists a human-readable string
    for every rule that triggered (a candidate can trigger more than
    one)."""
    reasons: List[str] = []

    for result in (
        _rule_duration_mismatch(candidate, today, config),
        _rule_expert_zero_experience(candidate),
        _rule_inverted_salary(candidate),
        _rule_job_before_education(candidate),
        _rule_high_completeness_low_connections(candidate, config),
        _rule_impossible_timeline(candidate, today),
        _rule_education_hierarchy_inversion(candidate),
    ):
        if result is not None:
            reasons.append(result)

    return (len(reasons) > 0, reasons)
