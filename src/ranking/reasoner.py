"""src/ranking/reasoner.py -- rule-based reasoning generation. NO LLM API calls.

Every claim in the generated text is read directly from the candidate's
own data (profile, skills, redrob_signals) -- no fixed template with only
the name swapped in, and no claim that isn't grounded in an actual field
value.
"""

from datetime import date
from typing import List

from src.config import config
from src.schema import Candidate


class RuleReasoner:
    def __init__(self) -> None:
        pass

    def generate(self, candidate: Candidate, rank: int, score: float, today: date) -> str:
        """Generate 1-2 sentence reasoning for why this candidate is
        ranked here, entirely from the candidate's own data."""
        tier = self._tone_tier(rank)

        sentence1 = self._opening_sentence(candidate, tier)

        positives = self._positive_clauses(candidate)
        concerns = self._concern_clauses(candidate, today) if tier != "top" else []

        clauses: List[str] = []
        if positives:
            clauses.append(self._join_clauses(positives, tier))
        if concerns:
            clauses.append(self._join_clauses(concerns, tier, is_concern=True))

        text = sentence1
        if clauses:
            text = text + " " + " ".join(clauses)

        if len(text) > config.reasoner_max_length:
            text = text[: config.reasoner_max_length - 1].rstrip() + "…"

        return text

    # -----------------------------------------------------------------
    # Tone tiers
    # -----------------------------------------------------------------
    def _tone_tier(self, rank: int) -> str:
        t1, t2 = config.reasoner_rank_thresholds
        if rank <= t1:
            return "top"
        if rank <= t2:
            return "mid"
        return "lower"

    # -----------------------------------------------------------------
    # Opening: always-include facts
    # -----------------------------------------------------------------
    def _opening_sentence(self, candidate: Candidate, tier: str) -> str:
        title = candidate.profile.current_title
        years = candidate.profile.years_of_experience

        top_skills = sorted(
            candidate.skills,
            key=lambda s: (s.proficiency == "expert", s.endorsements),
            reverse=True,
        )[: config.reasoner_top_skills_count]
        skill_phrase = self._skills_phrase(top_skills)

        years_phrase = f"{years:.1f} years" if years != int(years) else f"{int(years)} years"

        if tier == "top":
            opener = f"Strong fit: {title} with {years_phrase} of experience."
        else:
            opener = f"{title} with {years_phrase} of experience."

        if skill_phrase:
            opener += f" {skill_phrase}"

        return opener

    def _skills_phrase(self, skills) -> str:
        if not skills:
            return ""
        expert_or_advanced = [s for s in skills if s.proficiency in ("expert", "advanced")]
        chosen = expert_or_advanced if expert_or_advanced else skills
        names = ", ".join(s.name for s in chosen[: config.reasoner_top_skills_count])
        verb = "Expert in" if expert_or_advanced else "Skilled in"
        return f"{verb} {names}."

    # -----------------------------------------------------------------
    # Positive clauses
    # -----------------------------------------------------------------
    def _positive_clauses(self, candidate: Candidate) -> List[str]:
        sig = candidate.redrob_signals
        clauses: List[str] = []

        if sig.recruiter_response_rate > config.reasoner_high_response_rate_threshold:
            clauses.append(f"high recruiter response rate ({sig.recruiter_response_rate:.0%})")

        if sig.notice_period_days <= config.notice_period_ideal_days:
            clauses.append(f"{sig.notice_period_days}-day notice period")

        if sig.open_to_work_flag:
            clauses.append("open to work")

        location = candidate.profile.location.lower()
        if any(city in location for city in config.tier1_cities):
            clauses.append(f"based in {candidate.profile.location} (preferred location)")

        return clauses

    # -----------------------------------------------------------------
    # Concern clauses (lower ranks only)
    # -----------------------------------------------------------------
    def _concern_clauses(self, candidate: Candidate, today: date) -> List[str]:
        sig = candidate.redrob_signals
        years = candidate.profile.years_of_experience
        clauses: List[str] = []

        if years < config.experience_acceptable_min or years > config.experience_acceptable_max:
            clauses.append(f"experience ({years:.1f} yrs) falls outside the typical range")

        if not sig.open_to_work_flag:
            clauses.append("not currently open to work")

        days_inactive = (today - sig.last_active_date).days
        if days_inactive > config.reasoner_low_activity_days_threshold:
            clauses.append(f"inactive for {days_inactive} days")

        if sig.notice_period_days > config.notice_period_max_days:
            clauses.append(f"{sig.notice_period_days}-day notice period may be a concern")

        return clauses

    # -----------------------------------------------------------------
    # Tone-appropriate joining
    # -----------------------------------------------------------------
    def _join_clauses(self, clauses: List[str], tier: str, is_concern: bool = False) -> str:
        joined = "; ".join(clauses)
        if is_concern:
            return f"Note: {joined}." if tier == "lower" else f"{joined.capitalize()}."
        return f"{joined.capitalize()}."
