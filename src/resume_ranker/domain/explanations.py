"""Rule-based reasoning generation. No LLM API calls.

Every claim in the generated text is read directly from the candidate's
own data. The reasoner stays in the domain layer because it is pure
business logic: no files, no indexes, no external services.
"""

from datetime import date
from typing import List

from resume_ranker.config import AppConfig
from resume_ranker.domain.schema import Candidate, Skill


class RuleReasoner:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate(self, candidate: Candidate, rank: int, score: float, today: date) -> str:
        """Generate 1-2 grounded sentences for a ranked candidate."""
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

        if len(text) > self.config.reasoner_max_length:
            text = text[: self.config.reasoner_max_length - 3].rstrip() + "..."

        return text

    def _tone_tier(self, rank: int) -> str:
        t1, t2 = self.config.reasoner_rank_thresholds
        if rank <= t1:
            return "top"
        if rank <= t2:
            return "mid"
        return "lower"

    def _opening_sentence(self, candidate: Candidate, tier: str) -> str:
        title = candidate.profile.current_title
        years = candidate.profile.years_of_experience

        top_skills = sorted(
            candidate.skills,
            key=lambda s: (s.proficiency == "expert", s.endorsements),
            reverse=True,
        )[: self.config.reasoner_top_skills_count]
        skill_phrase = self._skills_phrase(top_skills)

        years_phrase = f"{years:.1f} years" if years != int(years) else f"{int(years)} years"
        opener = (
            f"Strong fit: {title} with {years_phrase} of experience."
            if tier == "top"
            else f"{title} with {years_phrase} of experience."
        )

        if skill_phrase:
            opener += f" {skill_phrase}"

        return opener

    def _skills_phrase(self, skills: List[Skill]) -> str:
        if not skills:
            return ""
        expert_or_advanced = [s for s in skills if s.proficiency in ("expert", "advanced")]
        chosen = expert_or_advanced if expert_or_advanced else skills
        names = ", ".join(s.name for s in chosen[: self.config.reasoner_top_skills_count])
        verb = "Expert in" if expert_or_advanced else "Skilled in"
        return f"{verb} {names}."

    def _positive_clauses(self, candidate: Candidate) -> List[str]:
        sig = candidate.redrob_signals
        clauses: List[str] = []

        if sig.recruiter_response_rate > self.config.reasoner_high_response_rate_threshold:
            clauses.append(f"high recruiter response rate ({sig.recruiter_response_rate:.0%})")

        if sig.notice_period_days <= self.config.notice_period_ideal_days:
            clauses.append(f"{sig.notice_period_days}-day notice period")

        if sig.open_to_work_flag:
            clauses.append("open to work")

        location = candidate.profile.location.lower()
        if any(city in location for city in self.config.tier1_cities):
            clauses.append(f"based in {candidate.profile.location} (preferred location)")

        return clauses

    def _concern_clauses(self, candidate: Candidate, today: date) -> List[str]:
        sig = candidate.redrob_signals
        years = candidate.profile.years_of_experience
        clauses: List[str] = []

        if (
            years < self.config.experience_acceptable_min
            or years > self.config.experience_acceptable_max
        ):
            clauses.append(f"experience ({years:.1f} yrs) falls outside the typical range")

        if not sig.open_to_work_flag:
            clauses.append("not currently open to work")

        days_inactive = (today - sig.last_active_date).days
        if days_inactive > self.config.reasoner_low_activity_days_threshold:
            clauses.append(f"inactive for {days_inactive} days")

        if sig.notice_period_days > self.config.notice_period_max_days:
            clauses.append(f"{sig.notice_period_days}-day notice period may be a concern")

        return clauses

    def _join_clauses(self, clauses: List[str], tier: str, is_concern: bool = False) -> str:
        joined = "; ".join(clauses)
        if is_concern:
            return f"Note: {joined}." if tier == "lower" else f"{joined.capitalize()}."
        return f"{joined.capitalize()}."
