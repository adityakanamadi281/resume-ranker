"""Build rich embedding text from a candidate profile."""

import re

from resume_ranker.config import AppConfig
from resume_ranker.domain.schema import Candidate, CareerEntry, Education, Skill

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _clean(text: str, config: AppConfig) -> str:
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()
    if len(text) > config.text_builder_max_description_chars:
        text = text[: config.text_builder_max_description_chars]
    return text


def _career_entry_text(entry: CareerEntry, config: AppConfig) -> str:
    return _clean(
        f"{entry.company}: {entry.title} ({entry.duration_months} months) - {entry.description}",
        config,
    )


def _skill_text(skill: Skill, config: AppConfig) -> str:
    return _clean(f"{skill.name} ({skill.proficiency}, {skill.duration_months} months)", config)


def _education_text(edu: Education, config: AppConfig) -> str:
    return _clean(f"{edu.degree} in {edu.field_of_study} from {edu.institution}", config)


def build(candidate: Candidate, config: AppConfig) -> str:
    """Build a single rich text blob for embedding, weighting career
    history most heavily (repeated config.text_builder_career_repeat_count
    times), with profile headline/title leading and skills/education
    appearing once each."""
    sep = config.text_builder_separator

    profile_part = sep.join(_clean(t, config) for t in (
        candidate.profile.current_title,
        candidate.profile.headline,
        candidate.profile.summary,
    ) if t)

    career_texts = [_career_entry_text(e, config) for e in candidate.career_history]
    career_part = sep.join(career_texts)

    skills_part = sep.join(_skill_text(s, config) for s in candidate.skills)
    education_part = sep.join(_education_text(e, config) for e in candidate.education)

    parts = [profile_part]
    for _ in range(config.text_builder_career_repeat_count):
        if career_part:
            parts.append(career_part)
    if skills_part:
        parts.append(skills_part)
    if education_part:
        parts.append(education_part)

    return sep.join(p for p in parts if p)
