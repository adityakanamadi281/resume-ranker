"""
src/schema.py -- Pydantic v2 models for candidate records.

Field names and constraints mirror data/input/candidate_schema.json
exactly (the JSON Schema is the source of truth for field shape; this
module is the runtime validation layer built from it). Per Rule 1, no
magic numbers appear inline here beyond the ones the schema itself
declares as field constraints (e.g. proficiency enum values) -- these
are structural/data-format facts, not tunable scoring parameters, so
they belong here rather than in config.py.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def load_schema_from_json(path: Path) -> Dict[str, Any]:
    """Read and return the JSON schema as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        payload: Dict[str, Any] = json.load(f)
        return payload


class Profile(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float = Field(ge=0, le=50)
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


class CareerEntry(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    company: str
    title: str
    start_date: date
    end_date: Optional[date] = None
    duration_months: int = Field(ge=0)
    is_current: bool
    industry: str
    company_size: str
    description: str = ""


class Education(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    institution: str
    degree: str
    field_of_study: str
    start_year: int = Field(ge=1970, le=2030)
    end_year: int = Field(ge=1970, le=2035)
    grade: Optional[str] = None
    tier: Optional[str] = None


class Skill(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    name: str
    proficiency: str = Field(pattern="^(beginner|intermediate|advanced|expert)$")
    endorsements: int = Field(ge=0)
    duration_months: int = Field(default=0, ge=0)


class Certification(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    name: str
    issuer: str
    year: int


class Language(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    language: str
    proficiency: str = Field(pattern="^(basic|conversational|professional|native)$")


class SalaryRange(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    min: float = Field(ge=0)
    max: float = Field(ge=0)


class RedrobSignals(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    profile_completeness_score: float = Field(ge=0, le=100)
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(ge=0)
    applications_submitted_30d: int = Field(ge=0)
    recruiter_response_rate: float = Field(ge=0, le=1)
    avg_response_time_hours: float = Field(ge=0)
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
    connection_count: int = Field(ge=0)
    endorsements_received: int = Field(ge=0)
    notice_period_days: int = Field(ge=0, le=180)
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float = Field(ge=-1, le=100)
    search_appearance_30d: int = Field(ge=0)
    saved_by_recruiters_30d: int = Field(ge=0)
    interview_completion_rate: float = Field(ge=0, le=1)
    offer_acceptance_rate: float = Field(ge=-1, le=1)
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


class Candidate(BaseModel):
    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    candidate_id: str = Field(pattern=r"^CAND_[0-9]{7}$")
    profile: Profile
    career_history: List[CareerEntry] = Field(min_length=1, max_length=10)
    education: List[Education] = Field(default_factory=list, max_length=5)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals
