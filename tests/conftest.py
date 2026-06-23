"""tests/conftest.py -- shared pytest fixtures."""

from datetime import date
from typing import Any, Dict

import pytest

from src.schema import Candidate


def _base_candidate_dict(**overrides: Any) -> Dict[str, Any]:
    """Build a minimal, schema-valid candidate dict. Callers pass whole
    sub-dict overrides, e.g. profile={"years_of_experience": 20}, which
    get shallow-merged into the matching top-level section."""
    base: Dict[str, Any] = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Candidate One",
            "headline": "Senior AI Engineer",
            "summary": "Builds ranking and retrieval systems.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "Senior AI Engineer",
            "current_company": "Acme Corp",
            "current_company_size": "201-500",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Senior AI Engineer",
                "start_date": "2024-06-01",   # ~24 months before today (2026-06-21)
                "end_date": None,
                "duration_months": 24,
                "is_current": True,
                "industry": "Technology",
                "company_size": "201-500",
                "description": "Built retrieval systems.",
            }
        ],
        "education": [
            {
                "institution": "Example University",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2016,
                "end_year": 2020,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 36},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 5, "duration_months": 18},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "native"}],
        "redrob_signals": {
            "profile_completeness_score": 75.0,
            "signup_date": "2023-01-01",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 50,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.5,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {"Python": 80.0},
            "connection_count": 200,
            "endorsements_received": 15,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 15.0, "max": 25.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50.0,
            "search_appearance_30d": 20,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.7,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value

    return base


@pytest.fixture
def candidate_factory():
    """Returns a function that builds a valid Candidate, with overrides."""

    def _make(**overrides: Any) -> Candidate:
        return Candidate.model_validate(_base_candidate_dict(**overrides))

    return _make


@pytest.fixture
def candidate_dict_factory():
    """Returns a function that builds a valid raw candidate dict (not yet
    validated), for tests that need to inspect/mutate before validation."""
    return _base_candidate_dict


@pytest.fixture
def sample_candidate(candidate_factory) -> Candidate:
    return candidate_factory()


@pytest.fixture
def today() -> date:
    return date(2026, 6, 21)
