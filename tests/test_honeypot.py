"""tests/test_honeypot.py -- unit tests for honeypot detection rules."""

from src.features.honeypot_detector import detect


def test_duration_mismatch_detected(candidate_factory, today):
    candidate = candidate_factory(
        career_history=[{
            "company": "Acme Corp",
            "title": "Engineer",
            "start_date": "2022-01-01",
            "end_date": "2022-03-01",  # 2 months
            "duration_months": 24,  # claimed 24 months -- way off
            "is_current": False,
            "industry": "Technology",
            "company_size": "201-500",
            "description": "",
        }]
    )
    is_honeypot, reasons = detect(candidate, today)
    assert is_honeypot
    assert any("duration mismatch" in r for r in reasons)


def test_expert_zero_experience_detected(candidate_factory, today):
    candidate = candidate_factory(
        skills=[{"name": "Rust", "proficiency": "expert", "endorsements": 0, "duration_months": 0}]
    )
    is_honeypot, reasons = detect(candidate, today)
    assert is_honeypot
    assert any("expert" in r and "0 months" in r for r in reasons)


def test_inverted_salary_detected(candidate_factory, today):
    candidate = candidate_factory(
        redrob_signals={"expected_salary_range_inr_lpa": {"min": 30.0, "max": 10.0}}
    )
    is_honeypot, reasons = detect(candidate, today)
    assert is_honeypot
    assert any("inverted salary" in r for r in reasons)


def test_job_before_education_detected(candidate_factory, today):
    candidate = candidate_factory(
        education=[{
            "institution": "Example University",
            "degree": "B.Tech",
            "field_of_study": "CS",
            "start_year": 2018,
            "end_year": 2022,
            "grade": None,
            "tier": "tier_1",
        }],
        career_history=[{
            "company": "Acme Corp",
            "title": "Engineer",
            "start_date": "2019-01-01",  # before education end_year 2022
            "end_date": None,
            "duration_months": 12,
            "is_current": True,
            "industry": "Technology",
            "company_size": "201-500",
            "description": "",
        }],
    )
    is_honeypot, reasons = detect(candidate, today)
    assert is_honeypot
    assert any("before earliest education" in r for r in reasons)


def test_high_completeness_low_connections_detected(candidate_factory, today):
    candidate = candidate_factory(
        redrob_signals={"profile_completeness_score": 99.0, "connection_count": 5}
    )
    is_honeypot, reasons = detect(candidate, today)
    assert is_honeypot
    assert any("connection_count" in r for r in reasons)


def test_impossible_timeline_detected(candidate_factory, today):
    candidate = candidate_factory(
        profile={"years_of_experience": 10.0},
        education=[{
            "institution": "Example University",
            "degree": "B.Tech",
            "field_of_study": "CS",
            "start_year": 2020,
            "end_year": 2024,  # graduated 2 years before `today` (2026)
            "grade": None,
            "tier": "tier_1",
        }],
    )
    is_honeypot, reasons = detect(candidate, today)
    assert is_honeypot
    assert any("years of stated experience exceeds" in r for r in reasons)


def test_valid_candidate_is_not_honeypot(sample_candidate, today):
    is_honeypot, reasons = detect(sample_candidate, today)
    assert not is_honeypot
    assert reasons == []


def test_multiple_rules_all_reasons_returned(candidate_factory, today):
    candidate = candidate_factory(
        redrob_signals={
            "expected_salary_range_inr_lpa": {"min": 30.0, "max": 10.0},
            "profile_completeness_score": 99.0,
            "connection_count": 3,
        },
        skills=[{"name": "Rust", "proficiency": "expert", "endorsements": 0, "duration_months": 0}],
    )
    is_honeypot, reasons = detect(candidate, today)
    assert is_honeypot
    assert len(reasons) >= 3
