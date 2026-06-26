"""tests/test_scorer.py -- unit tests for scoring."""

from resume_ranker.config import config
from resume_ranker.domain.scoring import (
    TopNAggregator,
    compute_behavioral_score,
    compute_composite_score,
    compute_experience_score,
    compute_skill_score,
    compute_title_score,
    extract_jd_keywords,
)
from resume_ranker.features import signal_features


def test_title_score_exact_match(candidate_factory):
    candidate = candidate_factory(profile={"headline": "ai engineer", "current_title": "AI Engineer"})
    keywords = extract_jd_keywords("Senior AI Engineer required with experience")
    score = compute_title_score(candidate, keywords, config)
    assert score == config.title_score_exact_match


def test_title_score_partial_match(candidate_factory):
    candidate = candidate_factory(profile={"headline": "ai engineer backend specialist"})
    keywords = extract_jd_keywords("We are hiring an AI Engineer")
    score = compute_title_score(candidate, keywords, config)
    assert score == config.title_score_partial_match


def test_title_score_no_match(candidate_factory):
    # Override both headline AND the career_history entry's title so no AI/engineer tokens bleed in
    candidate = candidate_factory(
        profile={"headline": "zzz qqq xxx", "current_title": "zzz role"},
        career_history=[{
            "company": "Acme Corp",
            "title": "zzz role",   # does not contain any JD keywords
            "start_date": "2024-06-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": "Technology",
            "company_size": "201-500",
            "description": "",
        }],
    )
    keywords = extract_jd_keywords("We are hiring an AI Engineer")
    score = compute_title_score(candidate, keywords, config)
    assert score == config.title_score_no_match


def test_skill_score_zero_matches(candidate_factory):
    candidate = candidate_factory(
        skills=[{"name": "Zzz", "proficiency": "expert", "endorsements": 0, "duration_months": 12}]
    )
    keywords = extract_jd_keywords("Looking for Python and FAISS experience")
    score = compute_skill_score(candidate, keywords, config)
    assert score == 0.0


def test_skill_score_some_matches(candidate_factory):
    candidate = candidate_factory(
        skills=[
            {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 12},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 0, "duration_months": 12},
        ]
    )
    keywords = extract_jd_keywords("Looking for Python and FAISS experience")
    score = compute_skill_score(candidate, keywords, config)
    assert 0.0 < score < 1.0
    assert score == min(2 / config.skill_match_denominator, 1.0)


def test_skill_score_max_clamped_to_one(candidate_factory):
    many_skills = [
        {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 12}
        for _ in range(config.skill_match_denominator * 2)
    ]
    candidate = candidate_factory(skills=many_skills)
    keywords = extract_jd_keywords("Python required")
    assert compute_skill_score(candidate, keywords, config) == 1.0


def test_behavioral_score_all_max():
    features = {
        "open_to_work": 1.0, "notice_period_score": 1.0, "response_rate": 1.0,
        "experience_fit": 1.0, "location_fit": 1.0, "profile_completeness": 1.0,
        "verified": 1.0, "recent_activity": 1.0,
    }
    import pytest
    assert compute_behavioral_score(features, config) == pytest.approx(1.0, abs=1e-9)


def test_behavioral_score_all_min():
    features = {
        "open_to_work": 0.0, "notice_period_score": 0.0, "response_rate": 0.0,
        "experience_fit": 0.0, "location_fit": 0.0, "profile_completeness": 0.0,
        "verified": 0.0, "recent_activity": 0.0,
    }
    assert compute_behavioral_score(features, config) == 0.0


def test_behavioral_score_mixed_in_bounds():
    features = {
        "open_to_work": 1.0, "notice_period_score": 0.5, "response_rate": 0.3,
        "experience_fit": 0.8, "location_fit": 0.6, "profile_completeness": 0.9,
        "verified": 0.4, "recent_activity": 0.7,
    }
    score = compute_behavioral_score(features, config)
    assert 0.0 <= score <= 1.0


def test_experience_score_ideal_range(candidate_factory):
    midpoint = (config.experience_ideal_min + config.experience_ideal_max) / 2
    candidate = candidate_factory(profile={"years_of_experience": midpoint})
    assert compute_experience_score(candidate, config) == 1.0


def test_experience_score_acceptable_range_interpolates(candidate_factory):
    below_ideal = (config.experience_acceptable_min + config.experience_ideal_min) / 2
    candidate = candidate_factory(profile={"years_of_experience": below_ideal})
    score = compute_experience_score(candidate, config)
    assert 0.0 <= score <= 1.0


def test_experience_score_out_of_range_zero(candidate_factory):
    candidate = candidate_factory(
        profile={"years_of_experience": config.experience_acceptable_max + 10}
    )
    assert compute_experience_score(candidate, config) == 0.0


def test_composite_score_weights_sum_to_one():
    total = (
        config.weight_semantic + config.weight_skills + config.weight_behavior + config.weight_experience + config.weight_location
    )
    assert abs(total - 1.0) < 0.01


def test_composite_score_equals_weighted_sum(sample_candidate, today):
    keywords = extract_jd_keywords("Senior AI Engineer Python FAISS")
    features = signal_features.process(sample_candidate, today, config)
    result = compute_composite_score(sample_candidate, 0.6, keywords, features, config)
    expected = max(0.0, min(1.0,
        config.weight_semantic * result["semantic_score"]
        + config.weight_skills * result["skill_score"]
        + config.weight_behavior * result["behavioral_score"]
        + config.weight_experience * result["experience_score"]
        + config.weight_location * result["location_score"]
    ))
    assert abs(result["composite_score"] - expected) < 1e-6


def test_all_component_scores_in_bounds(sample_candidate, today):
    keywords = extract_jd_keywords("Senior AI Engineer Python FAISS")
    features = signal_features.process(sample_candidate, today, config)
    result = compute_composite_score(sample_candidate, 0.6, keywords, features, config)
    for key in ("semantic_score", "title_score", "skill_score",
                "behavioral_score", "experience_score", "composite_score"):
        assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of bounds"


def test_tech_career_alignment_penalty(candidate_factory, today):
    # Candidate who has never held a tech-related role (e.g. Operations Manager)
    candidate = candidate_factory(
        profile={"headline": "Operations Manager", "current_title": "Operations Manager"},
        career_history=[{
            "company": "Pied Piper",
            "title": "Operations Manager",
            "start_date": "2024-06-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": "Technology",
            "company_size": "201-500",
            "description": "Managed operations.",
        }]
    )
    keywords = extract_jd_keywords("Senior AI Engineer Python FAISS")
    features = signal_features.process(candidate, today, config)
    result = compute_composite_score(candidate, 0.6, keywords, features, config)
    # Title, skill, and experience scores should be zeroed out
    assert result["title_score"] == 0.0
    assert result["skill_score"] == 0.0
    assert result["experience_score"] == 0.0
    # Composite score should only contain semantic + behavioral components
    expected = max(0.0, min(1.0,
        config.weight_semantic * result["semantic_score"]
        + config.weight_behavior * result["behavioral_score"] + config.weight_location * result["location_score"]
    ))
    assert abs(result["composite_score"] - expected) < 1e-6


def test_trust_weighted_skill_scoring_and_stuffing_penalty(candidate_factory):
    # Candidate with more than 15 skills should get stuffing penalty
    many_skills = [
        {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 0, "duration_months": 3}
        for i in range(20)
    ]
    candidate = candidate_factory(skills=many_skills)
    keywords = extract_jd_keywords("Skill0 Skill1 Skill2 Skill3")
    score = compute_skill_score(candidate, keywords, config)
    # Matching skills with duration < 6 and 0 endorsements should score 0.0
    assert score == 0.0

    # Let's verify with high trust skills
    trusted_skills = [
        {"name": f"Skill{i}", "proficiency": "advanced", "endorsements": 5, "duration_months": 12}
        for i in range(20)
    ]
    candidate = candidate_factory(skills=trusted_skills)
    score = compute_skill_score(candidate, keywords, config)
    # They should have got some score but diluted by stuffing penalty (> 15 skills)
    assert 0.0 < score < 1.0


def test_tie_break_by_candidate_id_ascending():
    aggregator = TopNAggregator(config)
    scored = [("CAND_0000002", 0.5), ("CAND_0000001", 0.5)]
    result = aggregator.aggregate(scored)
    assert result[0][0] == "CAND_0000001"
    assert result[1][0] == "CAND_0000002"
