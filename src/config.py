"""
src/config.py -- THE ONLY PLACE FOR CONSTANTS.

This is the single source of truth for every tunable parameter in the
resume ranking system. Per the project's Rule 1 (ZERO HARDCODED VALUES),
no other file in this codebase may contain a literal number, string, or
list that represents a tunable judgment -- every such value lives here,
as a named field on AppConfig, and is referenced everywhere else as
config.xxx.

Values can be overridden at runtime via environment variables prefixed
with RANKER_ (e.g. RANKER_WEIGHT_SEMANTIC=0.4) or via a .env file,
through pydantic-settings' BaseSettings.
"""

from pathlib import Path
from typing import List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RANKER_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    artifacts_dir: Path = Path("artifacts")

    # ---------------------------------------------------------------
    # Model config
    # ---------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    # Dimension used by the TF-IDF fallback embedder when
    # sentence-transformers / the model weights are unavailable (e.g. no
    # network access to download the model). See src/embedding/embedder.py.
    # Kept modest (vs. the primary model's 384) because the fallback's
    # TruncatedSVD fit cost scales with this value, and it must complete
    # well inside the 5-minute / CPU-only / 16GB budget at 100K records.
    embedding_dim_fallback: int = 128
    embedding_random_seed: int = 42
    fallback_tfidf_max_features: int = 5000
    fallback_tfidf_min_df: int = 3
    fallback_tfidf_ngram_max: int = 1
    fallback_svd_fit_sample_size: int = 5000
    fallback_svd_n_iter: int = 5

    # ---------------------------------------------------------------
    # FAISS config
    # ---------------------------------------------------------------
    faiss_nlist: int = 100
    faiss_nprobe: int = 10
    faiss_top_k: int = 500
    faiss_train_sample_size: int = 10_000
    # FAISS's own documented rule of thumb for stable IVF cluster
    # training: at least ~39x nlist training points. Below this
    # threshold we fall back to an exact flat index (see
    # src/embedding/index.py) rather than force an under-trained,
    # lower-quality IVF index.
    faiss_ivf_training_multiplier: int = 39

    # ---------------------------------------------------------------
    # Scoring weights (must sum to 1.0)
    # ---------------------------------------------------------------
    weight_semantic: float = 0.35
    weight_title: float = 0.15
    weight_skill: float = 0.20
    weight_behavioral: float = 0.20
    weight_experience: float = 0.10

    # ---------------------------------------------------------------
    # Behavioral sub-weights (used inside signal_processor's aggregate
    # behavioral score; must also sum to 1.0)
    # ---------------------------------------------------------------
    behavioral_weight_open_to_work: float = 0.15
    behavioral_weight_notice_period: float = 0.15
    behavioral_weight_response_rate: float = 0.15
    behavioral_weight_experience_fit: float = 0.15
    behavioral_weight_location_fit: float = 0.10
    behavioral_weight_profile_completeness: float = 0.10
    behavioral_weight_verified: float = 0.10
    behavioral_weight_recent_activity: float = 0.10

    # ---------------------------------------------------------------
    # Title-match scores
    # ---------------------------------------------------------------
    title_score_exact_match: float = 1.0
    title_score_partial_match: float = 0.7
    title_score_no_match: float = 0.3

    # ---------------------------------------------------------------
    # Signal thresholds
    # ---------------------------------------------------------------
    notice_period_ideal_days: int = 30
    notice_period_max_days: int = 90
    response_time_ideal_hours: int = 24
    response_time_score_floor: float = 0.3
    profile_views_normalization: int = 1000
    saved_by_recruiters_normalization: int = 50
    skill_assessment_max_score: float = 100.0
    github_activity_max_score: float = 100.0
    github_activity_unlinked_sentinel: float = -1.0
    offer_acceptance_unlinked_sentinel: float = -1.0

    # ---------------------------------------------------------------
    # Recent-activity recency buckets (days since last_active_date)
    # ---------------------------------------------------------------
    recent_activity_days_tier1: int = 7
    recent_activity_days_tier2: int = 30
    recent_activity_days_tier3: int = 90
    recent_activity_score_tier1: float = 1.0
    recent_activity_score_tier2: float = 0.8
    recent_activity_score_tier3: float = 0.5
    recent_activity_score_tier4: float = 0.2

    # ---------------------------------------------------------------
    # Experience range
    # ---------------------------------------------------------------
    experience_ideal_min: float = 3.0
    experience_ideal_max: float = 8.0
    experience_acceptable_min: float = 1.0
    experience_acceptable_max: float = 15.0

    # ---------------------------------------------------------------
    # Location scoring
    # ---------------------------------------------------------------
    tier1_cities: List[str] = Field(
        default_factory=lambda: [
            "pune", "noida", "delhi", "bangalore", "hyderabad",
            "chennai", "mumbai", "gurgaon", "kolkata",
        ]
    )
    location_score_tier1: float = 1.0
    location_score_india_other: float = 0.7
    location_score_abroad: float = 0.4
    location_country_india_name: str = "india"

    # ---------------------------------------------------------------
    # Honeypot detection
    # ---------------------------------------------------------------
    honeypot_duration_tolerance_months: int = 3
    honeypot_high_completeness_threshold: float = 0.95
    honeypot_low_connections_threshold: int = 10
    # Disqualification threshold: >10% honeypots in top-N is a hard
    # constraint failure (see scripts/validate.py and src/pipeline.py).
    honeypot_max_top_n_share: float = 0.10

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------
    top_n: int = 100
    progress_log_interval: int = 10_000
    csv_columns: List[str] = Field(
        default_factory=lambda: ["candidate_id", "rank", "score", "reasoning"]
    )

    # ---------------------------------------------------------------
    # Reasoning
    # ---------------------------------------------------------------
    reasoner_max_length: int = 200
    reasoner_top_skills_count: int = 5
    reasoner_rank_thresholds: List[int] = Field(default_factory=lambda: [10, 50])
    reasoner_high_response_rate_threshold: float = 0.8
    reasoner_low_activity_days_threshold: int = 90

    # ---------------------------------------------------------------
    # Score constants
    # ---------------------------------------------------------------
    score_max: float = 1.0
    score_min: float = 0.0
    score_open_to_work_true: float = 1.0
    score_open_to_work_false: float = 0.3
    score_willing_to_relocate_true: float = 1.0
    score_willing_to_relocate_false: float = 0.5
    skill_match_denominator: int = 20

    # ---------------------------------------------------------------
    # Text builder
    # ---------------------------------------------------------------
    text_builder_career_repeat_count: int = 2
    text_builder_max_description_chars: int = 2000
    text_builder_separator: str = " | "

    # ---------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------
    @field_validator("weight_semantic", "weight_title", "weight_skill",
                      "weight_behavioral", "weight_experience")
    @classmethod
    def _weights_in_unit_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"weight must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_weight_sums(self) -> "AppConfig":
        primary_sum = (
            self.weight_semantic + self.weight_title + self.weight_skill
            + self.weight_behavioral + self.weight_experience
        )
        if abs(primary_sum - 1.0) > 0.01:
            raise ValueError(
                f"Primary scoring weights must sum to ~1.0, got {primary_sum:.4f}"
            )

        behavioral_sum = (
            self.behavioral_weight_open_to_work + self.behavioral_weight_notice_period
            + self.behavioral_weight_response_rate + self.behavioral_weight_experience_fit
            + self.behavioral_weight_location_fit + self.behavioral_weight_profile_completeness
            + self.behavioral_weight_verified + self.behavioral_weight_recent_activity
        )
        if abs(behavioral_sum - 1.0) > 0.01:
            raise ValueError(
                f"Behavioral sub-weights must sum to ~1.0, got {behavioral_sum:.4f}"
            )
        return self

    @model_validator(mode="after")
    def _validate_experience_ranges(self) -> "AppConfig":
        if not (self.experience_acceptable_min <= self.experience_ideal_min
                <= self.experience_ideal_max <= self.experience_acceptable_max):
            raise ValueError(
                "Experience ranges must satisfy: acceptable_min <= ideal_min "
                "<= ideal_max <= acceptable_max"
            )
        return self

    @field_validator("input_dir", "output_dir", "artifacts_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return v if isinstance(v, Path) else Path(str(v))


# Global singleton -- every other module imports this instance.
config = AppConfig()
