"""End-to-end ranking application orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import orjson

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from resume_ranker.app.output_writer import SubmissionValidator, SubmissionWriter
from datetime import date

from resume_ranker.app.run_context import RunContext
from resume_ranker.config import AppConfig, config as default_config
from resume_ranker.domain import honeypot_rules
from resume_ranker.domain.explanations import RuleReasoner
from resume_ranker.domain.schema import Candidate
from resume_ranker.domain.scoring import (
    TopNAggregator,
    extract_jd_keywords,
    has_tech_career,
    _candidate_current_title_text,
)
from resume_ranker.evaluation.audit import build_audit_report, write_audit_report
from resume_ranker.exceptions import PipelineError
from resume_ranker.features import signal_features, text_builder
from resume_ranker.infrastructure.artifact_store import ArtifactManifest, ArtifactStore
from resume_ranker.infrastructure.candidate_reader import stream_candidates
from resume_ranker.infrastructure.embedder import LocalEmbedder
from resume_ranker.infrastructure.jd_reader import parse_job_description
from resume_ranker.infrastructure.vector_index import FaissCandidateIndex

logger = logging.getLogger(__name__)


FeatureBuildResult = Tuple[str, str, Dict[str, Any]]


def _build_candidate_features(
    candidate: Candidate,
    today: date,
    config: AppConfig,
) -> FeatureBuildResult:
    features = signal_features.process(candidate, today, config)
    candidate_text = text_builder.build(candidate, config)
    row: Dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "embedding_text": candidate_text,
        "years_of_experience": candidate.profile.years_of_experience,
        "location": candidate.profile.location,
        "country": candidate.profile.country,
        "current_title": candidate.profile.current_title,
        "current_title_text": _candidate_current_title_text(candidate),
        "skills_json": orjson.dumps([{"name": s.name, "duration_months": s.duration_months, "endorsements": s.endorsements} for s in candidate.skills]).decode(),
        "has_tech_career": has_tech_career(candidate, config),
        "candidate_json": candidate.model_dump_json(),
        **features,
    }
    return candidate.candidate_id, candidate_text, row


class RankingPipeline:
    def __init__(self, config: AppConfig = default_config) -> None:
        self.config = config
        self.embedder = LocalEmbedder(config)
        self.aggregator = TopNAggregator(config)
        self.reasoner = RuleReasoner(config)
        self.writer = SubmissionWriter(config)
        self.validator = SubmissionValidator(config)
        self.artifacts = ArtifactStore(config)
        self.last_audit_report: dict[str, Any] | None = None

    def _stream_and_filter(
        self,
        candidates_path: Path,
        context: RunContext,
    ) -> Tuple[Dict[str, Candidate], List[Candidate], int, int]:
        candidates_by_id: Dict[str, Candidate] = {}
        valid_candidates: List[Candidate] = []
        n_honeypots = 0
        n_total = 0

        for candidate in stream_candidates(candidates_path, self.config):
            n_total += 1
            is_honeypot, reasons = honeypot_rules.detect(candidate, context.today, self.config)
            if is_honeypot:
                n_honeypots += 1
                logger.debug("Honeypot skipped: %s (%s)", candidate.candidate_id, reasons)
                continue

            candidates_by_id[candidate.candidate_id] = candidate
            valid_candidates.append(candidate)

            if n_total % self.config.progress_log_interval == 0:
                logger.info(
                    "Processed %d candidates (%d valid, %d honeypots)",
                    n_total,
                    len(valid_candidates),
                    n_honeypots,
                )

        logger.info(
            "Streaming complete: %d total, %d valid, %d honeypots (%.2f%%)",
            n_total,
            len(valid_candidates),
            n_honeypots,
            100.0 * n_honeypots / n_total if n_total else 0.0,
        )
        return candidates_by_id, valid_candidates, n_honeypots, n_total

    def _build_feature_table(
        self,
        candidates: List[Candidate],
        context: RunContext,
    ) -> Tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_build_candidate_features)(candidate, context.today, self.config)
            for candidate in candidates
        )
        ids = np.asarray([row[0] for row in results], dtype=object)
        texts = np.asarray([row[1] for row in results], dtype=object)
        feature_rows = [row[2] for row in results]
        features = pl.DataFrame(feature_rows)
        self.artifacts.save_features(features)
        return ids, texts, features

    def _candidate_embeddings_manifest_for_cache(
        self,
        candidates_path: Path,
        jd_path: Optional[Path],
        ids: np.ndarray,
        total_candidates: int = 0,
        honeypots_excluded: int = 0,
    ) -> ArtifactManifest:
        cached = self.artifacts.load_manifest()
        cached_dim = cached.embedding_dim if cached is not None else 0
        return self.artifacts.expected_manifest(
            candidates_path=candidates_path,
            jd_path=jd_path,
            embedding_mode=self.embedder.embedding_mode,
            candidate_count=int(ids.shape[0]),
            embedding_dim=cached_dim,
            total_candidates=total_candidates,
            honeypots_excluded=honeypots_excluded,
        )

    def _embed_and_index(
        self,
        ids: np.ndarray,
        texts: np.ndarray,
        candidates_path: Path,
        jd_path: Optional[Path],
        total_candidates: int = 0,
        honeypots_excluded: int = 0,
    ) -> Tuple[np.ndarray, FaissCandidateIndex, ArtifactManifest]:
        self.artifacts.ensure_root()

        cache_manifest = self._candidate_embeddings_manifest_for_cache(
            candidates_path, jd_path, ids, total_candidates, honeypots_excluded
        )
        embeddings = self.artifacts.load_embeddings_if_valid(cache_manifest, ids)
        if embeddings is not None:
            logger.info("Loading cached embeddings from %s", self.artifacts.embeddings_path)
        else:
            logger.info("Encoding %d candidate texts with offline BGE model...", texts.shape[0])
            embeddings = self.embedder.encode(texts, batch_size=self.config.embedding_batch_size)

        final_manifest = self.artifacts.expected_manifest(
            candidates_path=candidates_path,
            jd_path=jd_path,
            embedding_mode=self.embedder.embedding_mode,
            candidate_count=int(ids.shape[0]),
            embedding_dim=int(embeddings.shape[1]),
            total_candidates=total_candidates,
            honeypots_excluded=honeypots_excluded,
        )
        self.artifacts.save_embeddings(embeddings, ids, final_manifest)

        index = FaissCandidateIndex(dim=int(embeddings.shape[1]))
        index.build(embeddings, ids)
        self.artifacts.save_index(index.index)
        return embeddings, index, final_manifest

    def precompute(self, candidates_path: Path, jd_path: Optional[Path] = None) -> None:
        context = RunContext(today=date.today())

        logger.info("Precompute Step 1: Streaming candidates and filtering honeypots from %s", candidates_path)
        _, valid_candidates, n_honeypots, n_total = self._stream_and_filter(
            candidates_path,
            context,
        )
        if not valid_candidates:
            raise PipelineError("No valid non-honeypot candidates found after filtering.")

        logger.info("Precompute Step 2: Building candidate feature table")
        ids, texts, _features = self._build_feature_table(valid_candidates, context)

        logger.info("Precompute Step 3: Embedding candidates and building FAISS index")
        self._embed_and_index(
            ids=ids,
            texts=texts,
            candidates_path=candidates_path,
            jd_path=jd_path,
            total_candidates=n_total,
            honeypots_excluded=n_honeypots,
        )

    def rank(self, jd_path: Path, output_path: Path) -> pl.DataFrame:
        context = RunContext(today=date.today())

        logger.info("Step 1: Parsing job description from %s", jd_path)
        with context.time_stage("parse_job_description"):
            jd_text = parse_job_description(jd_path)
            jd_keywords = extract_jd_keywords(jd_text)
        logger.info("JD length: %d characters", len(jd_text))

        with context.time_stage("load_artifacts"):
            # Check and load manifest.json
            logger.info("Step 2: Checking precomputation manifest")
            manifest = self.artifacts.load_manifest()
            if manifest is None:
                raise PipelineError("No precomputation manifest found. Run precompute first.")

            # Verify model name
            if manifest.model_name != self.config.embedding_model:
                raise PipelineError(
                    f"Configured embedding model '{self.config.embedding_model}' does not match "
                    f"precomputed model '{manifest.model_name}'."
                )

            # Load candidate features
            logger.info("Step 3: Loading precomputed candidate features")
            if not self.artifacts.features_path.exists():
                raise PipelineError(f"Candidate features file not found: {self.artifacts.features_path}")
            candidate_features = pl.read_parquet(self.artifacts.features_path)

            # Load FAISS index
            logger.info("Step 4: Loading FAISS index")
            if not self.artifacts.index_path.exists():
                raise PipelineError(f"FAISS index file not found: {self.artifacts.index_path}")

            import faiss
            faiss_index = faiss.read_index(str(self.artifacts.index_path))

            # Load candidate IDs mapped to index
            if not self.artifacts.candidate_ids_path.exists():
                raise PipelineError(f"Candidate IDs file not found: {self.artifacts.candidate_ids_path}")
            candidate_ids = np.load(self.artifacts.candidate_ids_path, allow_pickle=True)

        # Search top K candidates
        logger.info("Step 5: Embedding JD and searching FAISS index")
        with context.time_stage("embed_jd"):
            # Embed only the job description
            jd_embedding = self.embedder.encode_single(jd_text)

        with context.time_stage("faiss_search"):
            # Perform search using the raw faiss index
            query = np.ascontiguousarray(jd_embedding.reshape(1, -1), dtype=np.float32)
            top_k = min(self.config.faiss_top_k, len(candidate_ids))
            if top_k == 0:
                raise PipelineError("No candidates available in index.")

            scores, indices = faiss_index.search(query, top_k)
            valid = indices[0] >= 0
            search_ids = candidate_ids[indices[0][valid]]
            semantic_scores = scores[0][valid].astype(np.float32)

        logger.info("Retrieved %d candidates from semantic search", search_ids.shape[0])

        logger.info("Step 6: Hybrid re-ranking using Polars and NumPy")
        with context.time_stage("feature_scoring"):
            # Align features with retrieved search_ids
            search_df = pl.DataFrame({
                "candidate_id": search_ids.astype(str),
                "search_idx": np.arange(len(search_ids)),
                "semantic_score": semantic_scores,
            })
            top_features = candidate_features.join(search_df, on="candidate_id").sort("search_idx")

            # Vectorized behavioral score
            behavioral_scores = (
                self.config.behavioral_weight_open_to_work * top_features["open_to_work"].to_numpy()
                + self.config.behavioral_weight_notice_period * top_features["notice_period_score"].to_numpy()
                + self.config.behavioral_weight_response_rate * top_features["response_rate"].to_numpy()
                + self.config.behavioral_weight_experience_fit * top_features["experience_fit"].to_numpy()
                + self.config.behavioral_weight_location_fit * top_features["location_fit"].to_numpy()
                + self.config.behavioral_weight_profile_completeness * top_features["profile_completeness"].to_numpy()
                + self.config.behavioral_weight_verified * top_features["verified"].to_numpy()
                + self.config.behavioral_weight_recent_activity * top_features["recent_activity"].to_numpy()
            )
            behavioral_scores = np.clip(behavioral_scores, self.config.score_min, self.config.score_max).astype(np.float32)

            # Vectorized experience score
            years = top_features["years_of_experience"].to_numpy()
            experience_scores: np.ndarray = np.zeros(len(years), dtype=np.float32)
            ideal_mask = (years >= self.config.experience_ideal_min) & (years <= self.config.experience_ideal_max)
            experience_scores[ideal_mask] = self.config.score_max

            below_mask = (years >= self.config.experience_acceptable_min) & (years < self.config.experience_ideal_min)
            span_below = self.config.experience_ideal_min - self.config.experience_acceptable_min
            if span_below > 0:
                experience_scores[below_mask] = (years[below_mask] - self.config.experience_acceptable_min) / span_below

            above_mask = (years > self.config.experience_ideal_max) & (years <= self.config.experience_acceptable_max)
            span_above = self.config.experience_acceptable_max - self.config.experience_ideal_max
            if span_above > 0:
                experience_scores[above_mask] = self.config.score_max - (years[above_mask] - self.config.experience_ideal_max) / span_above

            # Location score
            location_scores = top_features["location_fit"].to_numpy().astype(np.float32)

            # Title and skills matching
            title_scores: np.ndarray = np.zeros(len(top_features), dtype=np.float32)
            skill_scores: np.ndarray = np.zeros(len(top_features), dtype=np.float32)

            current_titles = top_features["current_title_text"].to_list()
            skills_jsons = top_features["skills_json"].to_list()
            has_tech_careers = top_features["has_tech_career"].to_numpy()

            jd_keyword_text = " ".join(sorted(jd_keywords))

            from resume_ranker.domain.scoring import _WORD_PATTERN
            from rapidfuzz import fuzz

            for idx in range(len(top_features)):
                # Title score
                title_text = current_titles[idx]
                title_tokens = set(_WORD_PATTERN.findall(title_text))
                if not title_tokens:
                    title_score = self.config.title_score_no_match
                else:
                    overlap = title_tokens & jd_keywords
                    overlap_fraction = len(overlap) / len(title_tokens)
                    if overlap_fraction >= self.config.score_max:
                        title_score = self.config.title_score_exact_match
                    elif overlap_fraction > self.config.score_min:
                        title_score = self.config.title_score_partial_match
                    else:
                        title_score = self.config.title_score_no_match

                # Skill score
                skills = orjson.loads(skills_jsons[idx])
                if not skills:
                    skill_score = self.config.score_min
                else:
                    matches = 0.0
                    for skill in skills:
                        name = skill["name"].lower()
                        skill_tokens = set(_WORD_PATTERN.findall(name))
                        fuzzy_match = fuzz.partial_ratio(name, jd_keyword_text) >= 85
                        if skill_tokens & jd_keywords or fuzzy_match:
                            if skill["duration_months"] >= self.config.skill_min_duration_months or skill["endorsements"] > 0:
                                matches += 1.0

                    skill_score = matches / self.config.skill_match_denominator
                    if len(skills) > self.config.skill_stuffing_threshold:
                        penalty = 1.0 - min(
                            (len(skills) - self.config.skill_stuffing_threshold)
                            * self.config.skill_stuffing_penalty_per_skill,
                            self.config.skill_stuffing_max_penalty,
                        )
                        skill_score *= penalty
                    skill_score = min(skill_score, self.config.score_max)

                # Career alignment penalty
                if not has_tech_careers[idx]:
                    title_score = 0.0
                    skill_score = 0.0
                    exp_score = 0.0
                else:
                    exp_score = float(experience_scores[idx])

                title_scores[idx] = title_score
                skill_scores[idx] = skill_score
                experience_scores[idx] = exp_score

            # fully vectorized score combination
            scores_matrix = np.column_stack([
                top_features["semantic_score"].to_numpy().astype(np.float32),
                skill_scores,
                behavioral_scores,
                experience_scores,
                location_scores,
            ])

            weights = np.array([
                self.config.weight_semantic,
                self.config.weight_skills,
                self.config.weight_behavior,
                self.config.weight_experience,
                self.config.weight_location,
            ], dtype=np.float32)

            final_scores = scores_matrix @ weights
            final_scores = np.clip(final_scores, self.config.score_min, self.config.score_max).astype(np.float32)

            top_ids, top_scores = self.aggregator.aggregate_arrays(search_ids, final_scores)

        logger.info("Step 7: Generating explanations and writing output")
        with context.time_stage("explanation"):
            # build map from candidate_id to rank and score
            id_to_rank_score = {str(c_id): (rank, float(score)) for rank, (c_id, score) in enumerate(zip(top_ids, top_scores), start=1)}

            # filter top_features for top_ids to generate candidate objects
            final_top_features = top_features.filter(pl.col("candidate_id").is_in(top_ids.astype(str)))

            rows = []
            for row in final_top_features.to_dicts():
                c_id = row["candidate_id"]
                rank, score = id_to_rank_score[c_id]
                candidate = Candidate.model_validate_json(row["candidate_json"])
                reasoning = self.reasoner.generate(candidate, rank, score, context.today)
                rows.append({
                    "candidate_id": c_id,
                    "rank": rank,
                    "score": round(score, 4),
                    "reasoning": reasoning,
                })

            # Sort final rows to match final ranking order
            rows.sort(key=lambda r: r["rank"])
            df = pl.DataFrame(rows, schema=self.config.csv_columns)

        with context.time_stage("csv_write"):
            self.writer.write(df, output_path)

        validation_errors = self.validator.validate_csv(output_path)

        # Component means
        component_scores = []
        for idx in range(len(top_features)):
            component_scores.append({
                "semantic_score": float(top_features["semantic_score"][idx]),
                "title_score": float(title_scores[idx]),
                "skill_score": float(skill_scores[idx]),
                "behavioral_score": float(behavioral_scores[idx]),
                "experience_score": float(experience_scores[idx]),
                "location_score": float(location_scores[idx]),
            })

        audit_report = build_audit_report(
            total_candidates=manifest.total_candidates,
            valid_candidates=manifest.candidate_count,
            honeypots=manifest.honeypots_excluded,
            ranked_rows=df,
            stage_seconds=context.stage_seconds,
            manifest=manifest,
            validation_errors=validation_errors,
        )
        audit_report["feature_contribution_summary"] = self._component_means(component_scores)
        write_audit_report(audit_report, output_path.with_suffix(".audit.json"))
        self.last_audit_report = audit_report

        logger.info(
            "Ranking complete in %.1fs. %d candidates ranked.",
            context.elapsed_seconds(),
            df.height,
        )
        return df

    def run(self, candidates_path: Path, jd_path: Path, output_path: Path) -> pl.DataFrame:
        # Run precompute stage followed by ranking stage sequentially (backward compatibility)
        self.precompute(candidates_path, jd_path)
        return self.rank(jd_path, output_path)

    def _component_means(self, component_scores: List[dict[str, float]]) -> dict[str, float]:
        if not component_scores:
            return {}
        keys = component_scores[0].keys()
        return {
            key: round(sum(row[key] for row in component_scores) / len(component_scores), 4)
            for key in keys
        }
