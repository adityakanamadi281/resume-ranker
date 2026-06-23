"""src/pipeline.py -- end-to-end ranking pipeline orchestration."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import AppConfig, config as default_config
from src.embedding.embedder import LocalEmbedder
from src.embedding.index import FaissCandidateIndex
from src.exceptions import PipelineError
from src.features import honeypot_detector, signal_processor, text_builder
from src.parsers.candidate_parser import stream_candidates
from src.parsers.docx_parser import parse_job_description
from src.ranking.aggregator import Top100Aggregator
from src.ranking.reasoner import RuleReasoner
from src.ranking.scorer import compute_composite_score, extract_jd_keywords
from src.schema import Candidate

logger = logging.getLogger(__name__)


class RankingPipeline:
    def __init__(self, config: AppConfig = default_config) -> None:
        self.config = config
        self.embedder = LocalEmbedder(config.embedding_model, config.embedding_device)
        self.aggregator = Top100Aggregator()
        self.reasoner = RuleReasoner()
        self._today = pd.Timestamp.now().date()

    # -------------------------------------------------------------
    # Step 2: stream candidates, skip honeypots, build texts
    # -------------------------------------------------------------
    def _stream_and_filter(
        self, candidates_path: Path
    ) -> Tuple[Dict[str, Candidate], List[str], List[str], int]:
        """Stream every candidate, run honeypot detection, and build
        embedding text for every survivor. Returns (candidates_by_id,
        candidate_ids_in_order, texts_in_order, n_honeypots)."""
        candidates_by_id: Dict[str, Candidate] = {}
        ids: List[str] = []
        texts: List[str] = []
        n_honeypots = 0
        n_total = 0

        for candidate in stream_candidates(candidates_path):
            n_total += 1
            is_honeypot, reasons = honeypot_detector.detect(candidate, self._today)
            if is_honeypot:
                n_honeypots += 1
                logger.debug("Honeypot skipped: %s (%s)", candidate.candidate_id, reasons)
                continue

            candidates_by_id[candidate.candidate_id] = candidate
            ids.append(candidate.candidate_id)
            texts.append(text_builder.build(candidate))

            if n_total % self.config.progress_log_interval == 0:
                logger.info(
                    "Processed %d candidates (%d valid, %d honeypots)",
                    n_total, len(ids), n_honeypots,
                )

        logger.info(
            "Streaming complete: %d total, %d valid, %d honeypots (%.2f%%)",
            n_total, len(ids), n_honeypots,
            100.0 * n_honeypots / n_total if n_total else 0.0,
        )
        return candidates_by_id, ids, texts, n_honeypots

    # -------------------------------------------------------------
    # Step 3+4: embed + build/load FAISS index, with artifact caching
    # -------------------------------------------------------------
    def _embed_and_index(
        self, ids: List[str], texts: List[str]
    ) -> Tuple[np.ndarray, FaissCandidateIndex]:
        artifacts_dir = self.config.artifacts_dir
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = artifacts_dir / "candidate_embeddings.npy"
        id_map_path = artifacts_dir / "id_map.json"
        index_path = artifacts_dir / "faiss.index"

        cached_ids = None
        if id_map_path.exists():
            with open(id_map_path, "r", encoding="utf-8") as f:
                cached_ids = json.load(f)

        if embeddings_path.exists() and cached_ids == ids:
            logger.info("Loading cached embeddings from %s", embeddings_path)
            embeddings = np.load(embeddings_path)
        else:
            logger.info("Encoding %d candidate texts (no valid cache found)...", len(texts))
            t0 = time.time()
            embeddings = self.embedder.encode(texts, batch_size=self.config.embedding_batch_size)
            logger.info("Encoded %d embeddings in %.1fs", len(texts), time.time() - t0)
            np.save(embeddings_path, embeddings)
            with open(id_map_path, "w", encoding="utf-8") as f:
                json.dump(ids, f)

        index = FaissCandidateIndex(dim=embeddings.shape[1])
        index.build(embeddings, ids)
        # FAISS index objects are not trivially reusable across a future
        # cache-validity check (ids may differ on a future run with a
        # different candidate file), so we always rebuild the in-memory
        # index from the embeddings rather than deserializing
        # artifacts/faiss.index -- the on-disk file is still written
        # below for inspection/reuse by external tooling.
        import faiss
        faiss.write_index(index.index, str(index_path))

        return embeddings, index

    # -------------------------------------------------------------
    # Full run
    # -------------------------------------------------------------
    def run(self, candidates_path: Path, jd_path: Path, output_path: Path) -> pd.DataFrame:
        t_start = time.time()

        logger.info("Step 1: Parsing job description from %s", jd_path)
        jd_text = parse_job_description(jd_path)
        logger.info("JD length: %d characters", len(jd_text))
        jd_keywords = extract_jd_keywords(jd_text)

        logger.info("Step 2: Streaming candidates and filtering honeypots from %s", candidates_path)
        candidates_by_id, ids, texts, n_honeypots = self._stream_and_filter(candidates_path)
        if not ids:
            raise PipelineError("No valid (non-honeypot) candidates found after filtering.")

        logger.info("Step 3+4: Embedding candidates and building FAISS index")
        embeddings, index = self._embed_and_index(ids, texts)

        logger.info("Step 5: Embedding JD and searching index")
        jd_embedding = self.embedder.encode_single(jd_text)
        search_results = index.search(jd_embedding, top_k=self.config.faiss_top_k)
        logger.info("Retrieved %d candidates from semantic search", len(search_results))

        logger.info("Step 6: Hybrid re-ranking")
        scored: List[Tuple[str, float]] = []
        for candidate_id, semantic_score in search_results:
            candidate = candidates_by_id[candidate_id]
            behavioral_features = signal_processor.process(candidate, self._today)
            result = compute_composite_score(
                candidate, semantic_score, jd_keywords, behavioral_features
            )
            scored.append((candidate_id, result["composite_score"]))

        logger.info("Step 7: Aggregating top %d", self.config.top_n)
        top_n = self.aggregator.aggregate(scored)

        logger.info("Step 8: Generating reasoning")
        rows = []
        for rank, (candidate_id, score) in enumerate(top_n, start=1):
            candidate = candidates_by_id[candidate_id]
            reasoning = self.reasoner.generate(candidate, rank, score, self._today)
            rows.append({
                "candidate_id": candidate_id,
                "rank": rank,
                "score": round(score, 4),
                "reasoning": reasoning,
            })

        df = pd.DataFrame(rows, columns=self.config.csv_columns)

        logger.info("Step 9: Validating and saving to %s", output_path)
        self._validate_dataframe(df)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        elapsed = time.time() - t_start
        logger.info(
            "Pipeline complete in %.1fs. %d candidates ranked, %d honeypots excluded (%.2f%%).",
            elapsed, len(df), n_honeypots,
            100.0 * n_honeypots / (len(ids) + n_honeypots) if (len(ids) + n_honeypots) else 0.0,
        )

        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        expected_n = min(self.config.top_n, df.shape[0])
        if len(df) != expected_n:
            raise PipelineError(f"Expected {expected_n} rows, got {len(df)}")
        if list(df.columns) != self.config.csv_columns:
            raise PipelineError(f"Expected columns {self.config.csv_columns}, got {list(df.columns)}")
        if df["candidate_id"].duplicated().any():
            raise PipelineError("Duplicate candidate_id values in final output.")
        if not (df["score"].diff().dropna() <= 0).all():
            raise PipelineError("Scores are not non-increasing by rank.")
        if list(df["rank"]) != list(range(1, len(df) + 1)):
            raise PipelineError("Ranks are not exactly 1..N in order.")
