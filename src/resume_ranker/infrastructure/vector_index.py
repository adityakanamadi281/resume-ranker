"""FAISS IndexFlatIP vector search, CPU-only."""

from __future__ import annotations

import logging
from typing import Any, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissCandidateIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index: Any = faiss.IndexFlatIP(dim)
        self.candidate_ids: np.ndarray = np.empty((0,), dtype=object)

    def build(self, embeddings: np.ndarray, candidate_ids: np.ndarray) -> None:
        if embeddings.shape[0] != candidate_ids.shape[0]:
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but "
                f"{candidate_ids.shape[0]} candidate_ids were given"
            )
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.candidate_ids = candidate_ids.astype(object, copy=True)
        logger.info("FAISS IndexFlatIP built: %d vectors, dim=%d", vectors.shape[0], self.dim)

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        k = min(top_k, self.candidate_ids.shape[0])
        if k == 0:
            return np.empty((0,), dtype=object), np.empty((0,), dtype=np.float32)

        scores, indices = self.index.search(query, k)
        valid = indices[0] >= 0
        result_ids = self.candidate_ids[indices[0][valid]]
        result_scores = scores[0][valid].astype(np.float32)
        return result_ids, result_scores
