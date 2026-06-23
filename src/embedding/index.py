"""src/embedding/faiss_index.py -- FAISS IVF index build and search, CPU-only.

ALL index parameters come from config. Vectors are assumed already
L2-normalized by the embedder, so inner-product search is equivalent to
cosine similarity.
"""

import logging
from typing import List, Tuple

import numpy as np
import faiss

from src.config import config

logger = logging.getLogger(__name__)


class FaissCandidateIndex:
    """Wraps a FAISS IndexIVFFlat (inner-product metric) over candidate
    embeddings, with the candidate_id list kept alongside in the same
    order as the rows added to the index."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, config.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = config.faiss_nprobe
        self.candidate_ids: List[str] = []
        self._is_trained = False

    def build(self, embeddings: np.ndarray, candidate_ids: List[str]) -> None:
        """Train the IVF quantizer on a random sample (capped at
        config.faiss_train_sample_size, the same sampling principle the
        embedding fallback's SVD step uses) and add all embeddings to
        the index."""
        if embeddings.shape[0] != len(candidate_ids):
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but "
                f"{len(candidate_ids)} candidate_ids were given"
            )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        n = embeddings.shape[0]
        train_n = min(config.faiss_train_sample_size, n)
        # FAISS's own guidance for IVF training is roughly 39x nlist
        # training points minimum for stable cluster assignment (below
        # this it emits its own internal warning and quality degrades).
        # We treat that as the threshold for falling back to an exact
        # flat index instead of forcing an under-trained IVF index.
        min_recommended_training_points = config.faiss_nlist * config.faiss_ivf_training_multiplier
        if train_n < min_recommended_training_points:
            # Not enough vectors to train the requested number of IVF
            # cells -- fall back to a flat (exhaustive) index instead of
            # crashing, since FAISS requires at least nlist training
            # points. This only triggers on small/test inputs.
            logger.warning(
                "Only %d training vectors available for %d IVF cells; "
                "using a flat (exact search) index instead.",
                train_n, config.faiss_nlist,
            )
            self.index = faiss.IndexFlatIP(self.dim)
            self._is_trained = True
        else:
            rng = np.random.RandomState(config.embedding_random_seed)
            train_idx = rng.choice(n, size=train_n, replace=False)
            self.index.train(embeddings[train_idx])
            self._is_trained = True

        self.index.add(embeddings)
        self.candidate_ids = list(candidate_ids)
        logger.info("FAISS index built: %d vectors, dim=%d", n, self.dim)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Return up to top_k (candidate_id, similarity) pairs, sorted by
        descending similarity."""
        if not self._is_trained:
            raise RuntimeError("FaissCandidateIndex.build() must be called before search().")

        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        k = min(top_k, len(self.candidate_ids))
        if k == 0:
            return []

        scores, indices = self.index.search(query, k)
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.candidate_ids[idx], float(score)))
        results.sort(key=lambda pair: pair[1], reverse=True)
        return results
