"""
src/embedding/embedder.py -- local CPU-only text embedding.

PRIMARY PATH (per spec): sentence-transformers, fully local, CPU-only,
no API calls during ranking. The model is downloaded once from
HuggingFace the first time it is used (this requires network access at
that moment, exactly like `uv sync` requires network access to fetch
packages from PyPI -- it is a one-time setup dependency, not a
per-candidate API call during ranking. Once cached locally by
sentence-transformers, no further network access is needed).

FALLBACK PATH: if the sentence-transformers model cannot be loaded (no
network access to HuggingFace, or the package/model is unavailable in a
given sandbox), this module transparently falls back to a TF-IDF based
embedder built on scikit-learn, which needs no model download at all and
runs from PyPI-installed packages only. This keeps the pipeline runnable
in network-restricted CPU sandboxes (Rule 4: CPU only; Rule 3: no network
calls DURING ranking) while still using the spec's intended model when
it is available. Every text still gets embedded into a fixed-size,
L2-normalized vector either way, so the rest of the pipeline (FAISS
indexing, cosine-similarity scoring) is identical regardless of which
path was used. Which path was actually used is logged clearly so this is
never silent.
"""

import logging
from typing import List, Optional

import numpy as np

from src.config import config

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """Local CPU-only embedder. Tries sentence-transformers first (per
    spec); falls back to a TF-IDF + truncated-SVD embedder if the model
    cannot be loaded, so the pipeline still runs end-to-end without
    network access to a model hub."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._st_model = None
        self._fallback_vectorizer = None
        self._fallback_svd = None
        self._using_fallback = False
        self._fit_corpus_cache: Optional[List[str]] = None

        self._try_load_sentence_transformer()

    def _try_load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model '%s' on %s...",
                        self.model_name, self.device)
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Loaded sentence-transformers model successfully.")
        except Exception as e:  # noqa: BLE001 -- any load failure triggers the fallback
            logger.warning(
                "Could not load sentence-transformers model '%s' (%s). "
                "Falling back to local TF-IDF embedding (no model download required).",
                self.model_name, e,
            )
            self._st_model = None
            self._using_fallback = True

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def _fit_fallback(self, reference_texts: List[str]) -> None:
        """The TF-IDF fallback needs a fitted vocabulary. It is fit once,
        lazily, on the first batch of texts passed to encode() (which in
        this pipeline's usage is always the full candidate corpus plus
        the JD -- see src/pipeline.py), then reused for all subsequent
        calls via encode_single() so the JD and every candidate share
        the same vector space.

        The SVD step is fit on a random sample of
        config.fallback_svd_fit_sample_size documents rather than the
        full corpus (the same sampling principle config.faiss_nlist
        training already uses for the FAISS IVF quantizer) because SVD
        fit cost scales with corpus size and this keeps the whole
        embedding stage well inside the runtime budget at 100K records;
        every document is still transformed into the fitted space
        afterward, so no document is excluded from the output, only from
        the (computationally expensive) basis-fitting step."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._fallback_vectorizer = TfidfVectorizer(
            max_features=config.fallback_tfidf_max_features,
            ngram_range=(1, config.fallback_tfidf_ngram_max),
            min_df=config.fallback_tfidf_min_df,
            stop_words="english",
            sublinear_tf=True,
        )
        tfidf = self._fallback_vectorizer.fit_transform(reference_texts)

        n_components = min(
            config.embedding_dim_fallback, tfidf.shape[1] - 1, tfidf.shape[0] - 1
        )
        n_components = max(n_components, 1)

        rng = np.random.RandomState(config.embedding_random_seed)
        sample_size = min(config.fallback_svd_fit_sample_size, tfidf.shape[0])
        sample_idx = rng.choice(tfidf.shape[0], size=sample_size, replace=False)

        self._fallback_svd = TruncatedSVD(
            n_components=n_components,
            random_state=config.embedding_random_seed,
            algorithm="randomized",
            n_iter=config.fallback_svd_n_iter,
        )
        self._fallback_svd.fit(tfidf[sample_idx])
        logger.info(
            "Fitted TF-IDF fallback embedder: vocab=%d, dim=%d, svd_fit_sample=%d",
            len(self._fallback_vectorizer.vocabulary_), n_components, sample_size,
        )

    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        if self._fallback_vectorizer is None:
            self._fit_fallback(texts)
        tfidf = self._fallback_vectorizer.transform(texts)
        dense = self._fallback_svd.transform(tfidf)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (dense / norms).astype(np.float32)

    def encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode a batch of texts. Returns L2-normalized embeddings,
        shape (len(texts), dim)."""
        if not texts:
            dim = config.embedding_dim_fallback
            return np.zeros((0, dim), dtype=np.float32)

        safe_texts = [t if t.strip() else " " for t in texts]

        if self._st_model is not None:
            embeddings = self._st_model.encode(
                safe_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(safe_texts) > batch_size,
            )
            return np.asarray(embeddings, dtype=np.float32)

        return self._encode_fallback(safe_texts)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text. Returns a single L2-normalized embedding
        vector."""
        result = self.encode([text], batch_size=1)
        return result[0]
