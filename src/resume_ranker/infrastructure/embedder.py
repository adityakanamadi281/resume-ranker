"""Offline-only sentence-transformer embedding generation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from resume_ranker.config import AppConfig
from resume_ranker.exceptions import PipelineError

logger = logging.getLogger(__name__)


class LocalEmbedder:
    """BGE sentence-transformer embedder with no fallback and no downloads."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model_name = config.embedding_model
        self.device = config.embedding_device
        self._check_model_locally_available()
        self._model: Any = self._load_model()

    def _check_model_locally_available(self) -> None:
        if not self._is_model_cached(self.model_name):
            import sys
            from huggingface_hub.constants import HF_HUB_CACHE

            msg = (
                f"\nERROR: The offline embedding model '{self.model_name}' is not available locally.\n"
                f"Expected cache directory: {HF_HUB_CACHE}\n\n"
                f"To pre-download the model for offline usage, please run:\n"
                f"  python scripts/download_model.py\n\n"
                f"Alternatively, run this python command to download it manually:\n"
                f"  python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{self.model_name}')\"\n"
            )
            print(msg, file=sys.stderr)
            sys.exit(1)

    def _is_model_cached(self, model_name: str) -> bool:
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == model_name and repo.repo_type == "model":
                    return True
            return False
        except Exception:
            return False

    @property
    def embedding_mode(self) -> str:
        return "sentence-transformers-offline"

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading offline embedding model '%s' on %s...",
                self.model_name,
                self.device,
            )
            return SentenceTransformer(
                self.model_name,
                device=self.device,
                local_files_only=self.config.embedding_local_files_only,
            )
        except Exception as e:  # noqa: BLE001
            raise PipelineError(
                "Offline embedding model could not be loaded. "
                f"Expected '{self.model_name}' to be available locally. "
                "Pre-download/cache the model before ranking; no TF-IDF fallback is used."
            ) from e

    def encode(self, texts: np.ndarray, batch_size: int) -> np.ndarray:
        if texts.size == 0:
            return np.empty((0, 0), dtype=np.float32)

        safe_texts = np.where(np.char.str_len(texts.astype(str)) > 0, texts.astype(str), " ")
        embeddings = self._model.encode(
            safe_texts.tolist(),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=safe_texts.size > batch_size,
        )
        return np.asarray(embeddings, dtype=np.float32)  # type: ignore[no-any-return]

    def encode_single(self, text: str) -> np.ndarray:
        arr = np.asarray([text if text.strip() else " "], dtype=object)
        result = self.encode(arr, batch_size=1)
        return result[0]
