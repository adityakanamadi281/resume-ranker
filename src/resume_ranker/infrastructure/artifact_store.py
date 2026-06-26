"""Versioned storage for feature tables, embeddings, IDs, and FAISS indexes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import orjson
import polars as pl

from resume_ranker.config import AppConfig


@dataclass(frozen=True)
class ArtifactManifest:
    candidate_file_hash: str
    jd_file_hash: str
    model_name: str
    embedding_mode: str
    config_hash: str
    candidate_count: int
    embedding_dim: int
    created_at: str
    embedding_model: str = ""
    embedding_dimension: int = 0
    embedding_dtype: str = "float32"
    faiss_index: str = "IndexFlatIP"
    candidate_hash: str = ""
    schema_hash: str = ""
    version: int = 1
    total_candidates: int = 0
    honeypots_excluded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_file_hash": self.candidate_file_hash,
            "jd_file_hash": self.jd_file_hash,
            "model_name": self.model_name,
            "embedding_mode": self.embedding_mode,
            "config_hash": self.config_hash,
            "candidate_count": self.candidate_count,
            "embedding_dim": self.embedding_dim,
            "created_at": self.created_at,
            "embedding_model": self.embedding_model or self.model_name,
            "embedding_dimension": self.embedding_dimension or self.embedding_dim,
            "embedding_dtype": self.embedding_dtype,
            "faiss_index": self.faiss_index,
            "candidate_hash": self.candidate_hash or self.candidate_file_hash,
            "schema_hash": self.schema_hash,
            "version": self.version,
            "total_candidates": self.total_candidates,
            "honeypots_excluded": self.honeypots_excluded,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactManifest":
        def get_str(key: str, default: str = "") -> str:
            val = payload.get(key)
            return default if val is None else str(val)

        def get_int(key: str, default: int = 0) -> int:
            val = payload.get(key)
            return default if val is None else int(val)

        model_name = get_str("model_name", get_str("embedding_model", ""))
        embedding_dim = get_int("embedding_dim", get_int("embedding_dimension", 0))
        candidate_file_hash = get_str("candidate_file_hash", get_str("candidate_hash", ""))

        return cls(
            candidate_file_hash=candidate_file_hash,
            jd_file_hash=get_str("jd_file_hash", ""),
            model_name=model_name,
            embedding_mode=get_str("embedding_mode", "sentence-transformers-offline"),
            config_hash=get_str("config_hash", ""),
            candidate_count=get_int("candidate_count", 0),
            embedding_dim=embedding_dim,
            created_at=get_str("created_at", ""),
            embedding_model=get_str("embedding_model", model_name),
            embedding_dimension=get_int("embedding_dimension", embedding_dim),
            embedding_dtype=get_str("embedding_dtype", "float32"),
            faiss_index=get_str("faiss_index", "IndexFlatIP"),
            candidate_hash=get_str("candidate_hash", candidate_file_hash),
            schema_hash=get_str("schema_hash", ""),
            version=get_int("version", 1),
            total_candidates=get_int("total_candidates", 0),
            honeypots_excluded=get_int("honeypots_excluded", 0),
        )


class ArtifactStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = config.artifacts_dir
        self.features_path = self.root / config.feature_parquet_filename
        self.embeddings_path = self.root / config.embeddings_filename
        self.candidate_ids_path = self.root / config.candidate_ids_filename
        self.index_path = self.root / config.faiss_index_filename
        self.manifest_path = self.root / config.manifest_filename

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def file_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def config_hash(self) -> str:
        payload = self.config.model_dump(mode="json")
        for key in ("input_dir", "output_dir", "artifacts_dir"):
            payload.pop(key, None)
        serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(serialized).hexdigest()

    def expected_manifest(
        self,
        candidates_path: Path,
        jd_path: Optional[Path],
        embedding_mode: str,
        candidate_count: int,
        embedding_dim: int,
        total_candidates: int = 0,
        honeypots_excluded: int = 0,
    ) -> ArtifactManifest:
        cand_hash = self.file_hash(candidates_path) if candidates_path.exists() else ""
        jd_hash = self.file_hash(jd_path) if jd_path is not None and jd_path.exists() else ""
        schema_path = self.config.candidate_schema_path
        if schema_path is None:
            schema_path = self.config.input_dir / self.config.candidate_schema_filename
        schema_h = self.file_hash(schema_path) if schema_path.exists() else ""

        return ArtifactManifest(
            candidate_file_hash=cand_hash,
            jd_file_hash=jd_hash,
            model_name=self.config.embedding_model,
            embedding_mode=embedding_mode,
            config_hash=self.config_hash(),
            candidate_count=candidate_count,
            embedding_dim=embedding_dim,
            created_at=datetime.now(timezone.utc).isoformat(),
            embedding_model=self.config.embedding_model,
            embedding_dimension=embedding_dim,
            embedding_dtype="float32",
            faiss_index="IndexFlatIP",
            candidate_hash=cand_hash,
            schema_hash=schema_h,
            version=1,
            total_candidates=total_candidates,
            honeypots_excluded=honeypots_excluded,
        )

    def load_manifest(self) -> Optional[ArtifactManifest]:
        if not self.manifest_path.exists():
            return None
        payload = orjson.loads(self.manifest_path.read_bytes())
        return ArtifactManifest.from_dict(payload)

    def save_manifest(self, manifest: ArtifactManifest) -> None:
        self.ensure_root()
        self.manifest_path.write_bytes(
            orjson.dumps(manifest.to_dict(), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )

    def save_features(self, features: pl.DataFrame) -> None:
        self.ensure_root()
        features.write_parquet(self.features_path)

    def load_embeddings_if_valid(
        self,
        expected: ArtifactManifest,
        expected_ids: np.ndarray,
    ) -> Optional[np.ndarray]:
        manifest = self.load_manifest()
        if (
            manifest is None
            or not self.embeddings_path.exists()
            or not self.candidate_ids_path.exists()
        ):
            return None

        cached_ids = np.load(self.candidate_ids_path, allow_pickle=True)
        if cached_ids.shape != expected_ids.shape or not np.array_equal(cached_ids, expected_ids):
            return None

        current = manifest.to_dict()
        target = expected.to_dict()
        current.pop("created_at", None)
        target.pop("created_at", None)
        if current != target:
            return None

        embeddings: np.ndarray = np.load(self.embeddings_path)
        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        candidate_ids: np.ndarray,
        manifest: ArtifactManifest,
    ) -> None:
        self.ensure_root()
        np.save(self.embeddings_path, embeddings)
        np.save(self.candidate_ids_path, np.asarray(candidate_ids).astype(object))
        self.save_manifest(manifest)

    def save_index(self, index: Any) -> None:
        import faiss

        self.ensure_root()
        faiss.write_index(index, str(self.index_path))
