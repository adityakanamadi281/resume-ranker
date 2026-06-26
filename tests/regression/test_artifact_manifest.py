import json

import numpy as np

from resume_ranker.config import config
from resume_ranker.infrastructure.artifact_store import ArtifactStore


def test_artifact_manifest_prevents_stale_embedding_reuse(tmp_path):
    candidates_path = tmp_path / "candidates.jsonl"
    jd_path = tmp_path / "job_description.docx"
    candidates_path.write_text('{"candidate_id":"CAND_0000001"}\n', encoding="utf-8")
    jd_path.write_text("python engineer", encoding="utf-8")

    app_config = config.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
    store = ArtifactStore(app_config)
    manifest = store.expected_manifest(
        candidates_path=candidates_path,
        jd_path=jd_path,
        embedding_mode="fake-local",
        candidate_count=1,
        embedding_dim=2,
    )
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    store.save_embeddings(embeddings, ["CAND_0000001"], manifest)

    assert store.load_embeddings_if_valid(manifest, np.asarray(["CAND_0000001"])) is not None

    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    payload["config_hash"] = "stale"
    store.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert store.load_embeddings_if_valid(manifest, np.asarray(["CAND_0000001"])) is None
