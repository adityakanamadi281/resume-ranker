"""scripts/precompute.py -- pre-compute embeddings and FAISS index (allowed to exceed 5 min)."""

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from src.config import config
from src.embedding.embedder import LocalEmbedder
from src.embedding.index import FaissCandidateIndex
from src.features import honeypot_detector, text_builder
from src.parsers.candidate_parser import stream_candidates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings and FAISS index")
parser.add_argument("--candidates", type=Path, default=config.input_dir / "candidates.jsonl")
parser.add_argument("--output", type=Path, default=config.artifacts_dir)


def main() -> int:
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not args.candidates.exists():
        print(f"ERROR: Candidates file not found: {args.candidates}", file=sys.stderr)
        return 1

    t_start = time.time()
    today = datetime.date.today()

    ids = []
    texts = []
    n_total = 0
    n_honeypots = 0

    for candidate in stream_candidates(args.candidates):
        n_total += 1
        is_honeypot, _ = honeypot_detector.detect(candidate, today)
        if is_honeypot:
            n_honeypots += 1
            continue
        ids.append(candidate.candidate_id)
        texts.append(text_builder.build(candidate))

        if n_total % config.progress_log_interval == 0:
            logger.info("Processed %d candidates (%d valid, %d honeypots)", n_total, len(ids), n_honeypots)

    logger.info("Streaming complete: %d total, %d valid, %d honeypots", n_total, len(ids), n_honeypots)

    embedder = LocalEmbedder(config.embedding_model, config.embedding_device)
    logger.info("Encoding %d candidate texts...", len(texts))
    t0 = time.time()
    embeddings = embedder.encode(texts, batch_size=config.embedding_batch_size)
    logger.info("Encoded in %.1fs", time.time() - t0)

    embeddings_path = args.output / "candidate_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info("Saved embeddings to %s", embeddings_path)

    id_map_path = args.output / "id_map.json"
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    logger.info("Saved ID mapping to %s", id_map_path)

    index = FaissCandidateIndex(dim=embeddings.shape[1])
    index.build(embeddings, ids)

    import faiss
    index_path = args.output / "faiss.index"
    faiss.write_index(index.index, str(index_path))
    logger.info("Saved FAISS index to %s", index_path)

    elapsed = time.time() - t_start
    logger.info("Precompute complete in %.1fs (%.1f minutes)", elapsed, elapsed / 60.0)
    print(f"Success! Precomputed {len(ids)} embeddings in {elapsed:.1f}s. Artifacts in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
