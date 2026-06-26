"""Memory-efficient streaming JSONL candidate reader."""

import logging
from pathlib import Path
from typing import Iterator

import orjson
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from resume_ranker.config import AppConfig
from resume_ranker.domain.schema import Candidate

logger = logging.getLogger(__name__)


def _load_jsonschema_validator(config: AppConfig) -> Draft202012Validator | None:
    schema_path = config.candidate_schema_path
    if schema_path is None:
        schema_path = config.input_dir / config.candidate_schema_filename
    if not schema_path.exists():
        return None
    schema = orjson.loads(schema_path.read_bytes())
    return Draft202012Validator(schema)


def stream_candidates(path: Path, config: AppConfig) -> Iterator[Candidate]:
    """Stream-parse candidates.jsonl one line at a time, yielding validated
    Candidate objects. Never accumulates all candidates in memory.

    Invalid lines (malformed JSON or schema validation failures) are
    logged as warnings and skipped -- this function never raises on a
    single bad record, only on the file itself being missing.

    Raises:
        FileNotFoundError: if `path` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {path}. Pass a valid path via --candidates."
        )

    n_seen = 0
    n_valid = 0
    n_skipped = 0

    schema_validator = _load_jsonschema_validator(config)

    with open(path, "rb") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            n_seen += 1

            try:
                raw = orjson.loads(line)
            except orjson.JSONDecodeError as e:
                n_skipped += 1
                logger.warning("Line %d: invalid JSON, skipping (%s)", line_number, e)
                continue

            if schema_validator is not None:
                try:
                    schema_validator.validate(raw)
                except JsonSchemaValidationError as e:
                    n_skipped += 1
                    candidate_id = (
                        raw.get("candidate_id", "<unknown>")
                        if isinstance(raw, dict)
                        else "<unknown>"
                    )
                    logger.warning(
                        "Line %d: candidate %s failed JSON Schema validation, skipping (%s)",
                        line_number,
                        candidate_id,
                        e.message,
                    )
                    continue

            try:
                candidate = Candidate.model_validate(raw)
            except PydanticValidationError as e:
                n_skipped += 1
                candidate_id = raw.get("candidate_id", "<unknown>") if isinstance(raw, dict) else "<unknown>"
                logger.warning(
                    "Line %d: candidate %s failed schema validation, skipping (%s)",
                    line_number, candidate_id, e,
                )
                continue

            n_valid += 1
            if n_seen % config.progress_log_interval == 0:
                logger.info(
                    "Streamed %d records so far (%d valid, %d skipped)",
                    n_seen, n_valid, n_skipped,
                )

            yield candidate

    logger.info(
        "Finished streaming %s: %d records seen, %d valid, %d skipped",
        path, n_seen, n_valid, n_skipped,
    )
