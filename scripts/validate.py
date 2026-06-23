"""scripts/validate.py -- pre-submission format validator."""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

from src.config import config

parser = argparse.ArgumentParser(description="Validate submission.csv before final submission")
parser.add_argument("--submission", type=Path, default=config.output_dir / "submission.csv")

_CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def validate(submission_path: Path) -> List[str]:
    errors: List[str] = []

    if not submission_path.exists():
        return [f"Submission file not found: {submission_path}"]

    try:
        df = pd.read_csv(submission_path)
    except Exception as e:  # noqa: BLE001
        return [f"Failed to read {submission_path} as CSV: {e}"]

    expected_columns = config.csv_columns
    if list(df.columns) != expected_columns:
        errors.append(f"Columns: expected {expected_columns}, got {list(df.columns)}")
        return errors

    expected_rows = config.top_n
    if len(df) != expected_rows:
        errors.append(f"Row count: expected {expected_rows} data rows, got {len(df)}")

    expected_ranks = list(range(1, len(df) + 1))
    actual_ranks = sorted(df["rank"].tolist())
    if actual_ranks != expected_ranks:
        errors.append(
            f"Ranks: expected exactly 1..{len(df)} each once, "
            f"got {len(set(actual_ranks))} unique values"
        )

    duplicated_ids = df["candidate_id"][df["candidate_id"].duplicated()].tolist()
    if duplicated_ids:
        errors.append(f"Candidate IDs: {len(duplicated_ids)} duplicate(s) found: {duplicated_ids[:5]}")

    invalid_ids = [
        cid for cid in df["candidate_id"] if not _CANDIDATE_ID_PATTERN.match(str(cid))
    ]
    if invalid_ids:
        errors.append(f"Candidate IDs: {len(invalid_ids)} invalid format: {invalid_ids[:5]}")

    df_sorted_by_rank = df.sort_values("rank")
    scores = df_sorted_by_rank["score"].tolist()
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        errors.append("Scores: not non-increasing when sorted by rank")

    empty_reasoning = df["reasoning"].isna() | (df["reasoning"].astype(str).str.strip() == "")
    n_empty = int(empty_reasoning.sum())
    if n_empty > 0:
        print(f"WARNING: {n_empty} row(s) have empty reasoning.", file=sys.stderr)

    return errors


def main() -> int:
    args = parser.parse_args()
    errors = validate(args.submission)

    if not errors:
        print("PASS")
        return 0

    print(f"FAIL: {len(errors)} error(s) found")
    for e in errors:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
