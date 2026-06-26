"""Submission validation and CSV writing with Polars."""

from pathlib import Path
from typing import List

import polars as pl

from resume_ranker.config import AppConfig
from resume_ranker.exceptions import PipelineError


class SubmissionValidator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def validate_dataframe(self, df: pl.DataFrame) -> None:
        expected_n = min(self.config.top_n, df.height)
        if df.height != expected_n:
            raise PipelineError(f"Expected {expected_n} rows, got {df.height}")
        if df.columns != self.config.csv_columns:
            raise PipelineError(f"Expected columns {self.config.csv_columns}, got {df.columns}")
        if df.select(pl.col("candidate_id").is_duplicated().any()).item():
            raise PipelineError("Duplicate candidate_id values in final output.")

        scores = df.get_column("score").to_numpy()
        if scores.size > 1 and bool((scores[:-1] < scores[1:]).any()):
            raise PipelineError("Scores are not non-increasing by rank.")

        ranks = df.get_column("rank").to_numpy()
        expected_ranks = list(range(1, df.height + 1))
        if ranks.tolist() != expected_ranks:
            raise PipelineError("Ranks are not exactly 1..N in order.")

    def validate_csv(self, submission_path: Path) -> List[str]:
        if not submission_path.exists():
            return [f"Submission file not found: {submission_path}"]

        try:
            df = pl.read_csv(submission_path)
        except Exception as e:  # noqa: BLE001
            return [f"Failed to read {submission_path} as CSV: {e}"]

        errors: List[str] = []
        try:
            self.validate_dataframe(df)
        except PipelineError as e:
            errors.append(str(e))

        empty_reasoning = df.select(
            (pl.col("reasoning").is_null() | (pl.col("reasoning").cast(pl.Utf8).str.strip_chars() == ""))
            .sum()
            .alias("empty")
        ).item()
        if int(empty_reasoning) > 0:
            errors.append(f"{empty_reasoning} row(s) have empty reasoning")

        return errors


class SubmissionWriter:
    def __init__(self, config: AppConfig) -> None:
        self.validator = SubmissionValidator(config)

    def write(self, df: pl.DataFrame, output_path: Path) -> None:
        self.validator.validate_dataframe(df)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_path)
