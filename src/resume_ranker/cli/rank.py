"""CLI entry point for ranking candidates."""

import logging
from pathlib import Path
from typing import Optional
import typer

from resume_ranker.app.pipeline import RankingPipeline
from resume_ranker.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = typer.Typer(add_completion=False, help="Rank candidates against a job description")


@app.callback(invoke_without_command=True)
def rank(
    candidates: Optional[Path] = typer.Option(
        None,
        "--candidates",
        help="Optional path to candidates JSONL file (not used during ranking stage)",
    ),
    jd: Path = typer.Option(
        config.input_dir / "job_description.docx",
        "--jd",
        help="Path to job description DOCX file",
    ),
    output: Path = typer.Option(
        config.output_dir / "submission.csv",
        "--output",
        help="Path to write ranked candidates CSV",
    ),
    schema: Optional[Path] = typer.Option(
        None,
        "--schema",
        help="Optional path to candidate schema JSON file",
    ),
) -> None:
    if candidates is not None and not candidates.exists():
        typer.echo(f"ERROR: Candidates file not found: {candidates}", err=True)
        raise typer.Exit(code=1)
    if not jd.exists():
        typer.echo(f"ERROR: JD file not found: {jd}", err=True)
        raise typer.Exit(code=1)
    if schema is not None and not schema.exists():
        typer.echo(f"ERROR: Schema file not found: {schema}", err=True)
        raise typer.Exit(code=1)

    try:
        if schema is not None:
            config.candidate_schema_path = schema

        pipeline = RankingPipeline(config)
        df = pipeline.rank(jd, output)
    except Exception as e:
        typer.echo(f"ERROR: Ranking failed: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Success! Ranked {len(df)} candidates. Output: {output}")
    typer.echo(f"Audit report: {output.with_suffix('.audit.json')}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
