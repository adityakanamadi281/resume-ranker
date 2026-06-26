"""Pre-compute candidate embeddings and a FAISS index."""

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
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="Pre-compute candidate embeddings and FAISS index")


@app.callback(invoke_without_command=True)
def precompute(
    candidates: Path = typer.Option(
        config.input_dir / "candidates.jsonl",
        "--candidates",
        help="Path to candidates JSONL file",
    ),
    jd: Optional[Path] = typer.Option(
        None,
        "--jd",
        help="Optional path to job description DOCX file",
    ),
    schema: Optional[Path] = typer.Option(
        None,
        "--schema",
        help="Optional path to candidate schema JSON file",
    ),
) -> None:
    if not candidates.exists():
        typer.echo(f"ERROR: Candidates file not found: {candidates}", err=True)
        raise typer.Exit(code=1)
    if jd is not None and not jd.exists():
        typer.echo(f"ERROR: JD file not found: {jd}", err=True)
        raise typer.Exit(code=1)
    if schema is not None and not schema.exists():
        typer.echo(f"ERROR: Schema file not found: {schema}", err=True)
        raise typer.Exit(code=1)

    try:
        if schema is not None:
            config.candidate_schema_path = schema

        pipeline = RankingPipeline(config)
        pipeline.precompute(candidates, jd)
    except Exception as e:
        typer.echo(f"ERROR: Precomputation failed: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Success! Precomputation completed. Artifacts in {config.artifacts_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
