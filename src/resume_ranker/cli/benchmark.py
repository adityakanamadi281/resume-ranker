"""CLI entry point for benchmarking the ranking pipeline."""

import logging
from pathlib import Path
import typer

from resume_ranker.app.pipeline import RankingPipeline
from resume_ranker.config import config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = typer.Typer(add_completion=False, help="Benchmark the ranking pipeline stages")


@app.callback(invoke_without_command=True)
def benchmark(
    jd: Path = typer.Option(
        config.input_dir / "job_description.docx",
        "--jd",
        help="Path to job description DOCX file",
    ),
    output: Path = typer.Option(
        config.output_dir / "benchmark.csv",
        "--output",
        help="Path to write ranked candidates CSV",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print verbose pipeline execution logs",
    ),
) -> None:
    if not jd.exists():
        typer.echo(f"ERROR: JD file not found: {jd}", err=True)
        raise typer.Exit(code=1)

    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        pipeline = RankingPipeline(config)
        pipeline.rank(jd, output)

        audit = pipeline.last_audit_report
        if not audit or "stage_seconds" not in audit:
            typer.echo("ERROR: Timing data was not captured correctly.", err=True)
            raise typer.Exit(code=1)

        stage_seconds = audit["stage_seconds"]

        # Expected keys to report
        keys = [
            ("Load artifacts", "load_artifacts"),
            ("Embed JD", "embed_jd"),
            ("FAISS search", "faiss_search"),
            ("Feature scoring", "feature_scoring"),
            ("Explanation", "explanation"),
            ("CSV write", "csv_write"),
        ]

        total = 0.0
        for label, key in keys:
            val = stage_seconds.get(key, 0.0)
            typer.echo(f"{label:<24}{val:.2f} s")
            total += val

        typer.echo(f"{'Total':<24}{total:.2f} s")

    except Exception as e:
        typer.echo(f"ERROR: Benchmarking failed: {e}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
