"""Pre-submission CSV validator."""

from pathlib import Path
from typing import List
import typer

from resume_ranker.app.output_writer import SubmissionValidator
from resume_ranker.config import config

app = typer.Typer(add_completion=False, help="Validate submission.csv before final submission")


def validate(submission_path: Path) -> List[str]:
    return SubmissionValidator(config).validate_csv(submission_path)


@app.callback(invoke_without_command=True)
def validate_cli(
    submission: Path = typer.Option(
        config.output_dir / "submission.csv",
        "--submission",
        help="Path to submission.csv to validate",
    )
) -> None:
    errors = validate(submission)

    if not errors:
        typer.echo("PASS")
        raise typer.Exit(code=0)

    typer.echo(f"FAIL: {len(errors)} error(s) found", err=True)
    for error in errors:
        typer.echo(f"  - {error}", err=True)
    raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
