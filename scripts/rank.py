"""scripts/rank.py -- CLI entry point for reproduction."""

import argparse
import logging
import sys
from pathlib import Path

from src.config import config
from src.pipeline import RankingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

parser = argparse.ArgumentParser(description="Rank candidates against a job description")
parser.add_argument("--candidates", type=Path, default=config.input_dir / "candidates.jsonl")
parser.add_argument("--jd", type=Path, default=config.input_dir / "job_description.docx")
parser.add_argument("--output", type=Path, default=config.output_dir / "submission.csv")
parser.add_argument("--config", type=Path, default=None, help="Optional config override")


def main() -> int:
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"ERROR: Candidates file not found: {args.candidates}", file=sys.stderr)
        return 1
    if not args.jd.exists():
        print(f"ERROR: JD file not found: {args.jd}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        pipeline = RankingPipeline(config)
        df = pipeline.run(args.candidates, args.jd, args.output)
    except Exception as e:  # noqa: BLE001 -- surface any pipeline failure with a clean message
        print(f"ERROR: Pipeline failed: {e}", file=sys.stderr)
        return 1

    print(f"Success! Ranked {len(df)} candidates. Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
