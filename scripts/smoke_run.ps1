$ErrorActionPreference = "Stop"

uv run rank --candidates data/input/candidates.jsonl --jd data/input/job_description.docx --output data/output/submission.csv
uv run validate --submission data/output/submission.csv
