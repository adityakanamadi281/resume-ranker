"""src/parsers/docx_parser.py -- extract job description text from a .docx file."""

from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError


def parse_job_description(path: Path) -> str:
    """Extract all paragraph text from a .docx job description, preserving
    document order and section headers, joined with double newlines
    between paragraphs and with excess whitespace stripped.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if `path` is not a valid .docx file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Job description file not found: {path}. "
            f"Pass a valid path to a .docx job description via --jd."
        )

    try:
        document = Document(str(path))
    except PackageNotFoundError as e:
        raise ValueError(f"{path} is not a valid .docx file: {e}") from e
    except Exception as e:  # noqa: BLE001 -- surface any docx-level parse failure clearly
        raise ValueError(f"Failed to parse {path} as a .docx file: {e}") from e

    paragraphs = []
    for p in document.paragraphs:
        text = " ".join(p.text.split())  # normalize internal whitespace
        if text:
            paragraphs.append(text)

    if not paragraphs:
        raise ValueError(f"{path} contains no extractable paragraph text.")

    return "\n\n".join(paragraphs)
