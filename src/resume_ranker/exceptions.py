"""Custom exception hierarchy for the resume ranker system."""

from typing import List


class ResumeRankerError(Exception):
    """Base exception for the resume ranker system."""


class HoneypotDetectedError(ResumeRankerError):
    """Raised when a honeypot candidate is detected.

    Attributes:
        candidate_id: The ID of the detected honeypot candidate.
        reasons: List of reasons why the candidate was flagged.
    """

    def __init__(self, candidate_id: str, reasons: List[str]) -> None:
        self.candidate_id = candidate_id
        self.reasons = reasons
        super().__init__(f"Honeypot detected for {candidate_id}: {'; '.join(reasons)}")


class ValidationError(ResumeRankerError):
    """Raised when submission format validation fails."""


class PipelineError(ResumeRankerError):
    """Raised when the ranking pipeline encounters a fatal error."""
