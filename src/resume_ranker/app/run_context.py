"""Runtime context and stage timing for a ranking run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from types import TracebackType
from typing import Dict, Optional, Type


@dataclass
class RunContext:
    today: date
    started_at: float = field(default_factory=time.time)
    stage_seconds: Dict[str, float] = field(default_factory=dict)

    def elapsed_seconds(self) -> float:
        return time.time() - self.started_at

    def time_stage(self, name: str) -> "_StageTimer":
        return _StageTimer(self, name)


class _StageTimer:
    def __init__(self, context: RunContext, name: str) -> None:
        self.context = context
        self.name = name
        self.started_at = 0.0

    def __enter__(self) -> None:
        self.started_at = time.time()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.context.stage_seconds[self.name] = time.time() - self.started_at
