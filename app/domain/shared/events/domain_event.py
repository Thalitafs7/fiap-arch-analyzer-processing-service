"""
Shared Kernel — base de eventos de domínio.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class DomainEvent:
    """Classe base para todos os eventos de domínio."""
    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        kw_only=True,
    )

    @property
    def event_name(self) -> str:
        return self.__class__.__name__
