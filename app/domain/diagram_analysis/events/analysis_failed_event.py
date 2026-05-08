from __future__ import annotations
from dataclasses import dataclass

from app.domain.shared.events.domain_event import DomainEvent
from app.domain.shared.analysis_id import AnalysisId


@dataclass(frozen=True)
class AnalysisFailedEvent(DomainEvent):
    """Emitido quando qualquer etapa do pipeline falha."""
    analysis_id: AnalysisId
    step: str
    error_message: str
