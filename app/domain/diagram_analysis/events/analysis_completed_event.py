from __future__ import annotations
from dataclasses import dataclass

from app.domain.shared.events.domain_event import DomainEvent
from app.domain.shared.analysis_id import AnalysisId


@dataclass(frozen=True)
class AnalysisCompletedEvent(DomainEvent):
    """Emitido quando o pipeline completa com sucesso."""
    analysis_id: AnalysisId
    qa_score: float
