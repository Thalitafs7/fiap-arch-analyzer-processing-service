from __future__ import annotations
from dataclasses import dataclass

from app.domain.shared.events.domain_event import DomainEvent
from app.domain.shared.analysis_id import AnalysisId


@dataclass(frozen=True)
class DiagramIngestedEvent(DomainEvent):
    """Emitido após validação e encoding bem-sucedidos do arquivo."""
    analysis_id: AnalysisId
    file_type: str
    file_size_kb: float
