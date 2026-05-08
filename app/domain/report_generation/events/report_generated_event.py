from __future__ import annotations
from dataclasses import dataclass

from app.domain.shared.events.domain_event import DomainEvent
from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.report_id import ReportId


@dataclass(frozen=True)
class ReportGeneratedEvent(DomainEvent):
    """Emitido quando um relatório técnico é gerado com sucesso."""
    report_id: ReportId
    analysis_id: AnalysisId
    rag_used: bool
    risks_count: int
