from __future__ import annotations
from dataclasses import dataclass

from app.domain.shared.events.domain_event import DomainEvent
from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.report_id import ReportId


@dataclass(frozen=True)
class QAValidationCompletedEvent(DomainEvent):
    """Emitido após conclusão da validação de qualidade."""
    report_id: ReportId
    analysis_id: AnalysisId
    is_valid: bool
    completeness_score: float
