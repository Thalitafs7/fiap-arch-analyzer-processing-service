"""
Report Bounded Context — Aggregate Root.
ReportAggregate controla o ciclo de vida de um relatório técnico,
incluindo geração, validação de qualidade e estado final.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.report_id import ReportId
from app.domain.shared.events.domain_event import DomainEvent
from app.domain.report_generation.technical_report import TechnicalReport
from app.domain.report_generation.qa_score import QAScore
from app.domain.report_generation.events.report_generated_event import ReportGeneratedEvent
from app.domain.report_generation.events.qa_validation_completed_event import QAValidationCompletedEvent


@dataclass
class ReportAggregate:
    """
    Raiz de agregado do relatório técnico.
    Garante que QA seja sempre executado antes de marcar o relatório como válido.
    """
    id: ReportId
    analysis_id: AnalysisId
    report: Optional[TechnicalReport] = None
    qa_score: Optional[QAScore] = None
    _events: List[DomainEvent] = field(default_factory=list, repr=False)

    @classmethod
    def create(cls, report_id: ReportId, analysis_id: AnalysisId) -> "ReportAggregate":
        return cls(id=report_id, analysis_id=analysis_id)

    def attach_report(self, report: TechnicalReport) -> None:
        """Registra o relatório gerado e emite evento."""
        self.report = report
        self._events.append(
            ReportGeneratedEvent(
                report_id=self.id,
                analysis_id=self.analysis_id,
                rag_used=report.rag_used,
                risks_count=len(report.architectural_risks),
            )
        )

    def attach_qa(self, qa_score: QAScore) -> None:
        """Registra o resultado de QA e emite evento."""
        self.qa_score = qa_score
        self._events.append(
            QAValidationCompletedEvent(
                report_id=self.id,
                analysis_id=self.analysis_id,
                is_valid=qa_score.is_valid,
                completeness_score=qa_score.completeness_score,
            )
        )

    @property
    def is_valid(self) -> bool:
        return self.qa_score is not None and self.qa_score.is_valid

    def pull_events(self) -> List[DomainEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def to_persistence_dict(self) -> dict:
        """Serializa para persistência no banco de dados."""
        if self.report is None:
            raise ValueError("Relatório não pode ser persistido sem conteúdo.")
        result = self.report.to_dict()
        if self.qa_score:
            result.update({
                "qa_is_valid": self.qa_score.is_valid,
                "qa_completeness_score": self.qa_score.completeness_score,
                "qa_issues_found": list(self.qa_score.issues_found),
                "qa_quality_notes": self.qa_score.quality_notes,
            })
        return result
