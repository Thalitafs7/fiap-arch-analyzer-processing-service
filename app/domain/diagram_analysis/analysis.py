"""
Analysis Bounded Context — Aggregate Root.
AnalysisAggregate é a raiz de agregado que controla o ciclo de vida
de uma análise de diagrama, garantindo consistência de estado.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.events.domain_event import DomainEvent
from app.domain.diagram_analysis.analysis_status import AnalysisStatus
from app.domain.diagram_analysis.diagram_file import DiagramFile
from app.domain.diagram_analysis.extraction_result import ExtractionResult
from app.domain.diagram_analysis.events.diagram_received_event import DiagramReceivedEvent
from app.domain.diagram_analysis.events.diagram_ingested_event import DiagramIngestedEvent
from app.domain.diagram_analysis.events.components_extracted_event import ComponentsExtractedEvent
from app.domain.diagram_analysis.events.analysis_completed_event import AnalysisCompletedEvent
from app.domain.diagram_analysis.events.analysis_failed_event import AnalysisFailedEvent


@dataclass
class AnalysisAggregate:
    """
    Raiz de agregado da análise de diagrama.

    Controla todas as transições de estado e garante invariantes de domínio:
    - Um diagrama só pode ser processado a partir do estado RECEIVED
    - A extração só pode acontecer após a ingestão
    - O pipeline só pode completar após extração bem-sucedida
    """
    id: AnalysisId
    status: AnalysisStatus
    file_name: str
    file_type: str
    s3_key: Optional[str] = None
    sqs_message_id: Optional[str] = None
    diagram_file: Optional[DiagramFile] = None
    extraction_result: Optional[ExtractionResult] = None
    error_message: Optional[str] = None
    _events: List[DomainEvent] = field(default_factory=list, repr=False)

    @classmethod
    def create(
        cls,
        analysis_id: AnalysisId,
        file_name: str,
        file_type: str,
        s3_key: Optional[str] = None,
        sqs_message_id: Optional[str] = None,
        source: str = "upload",
    ) -> "AnalysisAggregate":
        """Factory — cria uma nova análise no estado RECEIVED."""
        aggregate = cls(
            id=analysis_id,
            status=AnalysisStatus.RECEIVED,
            file_name=file_name,
            file_type=file_type,
            s3_key=s3_key,
            sqs_message_id=sqs_message_id,
        )
        aggregate._events.append(
            DiagramReceivedEvent(
                analysis_id=analysis_id,
                file_name=file_name,
                source=source,
            )
        )
        return aggregate

    def start_ingestion(self, diagram_file: DiagramFile) -> None:
        """Registra o arquivo ingerido e transita para PROCESSING."""
        self._assert_status(AnalysisStatus.RECEIVED, "start_ingestion")
        self.diagram_file = diagram_file
        self.status = AnalysisStatus.PROCESSING
        self._events.append(
            DiagramIngestedEvent(
                analysis_id=self.id,
                file_type=diagram_file.file_type.value,
                file_size_kb=diagram_file.file_size_kb,
            )
        )

    def complete_extraction(self, extraction_result: ExtractionResult) -> None:
        """Registra o resultado de extração."""
        self._assert_status(AnalysisStatus.PROCESSING, "complete_extraction")
        self.extraction_result = extraction_result
        self._events.append(
            ComponentsExtractedEvent(
                analysis_id=self.id,
                components_count=len(extraction_result.components),
                patterns_count=len(extraction_result.patterns),
            )
        )

    def complete(self, qa_score: float) -> None:
        """Marca a análise como concluída com sucesso."""
        self._assert_status(AnalysisStatus.PROCESSING, "complete")
        self.status = AnalysisStatus.ANALYZED
        self._events.append(
            AnalysisCompletedEvent(
                analysis_id=self.id,
                qa_score=qa_score,
            )
        )

    def fail(self, step: str, error_message: str) -> None:
        """Marca a análise como falha."""
        self.status = AnalysisStatus.ERROR
        self.error_message = error_message
        self._events.append(
            AnalysisFailedEvent(
                analysis_id=self.id,
                step=step,
                error_message=error_message,
            )
        )

    def pull_events(self) -> List[DomainEvent]:
        """Retorna e limpa os eventos pendentes (padrão outbox)."""
        events = list(self._events)
        self._events.clear()
        return events

    def _assert_status(self, expected: AnalysisStatus, operation: str) -> None:
        if self.status != expected:
            raise ValueError(
                f"Operação '{operation}' inválida no estado '{self.status.value}'. "
                f"Estado esperado: '{expected.value}'."
            )
