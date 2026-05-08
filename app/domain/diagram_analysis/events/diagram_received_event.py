from __future__ import annotations
from dataclasses import dataclass

from app.domain.shared.events.domain_event import DomainEvent
from app.domain.shared.analysis_id import AnalysisId


@dataclass(frozen=True)
class DiagramReceivedEvent(DomainEvent):
    """Emitido quando um diagrama é recebido (via upload ou SQS)."""
    analysis_id: AnalysisId
    file_name: str
    source: str  # "upload" | "sqs"
