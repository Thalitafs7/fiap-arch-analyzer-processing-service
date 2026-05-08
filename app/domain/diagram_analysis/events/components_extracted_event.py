from __future__ import annotations
from dataclasses import dataclass

from app.domain.shared.events.domain_event import DomainEvent
from app.domain.shared.analysis_id import AnalysisId


@dataclass(frozen=True)
class ComponentsExtractedEvent(DomainEvent):
    """Emitido após extração de componentes via LLM Vision."""
    analysis_id: AnalysisId
    components_count: int
    patterns_count: int
