"""
Analysis Bounded Context — Interface do Repositório.
Define o contrato de persistência sem depender de infraestrutura.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from app.domain.diagram_analysis.analysis import AnalysisAggregate
from app.domain.shared.analysis_id import AnalysisId


class IAnalysisRepository(ABC):

    @abstractmethod
    def save(self, analysis: AnalysisAggregate) -> None:
        """Persiste (insert ou update) o aggregate."""

    @abstractmethod
    def get_by_id(self, analysis_id: AnalysisId) -> Optional[AnalysisAggregate]:
        """Recupera um aggregate pelo ID. Retorna None se não encontrado."""

    @abstractmethod
    def get_by_sqs_message_id(self, sqs_message_id: str) -> Optional[AnalysisAggregate]:
        """Recupera pelo ID da mensagem SQS (idempotência)."""

    @abstractmethod
    def update_status(self, analysis: AnalysisAggregate) -> None:
        """Atualiza apenas o status e error_message (otimização)."""
