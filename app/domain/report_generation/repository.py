"""
Report Bounded Context — Interface do Repositório.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.report_generation.report import ReportAggregate
from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.report_id import ReportId


class IReportRepository(ABC):

    @abstractmethod
    def save(self, report: ReportAggregate) -> None:
        """Persiste o aggregate de relatório."""

    @abstractmethod
    def get_by_analysis_id(self, analysis_id: AnalysisId) -> Optional[ReportAggregate]:
        """Recupera o relatório mais recente de uma análise."""

    @abstractmethod
    def list_reports(self, limit: int = 20, offset: int = 0) -> List[dict]:
        """Lista relatórios paginados para a API de consulta."""

    @abstractmethod
    def count(self) -> int:
        """Total de relatórios para paginação."""
