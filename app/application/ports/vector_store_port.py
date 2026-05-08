"""
Application Layer — Port para Vector Store (RAG).
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from app.domain.diagram_analysis.extraction_result import ExtractionResult
from app.domain.report_generation.rag_context import RagContext
from app.domain.shared.analysis_id import AnalysisId


class IVectorStore(ABC):
    """Port para indexação e recuperação semântica de análises anteriores."""

    @abstractmethod
    def index(self, analysis_id: AnalysisId, extraction: ExtractionResult) -> None:
        """
        Indexa a extração atual no vector store para uso futuro.
        Non-blocking — falhas devem ser logadas, não propagadas.
        """

    @abstractmethod
    def mark_as_reported(self, analysis_id: AnalysisId) -> None:
        """
        Marca o documento indexado como tendo relatório gerado,
        tornando-o disponível para consultas RAG futuras.
        """

    @abstractmethod
    def retrieve_context(
        self,
        extraction: ExtractionResult,
        exclude_analysis_id: AnalysisId,
    ) -> RagContext:
        """
        Recupera análises similares e monta o contexto RAG.
        Retorna RagContext.empty() se não houver contexto disponível.
        """
