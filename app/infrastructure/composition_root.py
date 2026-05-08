"""
Infrastructure Layer — Composition Root (Factory de Dependências).

Monta o grafo de dependências DDD: instancia os adaptadores de infraestrutura
e injeta nas use cases. Ponto único de wiring — nenhuma outra camada conhece
as implementações concretas.

Uso:
    from app.infrastructure.composition_root import build_analyze_use_case
    use_case = build_analyze_use_case(db_session)
    result = use_case.execute(file_bytes, file_name)
"""

from __future__ import annotations
from sqlalchemy.orm import Session

from app.application.use_cases.analyze_diagram import AnalyzeDiagramUseCase
from app.application.use_cases.retrieve_report import RetrieveReportUseCase
from app.domain.report_generation.guardrail import GuardrailService
from app.domain.shared.input_guardrail import InputGuardrailService
from app.domain.shared.output_guardrail import OutputGuardrailService

from app.infrastructure.persistence.sqlalchemy_analysis_repository import (
    SQLAlchemyAnalysisRepository,
)
from app.infrastructure.persistence.sqlalchemy_report_repository import (
    SQLAlchemyReportRepository,
)
from app.infrastructure.llm.openai_adapter import (
    OpenAIVisionAdapter,
    OpenAITextAdapter,
)
from app.infrastructure.vector_store.pgvector_adapter import PGVectorAdapter


def build_analyze_use_case(db: Session) -> AnalyzeDiagramUseCase:
    """
    Constrói o AnalyzeDiagramUseCase com todas as dependências injetadas.
    Chamado a cada request (o db session é por-request).
    """
    return AnalyzeDiagramUseCase(
        analysis_repo=SQLAlchemyAnalysisRepository(db),
        report_repo=SQLAlchemyReportRepository(db),
        vision_llm=OpenAIVisionAdapter(),
        text_llm=OpenAITextAdapter(),
        vector_store=PGVectorAdapter(db),
        guardrail_svc=GuardrailService(),
        input_guardrail=InputGuardrailService(),
        output_guardrail=OutputGuardrailService(),
    )


def build_retrieve_report_use_case(db: Session) -> RetrieveReportUseCase:
    """
    Constrói o RetrieveReportUseCase para consulta de relatórios.
    """
    return RetrieveReportUseCase(
        analysis_repo=SQLAlchemyAnalysisRepository(db),
        report_repo=SQLAlchemyReportRepository(db),
    )
