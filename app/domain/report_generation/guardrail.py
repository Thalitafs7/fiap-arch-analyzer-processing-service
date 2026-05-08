"""
Report Bounded Context — Domain Services.
GuardrailService encapsula as regras de negócio de validação do relatório
que não pertencem exclusivamente a nenhuma entidade.
"""

from __future__ import annotations
from typing import List

from app.domain.diagram_analysis.extraction_result import ExtractionResult
from app.domain.report_generation.technical_report import TechnicalReport
from app.shared.exceptions import ReportGenerationError


class GuardrailService:
    """
    Serviço de domínio que valida guardrails do relatório gerado.

    Regras aplicadas (independentes de LLM ou infraestrutura):
    1. components_identified não pode estar vazio
    2. Máximo 20% de componentes não encontrados na extração (anti-alucinação)
    3. Recomendações não podem estar vazias
    4. executive_summary precisa ter pelo menos 100 caracteres
    """

    HALLUCINATION_THRESHOLD = 0.20
    MIN_SUMMARY_LENGTH = 100

    def validate(
        self,
        report: TechnicalReport,
        extraction: ExtractionResult,
    ) -> None:
        """
        Valida o relatório contra os dados de extração.
        Levanta ReportGenerationError se algum guardrail falhar.
        """
        self._check_components_not_empty(report)
        self._check_hallucination(report, extraction)
        self._check_recommendations_not_empty(report)
        self._check_summary_length(report)

    def _check_components_not_empty(self, report: TechnicalReport) -> None:
        if not report.components_identified:
            raise ReportGenerationError(
                "Guardrail: components_identified está vazio.",
                step="report",
            )

    def _check_hallucination(
        self,
        report: TechnicalReport,
        extraction: ExtractionResult,
    ) -> None:
        report_set = {c.lower() for c in report.components_identified}
        source_set = {c.lower() for c in extraction.component_names}

        if not source_set:
            return

        hallucinated = report_set - source_set
        if len(hallucinated) > len(source_set) * self.HALLUCINATION_THRESHOLD:
            raise ReportGenerationError(
                f"Guardrail: componentes não encontrados na extração: {hallucinated}",
                step="report",
            )

    def _check_recommendations_not_empty(self, report: TechnicalReport) -> None:
        if not report.recommendations:
            raise ReportGenerationError(
                "Guardrail: relatório sem recomendações.",
                step="report",
            )

    def _check_summary_length(self, report: TechnicalReport) -> None:
        if len(report.executive_summary) < self.MIN_SUMMARY_LENGTH:
            raise ReportGenerationError(
                f"Guardrail: sumário executivo insuficiente "
                f"({len(report.executive_summary)} chars, mínimo {self.MIN_SUMMARY_LENGTH}).",
                step="report",
            )
