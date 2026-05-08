"""
Application Layer — Use Case: Retrieve Report.
Consulta um relatório pelo analysis_id.
"""

from __future__ import annotations
from typing import Optional

from app.domain.report_generation.repository import IReportRepository
from app.domain.diagram_analysis.repository import IAnalysisRepository
from app.domain.shared.analysis_id import AnalysisId
from app.shared.logging import get_logger

logger = get_logger(__name__)


class RetrieveReportUseCase:
    def __init__(
        self,
        analysis_repo: IAnalysisRepository,
        report_repo: IReportRepository,
    ):
        self._analysis_repo = analysis_repo
        self._report_repo = report_repo

    def execute(self, analysis_id_str: str) -> Optional[dict]:
        """
        Retorna o relatório completo de uma análise ou None se não encontrado.
        """
        try:
            analysis_id = AnalysisId.from_string(analysis_id_str)
        except ValueError:
            return None

        analysis = self._analysis_repo.get_by_id(analysis_id)
        if not analysis:
            return None

        report_aggregate = self._report_repo.get_by_analysis_id(analysis_id)

        result = {
            "analysis_id": str(analysis_id),
            "status": analysis.status.value,
            "file_name": analysis.file_name,
            "created_at": None,
            "report": None,
        }

        if report_aggregate and report_aggregate.report:
            report_dict = report_aggregate.report.to_dict()
            if report_aggregate.qa_score:
                report_dict["qa_is_valid"] = report_aggregate.qa_score.is_valid
                report_dict["qa_completeness_score"] = report_aggregate.qa_score.completeness_score
                report_dict["qa_issues_found"] = list(report_aggregate.qa_score.issues_found)
                report_dict["qa_quality_notes"] = report_aggregate.qa_score.quality_notes
            result["report"] = report_dict

        return result
