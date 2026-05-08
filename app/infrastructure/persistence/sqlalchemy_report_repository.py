"""
Infrastructure Layer — SQLAlchemy implementation of IReportRepository.
"""

from __future__ import annotations
import json
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.domain.report_generation.report import ReportAggregate
from app.domain.report_generation.repository import IReportRepository
from app.domain.report_generation.technical_report import TechnicalReport
from app.domain.report_generation.qa_score import QAScore
from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.report_id import ReportId
from app.shared.logging import get_logger

logger = get_logger(__name__)


class SQLAlchemyReportRepository(IReportRepository):
    """Repositório concreto de ReportAggregate usando SQLAlchemy (raw SQL)."""

    def __init__(self, db: Session):
        self._db = db

    def save(self, report: ReportAggregate) -> None:
        if report.report is None:
            raise ValueError("Não é possível persistir ReportAggregate sem TechnicalReport.")

        r = report.report
        qa = report.qa_score

        self._db.execute(
            text("""
                INSERT INTO reports (
                    id, analysis_id, components_identified, architectural_risks,
                    recommendations, executive_summary, rag_used,
                    qa_is_valid, qa_completeness_score, qa_issues_found, qa_quality_notes
                ) VALUES (
                    :id, :analysis_id, :components_identified, :architectural_risks,
                    :recommendations, :executive_summary, :rag_used,
                    :qa_is_valid, :qa_completeness_score, :qa_issues_found, :qa_quality_notes
                )
            """),
            {
                "id": str(report.id),
                "analysis_id": str(report.analysis_id),
                "components_identified": json.dumps(r.components_identified, ensure_ascii=False),
                "architectural_risks": json.dumps(
                    [risk.to_dict() for risk in r.architectural_risks], ensure_ascii=False
                ),
                "recommendations": json.dumps(
                    [str(rec) for rec in r.recommendations], ensure_ascii=False
                ),
                "executive_summary": r.executive_summary,
                "rag_used": r.rag_used,
                "qa_is_valid": qa.is_valid if qa else None,
                "qa_completeness_score": qa.completeness_score if qa else None,
                "qa_issues_found": json.dumps(list(qa.issues_found), ensure_ascii=False) if qa else None,
                "qa_quality_notes": qa.quality_notes if qa else None,
            },
        )
        self._db.commit()
        logger.info("report.saved", report_id=str(report.id), analysis_id=str(report.analysis_id))

    def get_by_analysis_id(self, analysis_id: AnalysisId) -> Optional[ReportAggregate]:
        row = self._db.execute(
            text("""
                SELECT * FROM reports
                WHERE analysis_id = :id
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"id": str(analysis_id)},
        ).mappings().first()

        if not row:
            return None

        return self._row_to_aggregate(dict(row))

    def list_reports(self, limit: int = 20, offset: int = 0) -> List[dict]:
        rows = self._db.execute(
            text("""
                SELECT r.*, a.status, a.file_name, a.created_at as analysis_created_at
                FROM reports r
                JOIN analyses a ON r.analysis_id = a.id
                ORDER BY r.created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            {"limit": limit, "offset": offset},
        ).mappings().all()
        return [dict(r) for r in rows]

    def count(self) -> int:
        result = self._db.execute(text("SELECT COUNT(*) FROM reports")).scalar()
        return result or 0

    def _row_to_aggregate(self, row: dict) -> ReportAggregate:
        def _parse_json(val):
            if isinstance(val, str):
                return json.loads(val)
            return val or []

        report = TechnicalReport.from_dict({
            "components_identified": _parse_json(row.get("components_identified")),
            "architectural_risks": _parse_json(row.get("architectural_risks")),
            "recommendations": _parse_json(row.get("recommendations")),
            "executive_summary": row.get("executive_summary", ""),
            "rag_used": row.get("rag_used", False),
        })

        qa = None
        if row.get("qa_is_valid") is not None:
            qa = QAScore(
                is_valid=row["qa_is_valid"],
                completeness_score=float(row.get("qa_completeness_score") or 0),
                issues_found=_parse_json(row.get("qa_issues_found")),
                quality_notes=row.get("qa_quality_notes") or "",
            )

        aggregate = ReportAggregate(
            id=ReportId.from_string(str(row["id"])),
            analysis_id=AnalysisId.from_string(str(row["analysis_id"])),
            report=report,
            qa_score=qa,
        )
        return aggregate
