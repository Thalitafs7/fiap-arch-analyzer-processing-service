"""
Infrastructure Layer — SQLAlchemy implementation of IAnalysisRepository.
Traduz entre o aggregate de domínio e as tabelas do banco.
"""

from __future__ import annotations
import json
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.domain.diagram_analysis.analysis import AnalysisAggregate
from app.domain.diagram_analysis.repository import IAnalysisRepository
from app.domain.diagram_analysis.analysis_status import AnalysisStatus
from app.domain.shared.analysis_id import AnalysisId
from app.shared.logging import get_logger

logger = get_logger(__name__)


class SQLAlchemyAnalysisRepository(IAnalysisRepository):
    """Repositório concreto de AnalysisAggregate usando SQLAlchemy (raw SQL)."""

    def __init__(self, db: Session):
        self._db = db

    def save(self, analysis: AnalysisAggregate) -> None:
        """Upsert: insere ou atualiza dependendo da existência no banco."""
        existing = self._db.execute(
            text("SELECT id FROM analyses WHERE id = :id"),
            {"id": str(analysis.id)},
        ).first()

        if existing:
            self._update(analysis)
        else:
            self._insert(analysis)

    def _insert(self, analysis: AnalysisAggregate) -> None:
        self._db.execute(
            text("""
                INSERT INTO analyses (id, status, file_name, file_type, s3_key, sqs_message_id)
                VALUES (:id, :status, :file_name, :file_type, :s3_key, :sqs_message_id)
            """),
            {
                "id": str(analysis.id),
                "status": analysis.status.value,
                "file_name": analysis.file_name,
                "file_type": analysis.file_type,
                "s3_key": analysis.s3_key,
                "sqs_message_id": analysis.sqs_message_id,
            },
        )
        self._db.commit()
        logger.info("analysis.inserted", analysis_id=str(analysis.id))

        # Persiste extraction_result se já disponível
        if analysis.extraction_result:
            self._save_extraction(analysis)

    def _update(self, analysis: AnalysisAggregate) -> None:
        self._db.execute(
            text("""
                UPDATE analyses
                SET status = :status, error_message = :error_message
                WHERE id = :id
            """),
            {
                "id": str(analysis.id),
                "status": analysis.status.value,
                "error_message": analysis.error_message,
            },
        )
        self._db.commit()

        if analysis.extraction_result:
            existing_extraction = self._db.execute(
                text("SELECT id FROM extraction_results WHERE analysis_id = :id"),
                {"id": str(analysis.id)},
            ).first()
            if not existing_extraction:
                self._save_extraction(analysis)

    def _save_extraction(self, analysis: AnalysisAggregate) -> None:
        import uuid
        extraction = analysis.extraction_result
        self._db.execute(
            text("""
                INSERT INTO extraction_results (id, analysis_id, components, relationships, patterns, raw_description)
                VALUES (:id, :analysis_id, :components, :relationships, :patterns, :raw_description)
            """),
            {
                "id": str(uuid.uuid4()),
                "analysis_id": str(analysis.id),
                "components": json.dumps(extraction.component_names, ensure_ascii=False),
                "relationships": json.dumps(
                    [str(r) for r in extraction.relationships], ensure_ascii=False
                ),
                "patterns": json.dumps(
                    [str(p) for p in extraction.patterns], ensure_ascii=False
                ),
                "raw_description": extraction.raw_description,
            },
        )
        self._db.commit()

    def update_status(self, analysis: AnalysisAggregate) -> None:
        self._db.execute(
            text("""
                UPDATE analyses
                SET status = :status, error_message = :error_message
                WHERE id = :id
            """),
            {
                "id": str(analysis.id),
                "status": analysis.status.value,
                "error_message": analysis.error_message,
            },
        )
        self._db.commit()
        logger.info(
            "analysis.status_updated",
            analysis_id=str(analysis.id),
            status=analysis.status.value,
        )

    def get_by_id(self, analysis_id: AnalysisId) -> Optional[AnalysisAggregate]:
        row = self._db.execute(
            text("SELECT * FROM analyses WHERE id = :id"),
            {"id": str(analysis_id)},
        ).mappings().first()

        if not row:
            return None

        return self._row_to_aggregate(dict(row))

    def get_by_sqs_message_id(self, sqs_message_id: str) -> Optional[AnalysisAggregate]:
        row = self._db.execute(
            text("SELECT * FROM analyses WHERE sqs_message_id = :msg_id LIMIT 1"),
            {"msg_id": sqs_message_id},
        ).mappings().first()

        if not row:
            return None

        return self._row_to_aggregate(dict(row))

    def _row_to_aggregate(self, row: dict) -> AnalysisAggregate:
        return AnalysisAggregate(
            id=AnalysisId.from_string(str(row["id"])),
            status=AnalysisStatus(row["status"]),
            file_name=row["file_name"],
            file_type=row["file_type"],
            s3_key=row.get("s3_key"),
            sqs_message_id=row.get("sqs_message_id"),
            error_message=row.get("error_message"),
        )
