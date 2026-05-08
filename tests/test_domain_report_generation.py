"""Testes unitários para o Bounded Context de geração de relatório."""

import pytest

from app.domain.shared.analysis_id import AnalysisId
from app.domain.shared.report_id import ReportId
from app.domain.report_generation.risk import RiskCategory, RiskItem, Severity
from app.domain.report_generation.qa_score import QAScore
from app.domain.report_generation.recommendation import Recommendation
from app.domain.report_generation.technical_report import TechnicalReport
from app.domain.report_generation.report import ReportAggregate


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class TestSeverity:
    @pytest.mark.parametrize("raw,expected", [
        ("ALTO", Severity.HIGH),
        ("alto", Severity.HIGH),
        ("MÉDIO", Severity.MEDIUM),
        ("médio", Severity.MEDIUM),
        ("BAIXO", Severity.LOW),
        ("desconhecido", Severity.LOW),
    ])
    def test_from_string(self, raw, expected):
        assert Severity.from_string(raw) == expected


# ---------------------------------------------------------------------------
# RiskItem
# ---------------------------------------------------------------------------

class TestRiskItem:
    BASE = {
        "type": "SPOF",
        "description": "Lambda sem concorrência reservada.",
        "severity": "ALTO",
        "affected_components": ["Lambda"],
        "mitigation": "Configurar reserved concurrency.",
    }

    def test_from_dict(self):
        r = RiskItem.from_dict(self.BASE)
        assert r.risk_category == RiskCategory.SPOF
        assert r.severity == Severity.HIGH
        assert "Lambda" in r.affected_components

    def test_to_dict_roundtrip(self):
        r = RiskItem.from_dict(self.BASE)
        d = r.to_dict()
        assert d["type"] == "SPOF"
        assert d["severity"] == "ALTO"
        assert d["mitigation"] == self.BASE["mitigation"]

    def test_unknown_category_falls_back_to_resilience(self):
        data = {**self.BASE, "type": "DESCONHECIDO"}
        with pytest.raises(ValueError):
            RiskItem.from_dict(data)

    def test_unknown_severity_defaults_to_low(self):
        data = {**self.BASE, "severity": "NENHUM"}
        r = RiskItem.from_dict(data)
        assert r.severity == Severity.LOW

    def test_frozen(self):
        r = RiskItem.from_dict(self.BASE)
        with pytest.raises((AttributeError, TypeError)):
            r.description = "outro"


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_plain_text(self):
        rec = Recommendation.from_string("Adicionar DLQ no SQS.")
        assert rec.text == "Adicionar DLQ no SQS."
        assert rec.rag_influenced is False

    def test_rag_influenced_detected(self):
        rec = Recommendation.from_string("[RAG] Adicionar circuit breaker.")
        assert rec.rag_influenced is True

    def test_str_returns_text(self):
        rec = Recommendation.from_string("Usar cache")
        assert str(rec) == "Usar cache"


# ---------------------------------------------------------------------------
# QAScore
# ---------------------------------------------------------------------------

class TestQAScore:
    def test_from_dict_valid(self):
        qa = QAScore.from_dict({
            "is_valid": True,
            "completeness_score": 0.9,
            "issues_found": [],
            "quality_notes": "Ótimo.",
        })
        assert qa.is_valid is True
        assert qa.completeness_score == 0.9

    def test_to_dict_status_analisado(self):
        qa = QAScore(is_valid=True, completeness_score=0.8, issues_found=[], quality_notes="")
        assert qa.to_dict()["status"] == "analisado"

    def test_to_dict_status_erro(self):
        qa = QAScore(is_valid=False, completeness_score=0.4, issues_found=["faltam riscos"], quality_notes="")
        assert qa.to_dict()["status"] == "erro"

    def test_from_dict_defaults(self):
        qa = QAScore.from_dict({})
        assert qa.is_valid is False
        assert qa.completeness_score == 0.0

    def test_min_score_constant(self):
        assert QAScore.MIN_SCORE == 0.6


# ---------------------------------------------------------------------------
# TechnicalReport
# ---------------------------------------------------------------------------

class TestTechnicalReport:
    REPORT_DICT = {
        "components_identified": ["API Gateway", "Lambda", "SQS"],
        "architectural_risks": [
            {
                "type": "SPOF",
                "description": "Lambda sem concorrência.",
                "severity": "ALTO",
                "affected_components": ["Lambda"],
                "mitigation": "Reserved concurrency.",
            },
            {
                "type": "Escalabilidade",
                "description": "DB sem read replica.",
                "severity": "MÉDIO",
                "affected_components": ["PostgreSQL"],
                "mitigation": "Adicionar replica.",
            },
        ],
        "recommendations": ["Configurar DLQ.", "[RAG] Adicionar circuit breaker."],
        "executive_summary": "Resumo executivo da arquitetura analisada.",
        "rag_used": True,
    }

    def test_from_dict(self):
        r = TechnicalReport.from_dict(self.REPORT_DICT)
        assert len(r.components_identified) == 3
        assert len(r.architectural_risks) == 2
        assert len(r.recommendations) == 2
        assert r.rag_used is True

    def test_risk_severity_summary(self):
        r = TechnicalReport.from_dict(self.REPORT_DICT)
        summary = r.risk_severity_summary
        assert summary["high"] == 1
        assert summary["medium"] == 1
        assert summary["low"] == 0

    def test_to_dict_roundtrip(self):
        r = TechnicalReport.from_dict(self.REPORT_DICT)
        d = r.to_dict()
        assert "components_identified" in d
        assert "architectural_risks" in d
        assert "recommendations" in d
        assert d["rag_used"] is True

    def test_rag_influenced_recommendations_detected(self):
        r = TechnicalReport.from_dict(self.REPORT_DICT)
        rag_recs = [rec for rec in r.recommendations if rec.rag_influenced]
        assert len(rag_recs) == 1

    def test_from_dict_empty(self):
        r = TechnicalReport.from_dict({})
        assert r.components_identified == []
        assert r.architectural_risks == []
        assert r.rag_used is False


# ---------------------------------------------------------------------------
# ReportAggregate
# ---------------------------------------------------------------------------

class TestReportAggregate:
    def _make(self):
        return ReportAggregate.create(
            report_id=ReportId.generate(),
            analysis_id=AnalysisId.generate(),
        )

    def _report(self):
        return TechnicalReport.from_dict({
            "components_identified": ["Lambda"],
            "architectural_risks": [
                {
                    "type": "SPOF",
                    "description": "desc",
                    "severity": "ALTO",
                    "affected_components": ["Lambda"],
                    "mitigation": "fix",
                }
            ],
            "recommendations": ["Rec 1"],
            "executive_summary": "Resumo.",
            "rag_used": False,
        })

    def _qa(self, valid=True, score=0.85):
        return QAScore(
            is_valid=valid,
            completeness_score=score,
            issues_found=[],
            quality_notes="ok",
        )

    def test_create_has_no_report(self):
        agg = self._make()
        assert agg.report is None
        assert agg.qa_score is None

    def test_attach_report_emits_event(self):
        agg = self._make()
        agg.attach_report(self._report())
        events = agg.pull_events()
        assert any(e.event_name == "ReportGeneratedEvent" for e in events)

    def test_attach_qa_emits_event(self):
        agg = self._make()
        agg.attach_report(self._report())
        agg.pull_events()
        agg.attach_qa(self._qa())
        events = agg.pull_events()
        assert any(e.event_name == "QAValidationCompletedEvent" for e in events)

    def test_is_valid_requires_qa(self):
        agg = self._make()
        agg.attach_report(self._report())
        assert agg.is_valid is False

    def test_is_valid_with_valid_qa(self):
        agg = self._make()
        agg.attach_report(self._report())
        agg.attach_qa(self._qa(valid=True))
        assert agg.is_valid is True

    def test_is_valid_with_invalid_qa(self):
        agg = self._make()
        agg.attach_report(self._report())
        agg.attach_qa(self._qa(valid=False))
        assert agg.is_valid is False

    def test_to_persistence_dict_without_report_raises(self):
        agg = self._make()
        with pytest.raises(ValueError, match="sem conteúdo"):
            agg.to_persistence_dict()

    def test_to_persistence_dict_with_report_and_qa(self):
        agg = self._make()
        agg.attach_report(self._report())
        agg.attach_qa(self._qa())
        d = agg.to_persistence_dict()
        assert "components_identified" in d
        assert d["qa_is_valid"] is True
        assert d["qa_completeness_score"] == 0.85

    def test_pull_events_clears(self):
        agg = self._make()
        agg.attach_report(self._report())
        agg.pull_events()
        assert agg.pull_events() == []
