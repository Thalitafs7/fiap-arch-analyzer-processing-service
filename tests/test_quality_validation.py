import json
import pytest
from unittest.mock import patch, MagicMock
from app.pipeline.quality_validation_step import run

EXTRACTION = {
    "components": ["API Gateway", "Lambda", "SQS", "PostgreSQL"],
}

VALID_REPORT = {
    "components_identified": ["API Gateway", "Lambda", "SQS", "PostgreSQL"],
    "architectural_risks": [
        {
            "type": "SPOF",
            "description": "Lambda sem concorrência reservada pode causar throttling.",
            "severity": "ALTO",
            "affected_components": ["Lambda"],
            "mitigation": "Configurar reserved concurrency.",
        }
    ],
    "recommendations": [
        "Configurar DLQ no SQS.",
        "Adicionar monitoramento no Lambda.",
    ],
    "executive_summary": (
        "A arquitetura analisada implementa um pipeline event-driven com API Gateway, "
        "Lambda, SQS e PostgreSQL. O principal risco identificado é a ausência de "
        "configuração de concorrência reservada no Lambda, que pode causar throttling "
        "sob alta carga. Recomenda-se implementar DLQ no SQS e monitoramento adequado."
    ),
    "rag_used": False,
}


def _mock_qa_response(data: dict):
    block = MagicMock()
    block.text = json.dumps(data)
    response = MagicMock()
    response.content = [block]
    return response


def test_qa_basic_checks_empty_components():
    bad_report = {**VALID_REPORT, "components_identified": []}
    result = run(EXTRACTION, bad_report)
    assert result["is_valid"] is False
    assert result["status"] == "erro"
    assert any("components_identified" in i for i in result["issues_found"])


def test_qa_basic_checks_short_summary():
    bad_report = {**VALID_REPORT, "executive_summary": "Curto."}
    result = run(EXTRACTION, bad_report)
    assert result["is_valid"] is False


def test_qa_basic_checks_no_recommendations():
    bad_report = {**VALID_REPORT, "recommendations": []}
    result = run(EXTRACTION, bad_report)
    assert result["is_valid"] is False


@patch("app.pipeline.quality_validation_step.anthropic.Anthropic")
def test_qa_valid_report(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_qa_response({
        "is_valid": True,
        "completeness_score": 0.92,
        "issues_found": [],
        "quality_notes": "Relatório bem estruturado.",
    })

    result = run(EXTRACTION, VALID_REPORT)
    assert result["is_valid"] is True
    assert result["completeness_score"] >= 0.6
    assert result["status"] == "analisado"


@patch("app.pipeline.quality_validation_step.anthropic.Anthropic")
def test_qa_low_score_marks_invalid(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_qa_response({
        "is_valid": True,
        "completeness_score": 0.4,
        "issues_found": [],
        "quality_notes": "Score baixo.",
    })

    result = run(EXTRACTION, VALID_REPORT)
    assert result["is_valid"] is False
    assert result["status"] == "erro"
