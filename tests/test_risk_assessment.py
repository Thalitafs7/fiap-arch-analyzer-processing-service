import json
import pytest
from unittest.mock import patch, MagicMock
from app.pipeline.risk_assessment_step import run
from app.shared.exceptions import RiskAnalysisError

EXTRACTION = {
    "components": ["API Gateway", "Lambda", "SQS", "PostgreSQL"],
    "relationships": ["API Gateway → Lambda: invoca"],
    "patterns": ["Event-driven"],
    "raw_description": "Pipeline serverless.",
}

VALID_RISKS = {
    "risks": [
        {
            "type": "SPOF",
            "description": "Lambda sem concorrência reservada.",
            "severity": "ALTO",
            "affected_components": ["Lambda"],
            "mitigation": "Configurar reserved concurrency.",
        }
    ],
    "severity_summary": {"high": 1, "medium": 0, "low": 0},
}


def _mock_response(text: str):
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    response = MagicMock()
    response.content = [text_block]
    return response


@patch("app.pipeline.risk_assessment_step.anthropic.Anthropic")
def test_risk_valid(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_response(json.dumps(VALID_RISKS))

    result = run(EXTRACTION)

    assert result["status"] == "em_processamento"
    assert len(result["risks"]) == 1
    assert result["severity_summary"]["high"] == 1


@patch("app.pipeline.risk_assessment_step.anthropic.Anthropic")
def test_risk_severity_recalculated(mock_cls):
    """severity_summary deve ser recalculado a partir dos riscos, não confiar no LLM."""
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    wrong_summary = {**VALID_RISKS, "severity_summary": {"high": 99, "medium": 0, "low": 0}}
    mock_client.messages.create.return_value = _mock_response(json.dumps(wrong_summary))

    result = run(EXTRACTION)
    assert result["severity_summary"]["high"] == 1  # recalculado


@patch("app.pipeline.risk_assessment_step.anthropic.Anthropic")
def test_risk_with_rag_context(mock_cls):
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_response(json.dumps(VALID_RISKS))

    rag = {"has_context": True, "rag_enrichment": "Risco recorrente: ausência de DLQ."}
    result = run(EXTRACTION, rag_result=rag)
    assert result["risks"]
