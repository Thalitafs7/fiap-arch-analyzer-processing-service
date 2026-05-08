import json
import pytest
from unittest.mock import patch, MagicMock
from app.pipeline.component_extraction_step import run
from app.shared.exceptions import ExtractionError

INGESTION_RESULT = {
    "file_name": "diagrama.png",
    "file_type": "png",
    "media_type": "image/png",
    "content_base64": "aGVsbG8=",
}

VALID_EXTRACTION = {
    "components": ["API Gateway", "Lambda", "S3", "SQS", "PostgreSQL"],
    "relationships": ["API Gateway → Lambda: invoca", "Lambda → S3: armazena"],
    "patterns": ["Event-driven", "Serverless"],
    "raw_description": "Arquitetura serverless com API Gateway acionando Lambda que persiste em S3.",
}


def _mock_response(text: str):
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


@patch("app.pipeline.component_extraction_step.anthropic.Anthropic")
def test_extraction_valid(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_response(
        json.dumps(VALID_EXTRACTION)
    )

    result = run(INGESTION_RESULT)

    assert result["status"] == "em_processamento"
    assert "API Gateway" in result["components"]
    assert len(result["relationships"]) > 0
    assert result["raw_description"]


@patch("app.pipeline.component_extraction_step.anthropic.Anthropic")
def test_extraction_invalid_json(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _mock_response("não é json")

    with pytest.raises(ExtractionError, match="JSON inválido"):
        run(INGESTION_RESULT)


@patch("app.pipeline.component_extraction_step.anthropic.Anthropic")
def test_extraction_empty_components(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    empty = {**VALID_EXTRACTION, "components": []}
    mock_client.messages.create.return_value = _mock_response(json.dumps(empty))

    with pytest.raises(ExtractionError, match="Nenhum componente"):
        run(INGESTION_RESULT)


@patch("app.pipeline.component_extraction_step.anthropic.Anthropic")
def test_extraction_strips_markdown_fences(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    fenced = f"```json\n{json.dumps(VALID_EXTRACTION)}\n```"
    mock_client.messages.create.return_value = _mock_response(fenced)

    result = run(INGESTION_RESULT)
    assert result["components"]
