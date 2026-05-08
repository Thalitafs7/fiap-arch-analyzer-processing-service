"""
Extraction Agent — usa LLM Vision para extrair
componentes, relacionamentos e padrões arquiteturais do diagrama.
Suporta OpenAI (gpt-4o) e Groq (llama-3.2-vision) via base_url configurável.
"""

import json
import re
from openai import OpenAI, APIError
from app.infrastructure.config.settings import get_settings
from app.shared.exceptions import ExtractionError
from app.shared.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """Você é um arquiteto de software sênior especializado em análise de diagramas de arquitetura.
Analise o diagrama fornecido e extraia com precisão os componentes, relacionamentos e padrões arquiteturais.
Retorne APENAS um JSON válido, sem texto adicional."""

EXTRACTION_PROMPT = """Analise este diagrama de arquitetura de software e extraia:

1. **components**: lista de todos os serviços, bancos de dados, filas, gateways e componentes visíveis
2. **relationships**: lista de relacionamentos entre componentes (formato: "ComponenteA → ComponenteB: descrição")
3. **patterns**: padrões arquiteturais identificados (ex: event-driven, microservices, CQRS, etc.)
4. **raw_description**: descrição textual completa do diagrama em português

Retorne JSON com exatamente estas chaves:
{
  "components": ["string"],
  "relationships": ["string"],
  "patterns": ["string"],
  "raw_description": "string"
}"""


def _build_messages(ingestion_result: dict) -> list:
    """Monta as mensagens para a API OpenAI."""
    media_type = ingestion_result["media_type"]
    b64 = ingestion_result["content_base64"]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                },
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        },
    ]


def run(ingestion_result: dict) -> dict:
    """
    Extrai componentes do diagrama via OpenAI Vision.

    Args:
        ingestion_result: saída do ingestion_agent (inclui content_base64, media_type, file_type)

    Returns:
        dict com status, components, relationships, patterns, raw_description
    """
    settings = get_settings()
    client_kwargs = {"api_key": settings.openai_api_key, "max_retries": 6}
    if settings.llm_base_url:
        client_kwargs["base_url"] = settings.llm_base_url
    client = OpenAI(**client_kwargs)

    vision_model = settings.llm_vision_model or settings.llm_model

    logger.info("extraction.start", file_name=ingestion_result.get("file_name"), model=vision_model)

    messages = _build_messages(ingestion_result)

    try:
        create_kwargs = {
            "model": vision_model,
            "max_tokens": 4096,
            "messages": messages,
        }
        # JSON mode só funciona com OpenAI; Groq/LLaMA não suportam
        if not settings.llm_base_url:
            create_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**create_kwargs)
    except APIError as e:
        raise ExtractionError(f"Erro na API LLM: {e}", step="extraction")

    raw_text = response.choices[0].message.content.strip()

    # LLaMA pode retornar JSON dentro de ```json ... ``` — limpar
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
    if fence_match:
        raw_text = fence_match.group(1).strip()

    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"LLM retornou JSON inválido: {e}", step="extraction")

    required_keys = {"components", "relationships", "patterns", "raw_description"}
    missing = required_keys - set(extracted.keys())
    if missing:
        raise ExtractionError(f"JSON incompleto, faltam chaves: {missing}", step="extraction")

    if not extracted["components"]:
        raise ExtractionError("Nenhum componente identificado no diagrama.", step="extraction")

    result = {
        "status": "em_processamento",
        "components": extracted["components"],
        "relationships": extracted["relationships"],
        "patterns": extracted["patterns"],
        "raw_description": extracted["raw_description"],
    }

    logger.info(
        "extraction.done",
        components_count=len(result["components"]),
        patterns_count=len(result["patterns"]),
    )
    return result
