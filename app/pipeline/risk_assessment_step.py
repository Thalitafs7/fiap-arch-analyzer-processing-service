"""
Risk Agent — classifica riscos arquiteturais usando OpenAI gpt-4o.
Categorias: SPOF, Segurança, Escalabilidade, Acoplamento, Observabilidade, Resiliência.
"""

import json
from openai import OpenAI, APIError
from app.infrastructure.config.settings import get_settings
from app.shared.exceptions import RiskAnalysisError
from app.shared.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """Você é um arquiteto de software sênior especializado em análise de riscos.
Analise os componentes e padrões fornecidos e classifique os riscos arquiteturais.
Retorne APENAS JSON válido, sem texto adicional."""

RISK_CATEGORIES = ["SPOF", "Segurança", "Escalabilidade", "Acoplamento", "Observabilidade", "Resiliência"]


def run(extraction_result: dict, rag_result: dict = None) -> dict:
    """
    Classifica riscos arquiteturais com base na extração e contexto RAG.

    Args:
        extraction_result: saída do extraction_agent
        rag_result:        saída do rag_agent (opcional, enriquece análise)

    Returns:
        dict com status, risks[], severity_summary
    """
    settings = get_settings()
    # max_retries=6: SDK faz backoff exponencial automático em 429/5xx,
    # respeitando o header Retry-After retornado pela OpenAI.
    client = OpenAI(api_key=settings.openai_api_key, max_retries=6)

    components = extraction_result.get("components", [])
    relationships = extraction_result.get("relationships", [])
    patterns = extraction_result.get("patterns", [])

    rag_context = ""
    if rag_result and rag_result.get("has_context") and rag_result.get("rag_enrichment"):
        rag_context = f"""
=== CONTEXTO DE ARQUITETURAS SIMILARES ===
{rag_result['rag_enrichment']}
Use este contexto para identificar padrões de risco recorrentes.
"""

    prompt = f"""Analise os riscos arquiteturais com base nos dados abaixo:

=== COMPONENTES ===
{json.dumps(components, ensure_ascii=False)}

=== RELACIONAMENTOS ===
{json.dumps(relationships, ensure_ascii=False)}

=== PADRÕES IDENTIFICADOS ===
{json.dumps(patterns, ensure_ascii=False)}

{rag_context}

Categorias de risco a avaliar: {", ".join(RISK_CATEGORIES)}

Retorne JSON com exatamente estas chaves:
{{
  "risks": [
    {{
      "type": "categoria do risco",
      "description": "descrição clara do problema",
      "severity": "ALTO|MÉDIO|BAIXO",
      "affected_components": ["lista de componentes afetados"],
      "mitigation": "recomendação de mitigação específica"
    }}
  ],
  "severity_summary": {{
    "high": 0,
    "medium": 0,
    "low": 0
  }}
}}

Regras:
- Inclua apenas riscos reais identificados nos dados — não invente.
- Cada risco deve referenciar ao menos um componente existente.
- severity_summary deve ser a contagem dos riscos por severidade."""

    logger.info("risk.start", components_count=len(components))

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
    except APIError as e:
        raise RiskAnalysisError(f"Erro na API OpenAI: {e}", step="risk")

    raw_text = response.choices[0].message.content.strip()

    if not raw_text:
        raise RiskAnalysisError("LLM não retornou texto na análise de riscos.", step="risk")

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RiskAnalysisError(f"JSON inválido retornado pelo LLM: {e}", step="risk")

    risks = parsed.get("risks", [])
    severity_summary = parsed.get("severity_summary", {"high": 0, "medium": 0, "low": 0})

    # Recalcula severity_summary para garantir consistência
    severity_summary = {
        "high": sum(1 for r in risks if r.get("severity", "").upper() == "ALTO"),
        "medium": sum(1 for r in risks if r.get("severity", "").upper() == "MÉDIO"),
        "low": sum(1 for r in risks if r.get("severity", "").upper() == "BAIXO"),
    }

    result = {
        "status": "em_processamento",
        "risks": risks,
        "severity_summary": severity_summary,
    }

    logger.info(
        "risk.done",
        risks_count=len(risks),
        high=severity_summary["high"],
        medium=severity_summary["medium"],
        low=severity_summary["low"],
    )
    return result
