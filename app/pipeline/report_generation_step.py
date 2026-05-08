"""
Report Agent — gera relatório técnico estruturado com guardrails e RAG.

Suporta dois backends configuráveis via REPORT_MODEL_BACKEND:
  "langchain"       → LangChain + LLM configurado (padrão)
  "finetuned_api"   → LLM fine-tunado via HuggingFace Inference API
  "finetuned_local" → LLM fine-tunado carregado localmente (requer GPU)

Os guardrails (grounding check, completude, JSON schema) são compartilhados
entre todos os backends — a troca de modelo não altera as garantias de qualidade.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.infrastructure.config.settings import get_settings
from app.shared.exceptions import ReportGenerationError
from app.shared.logging import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Helpers compartilhados
# ──────────────────────────────────────────────

def _build_rag_section(rag_result: dict) -> str:
    if rag_result and rag_result.get("has_context") and rag_result.get("rag_enrichment"):
        return f"""
=== CONTEXTO DE ARQUITETURAS SIMILARES (RAG) ===
{rag_result['rag_enrichment']}

Identifique com [RAG] as recomendações influenciadas por este contexto histórico.
"""
    return "Sem contexto histórico disponível para esta análise."


def _validate_guardrails(result: dict, source_components: list) -> None:
    """
    Guardrails pós-geração — compartilhados entre todos os backends.
    Verifica grounding (alucinação de componentes) e completude mínima.
    """
    report_components = set(c.lower() for c in result.get("components_identified", []))
    source_set = set(c.lower() for c in source_components)

    if not report_components:
        raise ReportGenerationError("Guardrail: components_identified está vazio.", step="report")

    hallucinated = report_components - source_set
    if source_set and len(hallucinated) > len(source_set) * 0.2:
        raise ReportGenerationError(
            f"Guardrail: componentes não encontrados na extração: {hallucinated}",
            step="report",
        )

    if not result.get("recommendations"):
        raise ReportGenerationError("Guardrail: relatório sem recomendações.", step="report")

    summary = result.get("executive_summary", "")
    if not summary or len(summary) < 100:
        raise ReportGenerationError("Guardrail: sumário executivo insuficiente (< 100 chars).", step="report")


# ──────────────────────────────────────────────
# Backend: LangChain (padrão)
# ──────────────────────────────────────────────

RISK_CATEGORIES = "SPOF, Segurança, Escalabilidade, Acoplamento, Observabilidade, Resiliência"


def _run_with_langchain(
    extraction_result: dict,
    rag_result: dict | None,
    settings,
) -> dict:
    """Gera o relatório (com riscos embutidos) via LangChain + LLM configurado."""
    components = extraction_result.get("components", [])
    relationships = extraction_result.get("relationships", [])
    patterns = extraction_result.get("patterns", [])
    rag_section = _build_rag_section(rag_result)
    has_rag = bool(rag_result and rag_result.get("has_context"))

    llm_kwargs = {
        "model": settings.llm_model,
        "max_tokens": 8192,
        "openai_api_key": settings.openai_api_key,
        "max_retries": 6,
    }
    if settings.llm_base_url:
        llm_kwargs["openai_api_base"] = settings.llm_base_url
    llm = ChatOpenAI(**llm_kwargs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um arquiteto de software sênior gerando relatórios técnicos.
Baseie-se APENAS nos dados fornecidos. Não invente componentes ou riscos.
Use linguagem técnica em português. Retorne APENAS JSON válido.
IMPORTANTE: TODAS as chaves do JSON são OBRIGATÓRIAS. Nunca omita nenhuma chave, especialmente "recommendations"."""),
        ("human", f"""Gere um relatório técnico completo com base nos dados extraídos do diagrama:

=== COMPONENTES ===
{json.dumps(components, ensure_ascii=False)}

=== RELACIONAMENTOS ===
{json.dumps(relationships, ensure_ascii=False)}

=== PADRÕES ARQUITETURAIS ===
{json.dumps(patterns, ensure_ascii=False)}

{rag_section}

Analise os riscos arquiteturais nas categorias: {RISK_CATEGORIES}.
Inclua apenas riscos reais identificados — cada risco deve referenciar ao menos um componente existente.

ATENÇÃO: O campo "recommendations" é OBRIGATÓRIO e deve conter no mínimo 3 recomendações práticas e específicas baseadas nos riscos identificados. Nunca retorne "recommendations" vazio.

Retorne JSON com exatamente estas chaves (TODAS obrigatórias):
{{
  "components_identified": ["lista de componentes — OBRIGATÓRIO"],
  "architectural_risks": [
    {{
      "type": "uma das categorias: {RISK_CATEGORIES}",
      "description": "descrição clara do problema",
      "severity": "ALTO|MÉDIO|BAIXO",
      "affected_components": ["componentes afetados"],
      "mitigation": "recomendação de mitigação específica"
    }}
  ],
  "recommendations": ["OBRIGATÓRIO — mínimo 3 recomendações práticas baseadas nos riscos. Use [RAG] nas influenciadas pelo contexto histórico"],
  "executive_summary": "sumário executivo em até 3 parágrafos — OBRIGATÓRIO, mínimo 100 caracteres",
  "rag_used": {str(has_rag).lower()}
}}"""),
    ])

    chain = prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({})
    except Exception as exc:
        raise ReportGenerationError(f"Erro ao gerar relatório via LangChain: {exc}", step="report")

    return result


# ──────────────────────────────────────────────
# Backend: LLM Fine-Tunado
# ──────────────────────────────────────────────

def _run_with_finetuned(
    extraction_result: dict,
    risk_result: dict,
    rag_result: dict | None,
    settings,
) -> dict:
    """Gera o relatório via LLM fine-tunado (HuggingFace API ou local)."""
    from app.infrastructure.llm.finetuning.inference import get_report_client

    client = get_report_client(settings)

    try:
        result = client.generate_report(extraction_result, risk_result, rag_result)
    except Exception as exc:
        raise ReportGenerationError(
            f"Erro ao gerar relatório via LLM fine-tunado: {exc}",
            step="report",
        )

    return result


# ──────────────────────────────────────────────
# Interface pública
# ──────────────────────────────────────────────

def run(extraction_result: dict, rag_result: dict = None, risk_result: dict = None) -> dict:
    """
    Gera o relatório técnico estruturado com análise de riscos embutida.

    O backend é selecionado via REPORT_MODEL_BACKEND:
      "langchain"       → LangChain + LLM (padrão)
      "finetuned_api"   → LLM fine-tunado via HuggingFace Inference API
      "finetuned_local" → LLM fine-tunado local (requer GPU)

    Args:
        extraction_result: saída do extraction_agent
        rag_result:        saída do rag_agent (opcional)
        risk_result:       saída do risk_agent (opcional, usado pelo backend fine-tuned)

    Returns:
        dict com components_identified, architectural_risks, recommendations,
              executive_summary, rag_used
    """
    settings = get_settings()
    backend = settings.report_model_backend
    components = extraction_result.get("components", [])
    has_rag = bool(rag_result and rag_result.get("has_context"))

    logger.info(
        "report.start",
        backend=backend,
        components=len(components),
        rag=has_rag,
    )

    # ── Seleção de backend ─────────────────────────────────────────
    if backend == "langchain":
        result = _run_with_langchain(extraction_result, rag_result, settings)
    elif backend in ("finetuned_api", "finetuned_local"):
        result = _run_with_finetuned(extraction_result, risk_result or {}, rag_result, settings)
    else:
        raise ReportGenerationError(
            f"REPORT_MODEL_BACKEND inválido: '{backend}'. "
            "Valores válidos: 'langchain', 'finetuned_api', 'finetuned_local'",
            step="report",
        )

    # ── Guardrails (compartilhados entre todos os backends) ────────
    _validate_guardrails(result, components)

    logger.info(
        "report.done",
        backend=backend,
        components=len(result.get("components_identified", [])),
        recommendations=len(result.get("recommendations", [])),
        rag_used=result.get("rag_used", False),
    )
    return result
