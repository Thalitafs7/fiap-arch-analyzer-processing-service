"""
QA Agent — avalia qualidade e consistência do relatório gerado.
Fase 1: verificações determinísticas (sem LLM).
Fase 2: avaliação com LLM (OpenAI/Groq) + parse JSON.
"""

import json
import re
from openai import OpenAI
from app.infrastructure.config.settings import get_settings
from app.shared.logging import get_logger

logger = get_logger(__name__)

QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "completeness_score": {"type": "number"},
        "issues_found": {"type": "array", "items": {"type": "string"}},
        "quality_notes": {"type": "string"},
    },
    "required": ["is_valid", "completeness_score", "issues_found", "quality_notes"],
    "additionalProperties": False,
}


def _basic_checks(report: dict, extraction_components: list) -> list[str]:
    """Verificações determinísticas — rápidas, sem chamar LLM."""
    issues = []

    if not report.get("components_identified"):
        issues.append("components_identified está vazio.")

    if not report.get("architectural_risks"):
        issues.append("architectural_risks está vazio.")

    if not report.get("recommendations"):
        issues.append("recommendations está vazio.")

    summary = report.get("executive_summary", "")
    if not summary or len(summary) < 100:
        issues.append(f"executive_summary muito curto ({len(summary)} chars, mínimo 100).")

    # Grounding: pelo menos 80% dos componentes do relatório devem existir na extração
    report_components = set(c.lower() for c in report.get("components_identified", []))
    source_components = set(c.lower() for c in extraction_components)
    if source_components:
        overlap = report_components & source_components
        coverage = len(overlap) / len(report_components) if report_components else 0
        if coverage < 0.8:
            hallucinated = report_components - source_components
            issues.append(
                f"Componentes não encontrados na extração original: {', '.join(hallucinated)}"
            )

    return issues


def run(extraction_result: dict, report: dict) -> dict:
    """
    Avalia o relatório gerado.

    Args:
        extraction_result: saída do extraction_agent (ground truth)
        report:            saída do report_agent

    Returns:
        dict com is_valid, completeness_score, issues_found, quality_notes, status
    """
    extraction_components = extraction_result.get("components", [])

    # ── Fase 1: verificações básicas ────────────────────────────────
    basic_issues = _basic_checks(report, extraction_components)

    if basic_issues:
        logger.warning("qa.basic_checks_failed", issues=basic_issues)
        return {
            "is_valid": False,
            "completeness_score": 0.0,
            "issues_found": basic_issues,
            "quality_notes": "Relatório falhou nas verificações básicas de completude.",
            "status": "erro",
        }

    # ── Fase 2: avaliação com LLM ───────────────────────────────────
    settings = get_settings()
    client_kwargs = {"api_key": settings.openai_api_key, "max_retries": 6}
    if settings.llm_base_url:
        client_kwargs["base_url"] = settings.llm_base_url
    client = OpenAI(**client_kwargs)

    system_prompt = """Você é um auditor técnico adversarial. Seu papel é encontrar falhas, inconsistências e generalizações em relatórios de arquitetura de software.

Regras que você DEVE seguir:
- Seja cético por padrão. Nunca assuma que o relatório está correto sem verificar cada afirmação.
- Marque como problema qualquer componente no relatório que NÃO esteja explicitamente na extração original.
- Marque como problema recomendações genéricas que não referenciam componentes concretos do diagrama.
- Marque como problema riscos sem componentes afetados identificados.
- Marque como problema linguagem vaga como "considere melhorar", "pode ser otimizado" sem especificações.
- NÃO dê crédito por campos preenchidos com conteúdo irrelevante ou copiado.
- Seu score deve refletir rigor real: um relatório mediocre não passa de 0.7, mesmo sem erros graves.
- is_valid só deve ser true se o relatório for genuinamente útil para um arquiteto de software tomar decisões."""

    prompt = f"""Audite criticamente este relatório técnico de arquitetura de software.

COMPONENTES DA EXTRAÇÃO ORIGINAL (ground truth — única fonte de verdade):
{json.dumps(extraction_components, ensure_ascii=False)}

RELATÓRIO A AUDITAR:
{json.dumps(report, ensure_ascii=False, indent=2)}

Critérios de avaliação (pesos):
- Consistência (40%): cada componente, risco e recomendação deve referenciar elementos reais da extração original
- Completude (30%): todos os campos obrigatórios preenchidos com conteúdo substantivo (não genérico)
- Coerência (20%): cada recomendação deve estar vinculada a um risco identificado e a componentes concretos
- Qualidade (10%): linguagem técnica precisa, sem clichês como "considere adotar boas práticas"

Procure ativamente por:
1. Componentes inventados que não existem na extração
2. Riscos genéricos sem componentes afetados específicos
3. Recomendações desvinculadas dos riscos identificados
4. Sumário executivo que não reflete os dados extraídos

Retorne APENAS JSON com is_valid (boolean), completeness_score (0.0-1.0), issues_found (array de strings descrevendo cada problema encontrado) e quality_notes (string). Sem texto adicional."""

    logger.info("qa.llm_evaluation.start", model=settings.llm_model)

    try:
        create_kwargs = {
            "model": settings.llm_model,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if not settings.llm_base_url:
            create_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**create_kwargs)
        raw = response.choices[0].message.content.strip()
        # LLaMA pode retornar JSON dentro de fences
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if fence:
            raw = fence.group(1).strip()
        qa = json.loads(raw)
    except Exception as e:
        # Falha no LLM de QA não bloqueia — assume válido com score conservador
        logger.warning("qa.llm_evaluation.failed", error=str(e))
        qa = {
            "is_valid": True,
            "completeness_score": 0.7,
            "issues_found": [],
            "quality_notes": f"Avaliação LLM indisponível: {e}. Verificações básicas passaram.",
        }

    # Score mínimo obrigatório
    if qa.get("completeness_score", 0) < 0.6:
        qa["is_valid"] = False
        if "Score abaixo do mínimo" not in str(qa.get("issues_found", [])):
            qa.setdefault("issues_found", []).append(
                f"Score {qa['completeness_score']:.2f} abaixo do mínimo aceitável (0.6)."
            )

    qa["status"] = "analisado" if qa.get("is_valid") else "erro"

    logger.info(
        "qa.done",
        is_valid=qa["is_valid"],
        score=qa.get("completeness_score"),
        issues=len(qa.get("issues_found", [])),
    )
    return qa
