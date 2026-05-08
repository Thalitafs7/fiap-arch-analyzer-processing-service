"""
Prompts compartilhados — single source of truth para treino, inferência e geração de dados.

Todos os módulos de fine-tuning (data_formatter, inference, data_generator)
devem importar daqui para garantir paridade entre treino e inferência.
"""

import json


RISK_CATEGORIES = "SPOF, Segurança, Escalabilidade, Acoplamento, Observabilidade, Resiliência"

SYSTEM_PROMPT = (
    "Você é um arquiteto de software sênior gerando relatórios técnicos.\n"
    "Baseie-se APENAS nos dados fornecidos. Não invente componentes ou riscos.\n"
    "Use linguagem técnica em português. Retorne APENAS JSON válido.\n"
    'IMPORTANTE: TODAS as chaves do JSON são OBRIGATÓRIAS. Nunca omita nenhuma chave, especialmente "recommendations".'
)


def build_user_message(
    extraction: dict,
    risks: dict,
    rag_result: dict | None = None,
) -> str:
    """
    Constrói a mensagem do usuário para geração de relatório.

    Usado em: data_formatter, inference, data_generator.
    Inclui componentes, relacionamentos, padrões, riscos e contexto RAG.
    """
    components = extraction.get("components", [])
    relationships = extraction.get("relationships", [])
    patterns = extraction.get("patterns", [])
    risk_list = risks.get("risks", [])
    severity = risks.get("severity_summary", {"high": 0, "medium": 0, "low": 0})
    has_rag = bool(
        rag_result
        and rag_result.get("has_context")
        and rag_result.get("rag_enrichment")
    )

    rag_section = (
        f"=== CONTEXTO DE ARQUITETURAS SIMILARES (RAG) ===\n{rag_result['rag_enrichment']}\n"
        "Identifique com [RAG] as recomendações influenciadas por este contexto histórico."
        if has_rag
        else "Sem contexto histórico disponível para esta análise."
    )

    return f"""Gere um relatório técnico completo com base nos dados extraídos do diagrama:

=== COMPONENTES ===
{json.dumps(components, ensure_ascii=False)}

=== RELACIONAMENTOS ===
{json.dumps(relationships, ensure_ascii=False)}

=== PADRÕES ARQUITETURAIS ===
{json.dumps(patterns, ensure_ascii=False)}

=== RISCOS IDENTIFICADOS ===
{json.dumps(risk_list, ensure_ascii=False)}

=== SEVERIDADE ===
Alto: {severity.get('high', 0)} | Médio: {severity.get('medium', 0)} | Baixo: {severity.get('low', 0)}

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
}}"""
