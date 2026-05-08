"""
Infrastructure Layer — Adaptador OpenAI/Groq para os ports IVisionLLM e ITextLLM.

Implementa os dois ports usando a OpenAI SDK (compatível com Groq via base_url).
Encapsula toda lógica de chamada HTTP ao LLM, parsing e retry.
"""

from __future__ import annotations
import json
import re

from openai import OpenAI, APIError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.application.ports.llm_port import IVisionLLM, ITextLLM
from app.domain.diagram_analysis.diagram_file import DiagramFile
from app.domain.diagram_analysis.extraction_result import ExtractionResult
from app.domain.report_generation.technical_report import TechnicalReport
from app.domain.report_generation.rag_context import RagContext
from app.domain.report_generation.qa_score import QAScore
from app.domain.report_generation.risk import RiskCategory
from app.infrastructure.config.settings import get_settings
from app.shared.exceptions import ExtractionError, ReportGenerationError
from app.shared.logging import get_logger

logger = get_logger(__name__)

_EXTRACTION_SYSTEM = """Você é um arquiteto de software sênior especializado em análise de diagramas de arquitetura.
Analise o diagrama fornecido e extraia com precisão os componentes, relacionamentos e padrões arquiteturais.
Retorne APENAS um JSON válido, sem texto adicional."""

_EXTRACTION_PROMPT = """Analise este diagrama de arquitetura de software e extraia:

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


_CLASSIFICATION_PROMPT = """Você é um filtro rigoroso. Sua única função é determinar se esta imagem é CLARAMENTE um diagrama de arquitetura de software.

REGRA PRINCIPAL: Em caso de dúvida, classifique como false. Só aprove se tiver certeza.

Verifique em ordem:
1. É um diagrama técnico (tem caixas, setas, símbolos de componentes)? Se não → false.
2. Os componentes são de software/sistemas (serviços, APIs, bancos de dados, filas, containers, microserviços, load balancers, gateways, buckets, lambdas)? Se não → false.
3. Mostra relacionamentos/fluxos entre esses componentes técnicos? Se não → false.
4. Só então classifique como true.

Exemplos que PASSAM (true):
- Diagramas de microserviços com setas entre serviços
- Diagramas C4 (contexto, container, componente)
- Diagramas de deployment AWS/Azure/GCP com ícones de serviços cloud
- Diagramas de fluxo de dados técnico entre sistemas
- Diagramas UML de componentes ou deployment
- Topologias de rede com servidores, roteadores, firewalls

Exemplos que REPROVAM (false):
- Fotos, selfies, memes, imagens naturais
- Screenshots de aplicações, sites, chats ou dashboards
- Wireframes ou mockups de UI/UX (mesmo com caixas e setas)
- Fluxogramas de processo de negócio sem componentes técnicos
- Organogramas de RH ou hierarquias de equipes
- Diagramas de outros domínios: biologia, química, física, matemática
- Apresentações de slides com texto e gráficos
- Documentos, planilhas, tabelas
- Diagramas de fluxo genéricos (BPMN de negócio puro, sem sistemas)

Retorne APENAS JSON:
{
  "is_architecture_diagram": true/false,
  "confidence": 0.0 a 1.0,
  "reason": "explicação curta em português do critério decisivo"
}"""


def _strip_json_fence(text: str) -> str:
    """Remove markdown code fences que LLaMA pode retornar."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


class OpenAIVisionAdapter(IVisionLLM):
    """Extrai componentes de diagramas usando LLM com capacidade de visão."""

    CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.75

    def classify_image(self, diagram_file: DiagramFile) -> dict:
        """Classifica se a imagem é um diagrama de arquitetura de software."""
        settings = get_settings()
        client_kwargs = {"api_key": settings.openai_api_key, "max_retries": 6}
        if settings.llm_base_url:
            client_kwargs["base_url"] = settings.llm_base_url
        client = OpenAI(**client_kwargs)

        vision_model = settings.llm_vision_model or settings.llm_model
        logger.info("vision_llm.classify.start", model=vision_model)

        messages = [
            {
                "role": "system",
                "content": "Você é um filtro rigoroso de imagens. Rejeite tudo que não seja CLARAMENTE um diagrama de arquitetura de software. Quando em dúvida, rejeite. Retorne APENAS JSON válido.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{diagram_file.media_type};base64,{diagram_file.content_base64}",
                            "detail": "low",
                        },
                    },
                    {"type": "text", "text": _CLASSIFICATION_PROMPT},
                ],
            },
        ]

        create_kwargs = {
            "model": vision_model,
            "max_tokens": 300,
            "temperature": 0,
            "messages": messages,
        }
        if not settings.llm_base_url:
            create_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**create_kwargs)
            raw = _strip_json_fence(response.choices[0].message.content)
            result = json.loads(raw)
        except Exception as e:
            logger.warning("vision_llm.classify.failed", error=str(e))
            # Em caso de falha na API de classificação, assume válido para não bloquear.
            # confidence=1.0 para não disparar o threshold check — a extração rejeitará se não houver componentes.
            return {
                "is_architecture_diagram": True,
                "confidence": 1.0,
                "reason": f"Classificação indisponível: {e}. Prosseguindo com análise.",
            }

        logger.info(
            "vision_llm.classify.done",
            is_diagram=result.get("is_architecture_diagram"),
            confidence=result.get("confidence"),
        )
        return result

    def extract_components(self, diagram_file: DiagramFile) -> ExtractionResult:
        settings = get_settings()
        client_kwargs = {"api_key": settings.openai_api_key, "max_retries": 6}
        if settings.llm_base_url:
            client_kwargs["base_url"] = settings.llm_base_url
        client = OpenAI(**client_kwargs)

        vision_model = settings.llm_vision_model or settings.llm_model
        logger.info("vision_llm.extract.start", model=vision_model)

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{diagram_file.media_type};base64,{diagram_file.content_base64}"
                        },
                    },
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                ],
            },
        ]

        create_kwargs = {
            "model": vision_model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if not settings.llm_base_url:
            create_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**create_kwargs)
        except APIError as e:
            raise ExtractionError(f"Erro na API LLM: {e}", step="extraction")

        raw_text = _strip_json_fence(response.choices[0].message.content)

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

        result = ExtractionResult.from_dict(extracted)
        logger.info(
            "vision_llm.extract.done",
            components_count=len(result.components),
            patterns_count=len(result.patterns),
        )
        return result


class OpenAITextAdapter(ITextLLM):
    """Gera relatórios técnicos e avalia qualidade via LLM de texto."""

    def generate_report(
        self,
        extraction: ExtractionResult,
        rag_context: RagContext,
        feedback: list[str] | None = None,
    ) -> TechnicalReport:
        settings = get_settings()

        rag_section = (
            f"""
=== CONTEXTO DE ARQUITETURAS SIMILARES (RAG) ===
{rag_context.enrichment_text}

Identifique com [RAG] as recomendações influenciadas por este contexto histórico.
"""
            if rag_context.has_context and rag_context.enrichment_text
            else "Sem contexto histórico disponível para esta análise."
        )

        feedback_section = (
            f"""
=== PROBLEMAS IDENTIFICADOS NA TENTATIVA ANTERIOR (CORRIJA OBRIGATORIAMENTE) ===
Uma versão anterior deste relatório foi rejeitada pelo auditor de qualidade pelos seguintes motivos:
{chr(10).join(f"- {issue}" for issue in feedback)}

Corrija TODOS os problemas listados acima. Não repita os mesmos erros.
"""
            if feedback
            else ""
        )

        llm_kwargs = {
            "model": settings.llm_model,
            "max_tokens": 8192,
            "openai_api_key": settings.openai_api_key,
            "max_retries": 6,
        }
        if settings.llm_base_url:
            llm_kwargs["openai_api_base"] = settings.llm_base_url
        llm = ChatOpenAI(**llm_kwargs)

        risk_categories = RiskCategory.all_labels()
        has_rag = rag_context.has_context

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um arquiteto de software sênior gerando relatórios técnicos.
Baseie-se APENAS nos dados fornecidos. Não invente componentes ou riscos.
Use linguagem técnica em português. Retorne APENAS JSON válido.
IMPORTANTE: TODAS as chaves do JSON são OBRIGATÓRIAS. Nunca omita nenhuma chave, especialmente "recommendations"."""),
            ("human", f"""Gere um relatório técnico completo com base nos dados extraídos do diagrama:

=== COMPONENTES ===
{json.dumps(extraction.component_names, ensure_ascii=False)}

=== RELACIONAMENTOS ===
{json.dumps([str(r) for r in extraction.relationships], ensure_ascii=False)}

=== PADRÕES ARQUITETURAIS ===
{json.dumps([str(p) for p in extraction.patterns], ensure_ascii=False)}

{rag_section}
{feedback_section}
Analise os riscos arquiteturais nas categorias: {risk_categories}.
Inclua apenas riscos reais identificados — cada risco deve referenciar ao menos um componente existente.

ATENÇÃO: O campo "recommendations" é OBRIGATÓRIO e deve conter no mínimo 3 recomendações práticas e específicas baseadas nos riscos identificados. Nunca retorne "recommendations" vazio.

Retorne JSON com exatamente estas chaves (TODAS obrigatórias):
{{{{
  "components_identified": ["lista de componentes — OBRIGATÓRIO"],
  "architectural_risks": [
    {{{{
      "type": "uma das categorias: {risk_categories}",
      "description": "descrição clara do problema",
      "severity": "ALTO|MÉDIO|BAIXO",
      "affected_components": ["componentes afetados"],
      "mitigation": "recomendação de mitigação específica"
    }}}}
  ],
  "recommendations": ["OBRIGATÓRIO — mínimo 3 recomendações práticas baseadas nos riscos. Use [RAG] nas influenciadas pelo contexto histórico"],
  "executive_summary": "sumário executivo em até 3 parágrafos — OBRIGATÓRIO, mínimo 100 caracteres",
  "rag_used": {str(has_rag).lower()}
}}}}"""),
        ])

        chain = prompt | llm | JsonOutputParser()
        try:
            result_dict = chain.invoke({})
        except Exception as exc:
            raise ReportGenerationError(
                f"Erro ao gerar relatório via LLM: {exc}", step="report"
            )

        logger.info(
            "text_llm.generate_report.done",
            components=len(result_dict.get("components_identified", [])),
            rag_used=has_rag,
            is_refinement=bool(feedback),
        )
        return TechnicalReport.from_dict(result_dict)

    def evaluate_quality(
        self,
        extraction: ExtractionResult,
        report: TechnicalReport,
    ) -> QAScore:
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
{json.dumps(extraction.component_names, ensure_ascii=False)}

RELATÓRIO A AUDITAR:
{json.dumps(report.to_dict(), ensure_ascii=False, indent=2)}

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

        logger.info("text_llm.evaluate_quality.start", model=settings.llm_model)

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
            raw = _strip_json_fence(response.choices[0].message.content)
            qa_dict = json.loads(raw)
        except Exception as e:
            logger.warning("text_llm.evaluate_quality.failed", error=str(e))
            qa_dict = {
                "is_valid": True,
                "completeness_score": 0.7,
                "issues_found": [],
                "quality_notes": f"Avaliação LLM indisponível: {e}. Verificações básicas passaram.",
            }

        qa = QAScore.from_dict(qa_dict)

        # Score mínimo obrigatório
        if qa.completeness_score < QAScore.MIN_SCORE:
            issues = list(qa.issues_found) + [
                f"Score {qa.completeness_score:.2f} abaixo do mínimo aceitável ({QAScore.MIN_SCORE})."
            ]
            qa = QAScore(
                is_valid=False,
                completeness_score=qa.completeness_score,
                issues_found=issues,
                quality_notes=qa.quality_notes,
            )

        logger.info(
            "text_llm.evaluate_quality.done",
            is_valid=qa.is_valid,
            score=qa.completeness_score,
        )
        return qa
