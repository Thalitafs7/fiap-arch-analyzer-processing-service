"""
RAG Agent — indexa extração no pgvector e recupera contexto histórico.
Não bloqueante: retorna has_context=False em caso de falha.
"""

import os
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.infrastructure.config.settings import get_settings
from app.shared.exceptions import RAGError
from app.shared.logging import get_logger

logger = get_logger(__name__)

_vector_store = None


def _get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
        )
        _vector_store = PGVector(
            embeddings=embeddings,
            collection_name="diagram_analyses",
            connection=settings.postgres_connection_string,
            use_jsonb=True,
        )
    return _vector_store


def _index_analysis(analysis_id: str, extraction_result: dict) -> None:
    store = _get_vector_store()
    page_content = f"""Diagrama de Arquitetura:
{extraction_result.get('raw_description', '')}

Componentes: {', '.join(extraction_result.get('components', []))}
Padrões: {', '.join(extraction_result.get('patterns', []))}
Relacionamentos: {', '.join(extraction_result.get('relationships', [])[:10])}"""

    metadata = {
        "analysis_id": analysis_id,
        "components_count": len(extraction_result.get("components", [])),
        "has_report": False,
    }

    doc = Document(page_content=page_content.strip(), metadata=metadata)
    store.add_documents([doc], ids=[analysis_id])
    logger.info("rag.indexed", analysis_id=analysis_id)


def _retrieve_context(extraction_result: dict, top_k: int = 3) -> dict:
    store = _get_vector_store()
    query = f"""{extraction_result.get('raw_description', '')}
Componentes: {', '.join(extraction_result.get('components', []))}
Padrões: {', '.join(extraction_result.get('patterns', []))}"""

    similar_docs = store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter={"has_report": True},
    )

    relevant = [(doc, score) for doc, score in similar_docs if score < 0.3]

    if not relevant:
        return {"has_context": False, "context_text": "", "similar_analyses": []}

    context_parts = []
    similar_refs = []

    for doc, score in relevant:
        similarity = round(1 - score, 3)
        meta = doc.metadata
        context_parts.append(f"""--- Análise Similar (similaridade: {similarity:.0%}) ---
{doc.page_content}
Riscos críticos: {meta.get('risks_high', 0)} alto(s)
Sumário: {meta.get('executive_summary', 'N/A')}""")
        similar_refs.append({
            "analysis_id": meta.get("analysis_id"),
            "similarity_score": similarity,
            "components_count": meta.get("components_count"),
            "risks_high": meta.get("risks_high", 0),
        })

    return {
        "has_context": True,
        "context_text": "\n".join(context_parts),
        "similar_analyses": similar_refs,
    }


def _build_enrichment(rag_context: dict, extraction_result: dict) -> str:
    settings = get_settings()
    llm_kwargs = {
        "model": settings.llm_model,
        "max_tokens": 4096,
        "openai_api_key": settings.openai_api_key,
        "max_retries": 6,
    }
    if settings.llm_base_url:
        llm_kwargs["openai_api_base"] = settings.llm_base_url
    llm = ChatOpenAI(**llm_kwargs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um arquiteto de software sênior analisando um diagrama.
Use o contexto de análises similares anteriores para enriquecer sua análise,
identificando padrões de risco recorrentes e boas práticas observadas.

CONTEXTO DE ANÁLISES SIMILARES:
{context}

Responda em português. Seja específico e baseie-se apenas nos dados fornecidos."""),
        ("human", """Com base nos componentes e contexto acima, identifique:
1. Padrões de risco que aparecem em arquiteturas similares
2. Boas práticas observadas em sistemas comparáveis
3. Recomendações adicionais com base no histórico

COMPONENTES ATUAIS: {components}
PADRÕES ATUAIS: {patterns}"""),
    ])

    chain = (
        {
            "context": lambda x: x["context_text"],
            "components": lambda x: ", ".join(x["components"]),
            "patterns": lambda x: ", ".join(x["patterns"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "context_text": rag_context["context_text"],
        "components": extraction_result.get("components", []),
        "patterns": extraction_result.get("patterns", []),
    })


def _has_previous_reports(db) -> bool:
    """Verifica se existem relatórios anteriores no banco (query leve, sem OpenAI)."""
    from sqlalchemy import text
    row = db.execute(text("SELECT 1 FROM reports LIMIT 1")).first()
    return row is not None


def run(analysis_id: str, extraction_result: dict, db=None) -> dict:
    """
    Indexa a extração atual e recupera contexto histórico.
    Não bloqueante — retorna has_context=False em caso de falha.
    Pula chamadas OpenAI quando não há histórico no banco.
    """
    _no_context = {
        "analysis_id": analysis_id,
        "has_context": False,
        "similar_analyses": [],
        "rag_enrichment": "",
    }

    # Skip: Groq/outros não têm API de embeddings — RAG requer OpenAI embeddings
    settings = get_settings()
    if settings.llm_base_url:
        logger.info("rag.skipped_no_embeddings", analysis_id=analysis_id, reason="llm_base_url configurado (Groq/outro)")
        return _no_context

    # Skip rápido: se não há relatórios anteriores, não gasta chamadas OpenAI
    if db is not None and not _has_previous_reports(db):
        logger.info("rag.skipped_no_history", analysis_id=analysis_id)
        return _no_context

    try:
        _index_analysis(analysis_id, extraction_result)
        rag_context = _retrieve_context(extraction_result, top_k=3)

        result = {
            "analysis_id": analysis_id,
            "has_context": rag_context["has_context"],
            "similar_analyses": rag_context.get("similar_analyses", []),
            "rag_enrichment": "",
        }

        if rag_context["has_context"]:
            result["rag_enrichment"] = _build_enrichment(rag_context, extraction_result)
            logger.info("rag.enrichment_generated", analysis_id=analysis_id)

        return result

    except Exception as e:
        logger.warning("rag.failed", error=str(e))
        raise RAGError(str(e))
