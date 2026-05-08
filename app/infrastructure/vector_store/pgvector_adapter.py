"""
Infrastructure Layer — PGVector adapter para o port IVectorStore.
Encapsula toda lógica de embeddings e busca semântica.
"""

from __future__ import annotations
from typing import Optional

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.application.ports.vector_store_port import IVectorStore
from app.domain.diagram_analysis.extraction_result import ExtractionResult
from app.domain.report_generation.rag_context import RagContext
from app.domain.shared.analysis_id import AnalysisId
from app.infrastructure.config.settings import get_settings
from app.shared.exceptions import RAGError
from app.shared.logging import get_logger

logger = get_logger(__name__)

_vector_store_instance: Optional[PGVector] = None


def _build_embeddings():
    """Retorna OpenAI embeddings se disponível, senão HuggingFace local."""
    settings = get_settings()
    if settings.llm_base_url:
        logger.info("vector_store.embeddings", backend="huggingface-local")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    logger.info("vector_store.embeddings", backend="openai")
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
    )


def _get_pgvector() -> PGVector:
    global _vector_store_instance
    if _vector_store_instance is None:
        settings = get_settings()
        _vector_store_instance = PGVector(
            embeddings=_build_embeddings(),
            collection_name="diagram_analyses",
            connection=settings.postgres_connection_string,
            use_jsonb=True,
        )
    return _vector_store_instance


class PGVectorAdapter(IVectorStore):
    """
    Implementação do port IVectorStore usando pgvector.
    Requer OpenAI embeddings — é automaticamente ignorado quando usando Groq.
    """

    def __init__(self, db: Session):
        self._db = db

    def index(self, analysis_id: AnalysisId, extraction: ExtractionResult) -> None:
        store = _get_pgvector()
        page_content = (
            f"Diagrama de Arquitetura:\n{extraction.raw_description}\n\n"
            f"Componentes: {', '.join(extraction.component_names)}\n"
            f"Padrões: {', '.join(str(p) for p in extraction.patterns)}\n"
            f"Relacionamentos: {', '.join(str(r) for r in extraction.relationships[:10])}"
        )

        doc = Document(
            page_content=page_content.strip(),
            metadata={
                "analysis_id": str(analysis_id),
                "components_count": len(extraction.components),
                "has_report": False,
            },
        )
        store.add_documents([doc], ids=[str(analysis_id)])
        logger.info("vector_store.indexed", analysis_id=str(analysis_id))

    def mark_as_reported(self, analysis_id: AnalysisId) -> None:
        self._db.execute(
            text("""
                UPDATE langchain_pg_embedding
                SET cmetadata = cmetadata || '{"has_report": true}'::jsonb
                WHERE cmetadata->>'analysis_id' = :aid
            """),
            {"aid": str(analysis_id)},
        )
        self._db.commit()
        logger.info("vector_store.marked_as_reported", analysis_id=str(analysis_id))

    def retrieve_context(
        self,
        extraction: ExtractionResult,
        exclude_analysis_id: AnalysisId,
    ) -> RagContext:
        # Skip rápido se não há relatórios anteriores
        if not self._has_previous_reports():
            logger.info("vector_store.retrieve.skipped_no_history")
            return RagContext.empty()

        try:
            store = _get_pgvector()
            query = (
                f"{extraction.raw_description}\n"
                f"Componentes: {', '.join(extraction.component_names)}\n"
                f"Padrões: {', '.join(str(p) for p in extraction.patterns)}"
            )

            similar_docs = store.similarity_search_with_score(
                query=query,
                k=3,
                filter={"has_report": True},
            )

            relevant = [(doc, score) for doc, score in similar_docs if score < 0.3]
            if not relevant:
                return RagContext.empty()

            context_parts = []
            similar_refs = []

            for doc, score in relevant:
                similarity = round(1 - score, 3)
                meta = doc.metadata
                context_parts.append(
                    f"--- Análise Similar (similaridade: {similarity:.0%}) ---\n"
                    f"{doc.page_content}\n"
                    f"Riscos críticos: {meta.get('risks_high', 0)} alto(s)\n"
                    f"Sumário: {meta.get('executive_summary', 'N/A')}"
                )
                similar_refs.append({
                    "analysis_id": meta.get("analysis_id"),
                    "similarity_score": similarity,
                })

            enrichment = self._build_enrichment(
                context_text="\n".join(context_parts),
                extraction=extraction,
            )

            return RagContext(
                has_context=True,
                enrichment_text=enrichment,
                similar_analyses_count=len(similar_refs),
            )

        except Exception as e:
            logger.warning("vector_store.retrieve.failed", error=str(e))
            raise RAGError(str(e))

    def _has_previous_reports(self) -> bool:
        row = self._db.execute(text("SELECT 1 FROM reports LIMIT 1")).first()
        return row is not None

    def _build_enrichment(self, context_text: str, extraction: ExtractionResult) -> str:
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
            "context_text": context_text,
            "components": extraction.component_names,
            "patterns": [str(p) for p in extraction.patterns],
        })
