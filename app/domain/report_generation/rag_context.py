from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RagContext:
    """Contexto recuperado via RAG para enriquecer a geração do relatório."""
    has_context: bool
    enrichment_text: str
    similar_analyses_count: int

    @classmethod
    def empty(cls) -> "RagContext":
        return cls(has_context=False, enrichment_text="", similar_analyses_count=0)

    @classmethod
    def from_dict(cls, data: dict) -> "RagContext":
        return cls(
            has_context=data.get("has_context", False),
            enrichment_text=data.get("rag_enrichment", ""),
            similar_analyses_count=len(data.get("similar_analyses", [])),
        )
