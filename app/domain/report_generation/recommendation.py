from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    """Recomendação arquitetural, podendo ser influenciada pelo RAG."""
    text: str
    rag_influenced: bool = False

    @classmethod
    def from_string(cls, text: str) -> "Recommendation":
        return cls(text=text, rag_influenced="[RAG]" in text)

    def __str__(self) -> str:
        return self.text
