"""
Analysis Bounded Context — Entidades.
Objetos com identidade que mudam ao longo do tempo.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from app.domain.diagram_analysis.component import (
    Component,
    Relationship,
    ArchitecturalPattern,
)


@dataclass
class ExtractionResult:
    """
    Resultado da extração de componentes de um diagrama via LLM Vision.
    Entidade imutável após criação — representa o ground truth do diagrama.
    """
    components: List[Component]
    relationships: List[Relationship]
    patterns: List[ArchitecturalPattern]
    raw_description: str

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionResult":
        return cls(
            components=[Component(name=c) for c in data.get("components", [])],
            relationships=[
                Relationship.from_string(r)
                for r in data.get("relationships", [])
            ],
            patterns=[
                ArchitecturalPattern(name=p)
                for p in data.get("patterns", [])
            ],
            raw_description=data.get("raw_description", ""),
        )

    def to_dict(self) -> dict:
        return {
            "status": "em_processamento",
            "components": [str(c) for c in self.components],
            "relationships": [str(r) for r in self.relationships],
            "patterns": [str(p) for p in self.patterns],
            "raw_description": self.raw_description,
        }

    @property
    def component_names(self) -> List[str]:
        return [c.name for c in self.components]

    def has_component(self, name: str) -> bool:
        return any(c.name.lower() == name.lower() for c in self.components)
