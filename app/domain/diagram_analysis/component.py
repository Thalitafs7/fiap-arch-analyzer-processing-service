from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    """Componente arquitetural identificado no diagrama."""
    name: str

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Component name cannot be empty")

    def matches(self, other: "Component") -> bool:
        return self.name.lower() == other.name.lower()

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Relationship:
    """Relacionamento entre dois componentes arquiteturais."""
    source: str
    target: str
    description: str

    @classmethod
    def from_string(cls, raw: str) -> "Relationship":
        """
        Parseia formato 'ComponenteA → ComponenteB: descrição'.
        Aceita também '->'.
        """
        import re
        pattern = r"^(.+?)\s*[→\->]+\s*(.+?)(?::\s*(.*))?$"
        match = re.match(pattern, raw.strip())
        if match:
            return cls(
                source=match.group(1).strip(),
                target=match.group(2).strip(),
                description=(match.group(3) or "").strip(),
            )
        return cls(source="", target="", description=raw)

    def __str__(self) -> str:
        if self.source and self.target:
            return f"{self.source} → {self.target}: {self.description}"
        return self.description


@dataclass(frozen=True)
class ArchitecturalPattern:
    """Padrão arquitetural identificado (ex: microservices, event-driven)."""
    name: str

    def __str__(self) -> str:
        return self.name
