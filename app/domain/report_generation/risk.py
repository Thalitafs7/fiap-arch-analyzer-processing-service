from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List


class RiskCategory(str, Enum):
    SPOF = "SPOF"
    SECURITY = "Segurança"
    SCALABILITY = "Escalabilidade"
    COUPLING = "Acoplamento"
    OBSERVABILITY = "Observabilidade"
    RESILIENCE = "Resiliência"

    @classmethod
    def all_labels(cls) -> str:
        return ", ".join(c.value for c in cls)


class Severity(str, Enum):
    HIGH = "ALTO"
    MEDIUM = "MÉDIO"
    LOW = "BAIXO"

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        mapping = {v.value.upper(): v for v in cls}
        result = mapping.get(value.upper())
        if result is None:
            return cls.LOW
        return result


@dataclass(frozen=True)
class RiskItem:
    """Risco arquitetural identificado, com categoria, severidade e mitigação."""
    risk_category: RiskCategory
    description: str
    severity: Severity
    affected_components: List[str]
    mitigation: str

    @classmethod
    def from_dict(cls, data: dict) -> "RiskItem":
        return cls(
            risk_category=RiskCategory(data.get("type", RiskCategory.RESILIENCE.value)),
            description=data.get("description", ""),
            severity=Severity.from_string(data.get("severity", "BAIXO")),
            affected_components=list(data.get("affected_components", [])),
            mitigation=data.get("mitigation", ""),
        )

    def to_dict(self) -> dict:
        return {
            "type": self.risk_category.value,
            "description": self.description,
            "severity": self.severity.value,
            "affected_components": list(self.affected_components),
            "mitigation": self.mitigation,
        }
