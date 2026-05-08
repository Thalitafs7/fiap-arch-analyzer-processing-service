"""
Report Bounded Context — Entidades.
TechnicalReport é a entidade central gerada pelo pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from app.domain.report_generation.risk import RiskItem
from app.domain.report_generation.recommendation import Recommendation


@dataclass
class TechnicalReport:
    """
    Relatório técnico gerado a partir de um diagrama de arquitetura.
    Inclui componentes identificados, riscos e recomendações.
    """
    components_identified: List[str]
    architectural_risks: List[RiskItem]
    recommendations: List[Recommendation]
    executive_summary: str
    rag_used: bool

    @classmethod
    def from_dict(cls, data: dict) -> "TechnicalReport":
        return cls(
            components_identified=list(data.get("components_identified", [])),
            architectural_risks=[
                RiskItem.from_dict(r)
                for r in data.get("architectural_risks", [])
            ],
            recommendations=[
                Recommendation.from_string(r)
                for r in data.get("recommendations", [])
            ],
            executive_summary=data.get("executive_summary", ""),
            rag_used=data.get("rag_used", False),
        )

    def to_dict(self) -> dict:
        return {
            "components_identified": self.components_identified,
            "architectural_risks": [r.to_dict() for r in self.architectural_risks],
            "recommendations": [str(r) for r in self.recommendations],
            "executive_summary": self.executive_summary,
            "rag_used": self.rag_used,
        }

    @property
    def risk_severity_summary(self) -> dict:
        return {
            "high": sum(1 for r in self.architectural_risks if r.severity.value == "ALTO"),
            "medium": sum(1 for r in self.architectural_risks if r.severity.value == "MÉDIO"),
            "low": sum(1 for r in self.architectural_risks if r.severity.value == "BAIXO"),
        }
