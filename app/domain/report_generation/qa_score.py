from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class QAScore:
    """Score de qualidade produzido pelo QA Agent."""
    is_valid: bool
    completeness_score: float
    issues_found: List[str]
    quality_notes: str

    MIN_SCORE = 0.6

    @classmethod
    def from_dict(cls, data: dict) -> "QAScore":
        return cls(
            is_valid=data.get("is_valid", False),
            completeness_score=data.get("completeness_score", 0.0),
            issues_found=list(data.get("issues_found", [])),
            quality_notes=data.get("quality_notes", ""),
        )

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "completeness_score": self.completeness_score,
            "issues_found": list(self.issues_found),
            "quality_notes": self.quality_notes,
            "status": "analisado" if self.is_valid else "erro",
        }
