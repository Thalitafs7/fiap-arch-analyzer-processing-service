"""
Métricas de domínio para avaliação do fine-tuning.

Avalia outputs do modelo em critérios específicos do pipeline de relatórios:
- JSON validity, schema completeness, grounding, risk coverage, etc.
- Domain score composto com pesos alinhados ao QA do pipeline de produção.
"""

import json
from dataclasses import dataclass


# Chaves obrigatórias no relatório
REQUIRED_KEYS = {
    "components_identified",
    "architectural_risks",
    "recommendations",
    "executive_summary",
    "rag_used",
}

# Categorias de risco esperadas
RISK_CATEGORIES = {
    "spof", "segurança", "escalabilidade",
    "acoplamento", "observabilidade", "resiliência",
}

# Pesos do QA (alinhados com quality_validation_step.py)
WEIGHT_CONSISTENCY = 0.40
WEIGHT_COMPLETENESS = 0.30
WEIGHT_COHERENCE = 0.20
WEIGHT_QUALITY = 0.10


@dataclass
class DomainMetrics:
    """Resultado da avaliação de domínio para uma amostra."""
    json_valid: bool = False
    schema_complete: float = 0.0
    grounding_score: float = 0.0
    risk_coverage: float = 0.0
    recommendation_score: float = 0.0
    summary_score: float = 0.0
    domain_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "json_valid": float(self.json_valid),
            "schema_complete": self.schema_complete,
            "grounding_score": self.grounding_score,
            "risk_coverage": self.risk_coverage,
            "recommendation_score": self.recommendation_score,
            "summary_score": self.summary_score,
            "domain_score": self.domain_score,
        }


def parse_model_output(raw: str) -> dict | None:
    """Tenta parsear a saída do modelo como JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def compute_schema_completeness(report: dict) -> float:
    """Fração das chaves obrigatórias presentes no report."""
    present = REQUIRED_KEYS.intersection(report.keys())
    return len(present) / len(REQUIRED_KEYS)


def compute_grounding_score(report: dict, input_components: list[str]) -> float:
    """
    Fração dos components_identified que existem nos componentes de input.
    Mede anti-alucinação (Consistency — peso 40% no QA).
    """
    report_components = report.get("components_identified", [])
    if not report_components or not input_components:
        return 0.0

    input_set = {c.lower().strip() for c in input_components}
    grounded = sum(
        1 for c in report_components
        if c.lower().strip() in input_set
    )
    return grounded / len(report_components)


def compute_risk_coverage(report: dict) -> float:
    """Fração das 6 categorias de risco representadas no report."""
    risks = report.get("architectural_risks", [])
    if not risks:
        return 0.0

    found_categories = set()
    for risk in risks:
        risk_type = risk.get("type", "").lower().strip()
        for cat in RISK_CATEGORIES:
            if cat in risk_type:
                found_categories.add(cat)

    return len(found_categories) / len(RISK_CATEGORIES)


def compute_recommendation_score(report: dict) -> float:
    """Score baseado no número de recomendações (mínimo 3)."""
    recs = report.get("recommendations", [])
    if not recs:
        return 0.0
    return min(len(recs) / 3.0, 1.0)


def compute_summary_score(report: dict) -> float:
    """Score baseado no comprimento do executive_summary (mínimo 100 chars)."""
    summary = report.get("executive_summary", "")
    if not summary:
        return 0.0
    return min(len(summary) / 100.0, 1.0)


def evaluate_single(
    model_output: str,
    input_components: list[str],
) -> DomainMetrics:
    """
    Avalia uma única saída do modelo contra as métricas de domínio.

    Args:
        model_output: texto bruto gerado pelo modelo
        input_components: lista de componentes do input (extraction)

    Returns:
        DomainMetrics com todos os scores calculados
    """
    metrics = DomainMetrics()

    report = parse_model_output(model_output)
    if report is None:
        return metrics  # tudo zero

    metrics.json_valid = True
    metrics.schema_complete = compute_schema_completeness(report)
    metrics.grounding_score = compute_grounding_score(report, input_components)
    metrics.risk_coverage = compute_risk_coverage(report)
    metrics.recommendation_score = compute_recommendation_score(report)
    metrics.summary_score = compute_summary_score(report)

    # Domain score composto (alinhado com pesos do QA de produção)
    # Consistency 40%: grounding + risk referencing components
    consistency = metrics.grounding_score

    # Completeness 30%: schema + recommendations + summary
    completeness = (
        metrics.schema_complete * 0.4
        + metrics.recommendation_score * 0.3
        + metrics.summary_score * 0.3
    )

    # Coherence 20%: risk coverage (risks linked to patterns)
    coherence = metrics.risk_coverage

    # Quality 10%: JSON validity + summary length
    quality = float(metrics.json_valid) * 0.5 + metrics.summary_score * 0.5

    metrics.domain_score = (
        WEIGHT_CONSISTENCY * consistency
        + WEIGHT_COMPLETENESS * completeness
        + WEIGHT_COHERENCE * coherence
        + WEIGHT_QUALITY * quality
    )

    return metrics


def evaluate_batch(
    outputs: list[str],
    inputs_components: list[list[str]],
) -> dict[str, float]:
    """
    Avalia um batch de saídas e retorna médias de todas as métricas.

    Returns:
        Dict com médias: eval_json_valid_rate, eval_schema_complete,
        eval_grounding_score, eval_risk_coverage, eval_recommendation_score,
        eval_summary_score, eval_domain_score
    """
    if not outputs:
        return {}

    all_metrics = [
        evaluate_single(out, comps)
        for out, comps in zip(outputs, inputs_components)
    ]

    n = len(all_metrics)
    return {
        "eval_json_valid_rate": sum(m.json_valid for m in all_metrics) / n,
        "eval_schema_complete": sum(m.schema_complete for m in all_metrics) / n,
        "eval_grounding_score": sum(m.grounding_score for m in all_metrics) / n,
        "eval_risk_coverage": sum(m.risk_coverage for m in all_metrics) / n,
        "eval_recommendation_score": sum(m.recommendation_score for m in all_metrics) / n,
        "eval_summary_score": sum(m.summary_score for m in all_metrics) / n,
        "eval_domain_score": sum(m.domain_score for m in all_metrics) / n,
    }
