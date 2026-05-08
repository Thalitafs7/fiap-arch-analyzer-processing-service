class PipelineError(Exception):
    """Erro genérico do pipeline de análise."""
    def __init__(self, message: str, step: str = "", analysis_id: str = ""):
        self.step = step
        self.analysis_id = analysis_id
        super().__init__(message)


class IngestionError(PipelineError):
    """Falha no ingestion-agent (validação ou download do S3)."""


class ExtractionError(PipelineError):
    """Falha no extraction-agent (Vision LLM)."""


class RiskAnalysisError(PipelineError):
    """Falha no risk-agent."""


class ReportGenerationError(PipelineError):
    """Falha no report-agent."""


class QAError(PipelineError):
    """Falha no qa-agent ou relatório rejeitado pelo QA."""


class GuardrailError(PipelineError):
    """Falha nos guardrails de entrada ou saída (prompt injection, PII, schema)."""


class RAGError(Exception):
    """Falha no rag-agent — não bloqueia o pipeline."""
