"""
Orchestrator — ponto de entrada do pipeline E2E.

Com DDD, o orchestrator delega toda a orquestração ao AnalyzeDiagramUseCase.
Esta camada existe apenas para manter compatibilidade com chamadas existentes
(main.py, sqs_consumer.py) sem exigir mudanças nessas entradas.

Fluxo de dependências:
    orchestrator → AnalyzeDiagramUseCase
                 → IAnalysisRepository  ← SQLAlchemyAnalysisRepository
                 → IReportRepository    ← SQLAlchemyReportRepository
                 → IVisionLLM           ← OpenAIVisionAdapter
                 → ITextLLM             ← OpenAITextAdapter
                 → IVectorStore         ← PGVectorAdapter
                 → GuardrailService     (domain service)
"""

from typing import Callable, Optional
from sqlalchemy.orm import Session

from app.infrastructure.composition_root import build_analyze_use_case

# Tipo do callback: (step_name, status, data_dict) -> None
StepCallback = Callable[[str, str, dict], None]


def run_pipeline(
    db: Session,
    file_bytes: bytes,
    file_name: str,
    s3_key: Optional[str] = None,
    sqs_message_id: Optional[str] = None,
    on_step: Optional[StepCallback] = None,
) -> dict:
    """
    Executa o pipeline completo de análise de diagrama.

    Delega ao AnalyzeDiagramUseCase (DDD Application Layer).

    Args:
        db:              sessão SQLAlchemy (injetada pelo FastAPI / SQS consumer)
        file_bytes:      conteúdo binário do arquivo
        file_name:       nome original do arquivo
        s3_key:          chave S3 (opcional, fluxo SQS)
        sqs_message_id:  ID da mensagem SQS (idempotência)
        on_step:         callback de progresso para streaming SSE

    Returns:
        dict com analysis_id, status, report e qa
    """
    source = "sqs" if sqs_message_id else "upload"
    use_case = build_analyze_use_case(db)

    return use_case.execute(
        file_bytes=file_bytes,
        file_name=file_name,
        s3_key=s3_key,
        sqs_message_id=sqs_message_id,
        source=source,
        on_step=on_step,
    )
