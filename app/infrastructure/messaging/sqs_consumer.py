"""
SQS Consumer — consome mensagens da fila enviadas pelo time SOAT.
Cada mensagem contém a URL pré-assinada do arquivo no S3 e metadados.
Dispara o pipeline de análise e envia o resultado via webhook.

Formato esperado da mensagem SQS:
{
  "file_name":        "diagrama.png",
  "s3_url":           "https://...",     <- URL pré-assinada do S3
  "callback_url":     "https://...",     <- Endpoint SOAT para receber o resultado final
  "soat_analysis_id": "uuid-soat"        <- ID da análise no sistema SOAT
}

Fluxo de integração com SOAT:
  1. Ao iniciar o processamento → PUT {SOAT_BASE_URL}/analyses/{soat_analysis_id}/status
     com body {"status": "em_processamento"}
  2. Ao finalizar → POST callback_url devolvendo soat_analysis_id + relatório

Melhorias implementadas (Workstream 2):
  - Graceful shutdown via SIGTERM/SIGINT
  - Idempotência por sqs_message_id (evita reprocessamento em entregas duplicadas)
  - Retry com exponential backoff no download (tenacity)
  - Log do ApproximateReceiveCount para detectar poison messages
  - Webhook de devolutiva após conclusão do pipeline (Workstream 1)
"""

import json
import signal
import time

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.infrastructure.config.settings import get_settings
from app.infrastructure.persistence.database import get_session_factory
from app.infrastructure.persistence.sqlalchemy_analysis_repository import SQLAlchemyAnalysisRepository
from app.pipeline.analysis_orchestrator import run_pipeline
from app.shared.logging import get_logger
from app.infrastructure.http.webhook_sender import send_webhook
from app.infrastructure.http.soat_client import update_analysis_status

logger = get_logger(__name__)

WAIT_TIME_SECONDS = 20      # long polling
MAX_MESSAGES = 5
VISIBILITY_TIMEOUT = 300    # 5 min para processar

# ──────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────

_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    logger.info("sqs.consumer.shutdown_requested", signal=signum)
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# ──────────────────────────────────────────────
# SQS client
# ──────────────────────────────────────────────

def _get_sqs_client():
    settings = get_settings()
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_access_key_id:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        kwargs["aws_session_token"] = settings.aws_session_token
    return boto3.client("sqs", **kwargs)


# ──────────────────────────────────────────────
# Download com retry
# ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.NetworkError,
        Exception,
    )),
    reraise=True,
)
def _download_file(s3_url: str) -> bytes:
    """Baixa o arquivo da URL pré-assinada do S3 com retry automático."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(s3_url)
        response.raise_for_status()
        return response.content


# ──────────────────────────────────────────────
# Processamento de mensagem
# ──────────────────────────────────────────────

def _process_message(body: dict, sqs_message_id: str) -> None:
    settings = get_settings()
    SessionLocal = get_session_factory()
    db = SessionLocal()

    s3_url = body.get("s3_url")
    file_name = body.get("file_name", "diagrama.png")
    callback_url = body.get("callback_url")
    soat_analysis_id = body.get("soat_analysis_id", "")

    try:
        if not s3_url:
            logger.error("sqs.message.missing_s3_url", body=body)
            return

        # ── Idempotência ─────────────────────────────────────────────
        analysis_repo = SQLAlchemyAnalysisRepository(db)
        existing = analysis_repo.get_by_sqs_message_id(sqs_message_id)
        if existing:
            logger.info(
                "sqs.message.duplicate_skipped",
                message_id=sqs_message_id,
                existing_analysis_id=str(existing.id),
                existing_status=existing.status.value,
            )
            return

        logger.info(
            "sqs.processing",
            file_name=file_name,
            s3_url=s3_url[:60],
            soat_analysis_id=soat_analysis_id or "n/a",
        )

        # ── Notifica SOAT: início do processamento ────────────────────
        update_analysis_status(soat_analysis_id, "em_processamento")

        # ── Download ─────────────────────────────────────────────────
        file_bytes = _download_file(s3_url)

        # ── Pipeline ─────────────────────────────────────────────────
        result = run_pipeline(
            db=db,
            file_bytes=file_bytes,
            file_name=file_name,
            s3_key=s3_url,
            sqs_message_id=sqs_message_id,
        )

        logger.info(
            "sqs.pipeline_completed",
            analysis_id=result["analysis_id"],
            status=result["status"],
            soat_analysis_id=soat_analysis_id or "n/a",
        )

        # ── Callback de sucesso para SOAT ─────────────────────────────
        send_webhook(
            callback_url=callback_url,
            analysis_id=result["analysis_id"],
            status=result["status"],
            report=result.get("report"),
            soat_analysis_id=soat_analysis_id,
        )

    except Exception as exc:
        logger.error(
            "sqs.pipeline_error",
            error=str(exc),
            file_name=file_name,
            soat_analysis_id=soat_analysis_id or "n/a",
        )

        # ── Callback de erro para SOAT ────────────────────────────────
        # analysis_id pode não existir se a falha foi antes do create_analysis.
        send_webhook(
            callback_url=callback_url,
            analysis_id=body.get("analysis_id", "unknown"),
            status="erro",
            error_message=str(exc),
            soat_analysis_id=soat_analysis_id,
        )

        raise  # Re-lança para que a mensagem volte à fila após VisibilityTimeout

    finally:
        db.close()


# ──────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────

def start() -> None:
    """Loop principal do consumer SQS com graceful shutdown."""
    settings = get_settings()
    sqs = _get_sqs_client()
    queue_url = settings.sqs_queue_url

    if not queue_url:
        logger.error("sqs.consumer.no_queue_url")
        return

    logger.info("sqs.consumer.started", queue_url=queue_url)

    while not _shutdown_requested:
        try:
            response = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=MAX_MESSAGES,
                WaitTimeSeconds=WAIT_TIME_SECONDS,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
                AttributeNames=["ApproximateReceiveCount"],
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            for message in messages:
                if _shutdown_requested:
                    break

                message_id = message["MessageId"]
                receipt_handle = message["ReceiptHandle"]
                receive_count = int(
                    message.get("Attributes", {}).get("ApproximateReceiveCount", 1)
                )

                if receive_count > 3:
                    logger.warning(
                        "sqs.message.possible_poison",
                        message_id=message_id,
                        receive_count=receive_count,
                    )

                try:
                    body = json.loads(message["Body"])
                    _process_message(body, sqs_message_id=message_id)

                    # Remove da fila somente após processamento bem-sucedido
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
                    logger.info("sqs.message.deleted", message_id=message_id)

                except Exception as exc:
                    logger.error(
                        "sqs.message.processing_failed",
                        message_id=message_id,
                        receive_count=receive_count,
                        error=str(exc),
                    )
                    # Mensagem volta à fila após VisibilityTimeout

        except (BotoCoreError, ClientError) as exc:
            logger.error("sqs.receive_error", error=str(exc))
            time.sleep(5)

    logger.info("sqs.consumer.stopped")
