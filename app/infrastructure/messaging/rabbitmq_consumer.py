"""
RabbitMQ consumer.

Comportamento espelhado do SQS consumer:
  - Graceful shutdown via SIGTERM/SIGINT
  - Idempotência por job_id (evita reprocessamento em entregas duplicadas)
  - Retry com exponential backoff no pipeline (tenacity)
  - Log de mensagens reenviadas (poison message detection via method.redelivered)
  - Notificação de status ao SOAT (update_analysis_status)
  - Webhook de devolutiva em sucesso e erro (send_webhook)

Formato esperado da mensagem:
{
    "file_name":        "diagrama.png",
    "file_b64":         "<base64 do binário>",
    "content_type":     "image/png",        # opcional
    "job_id":           "uuid",
    "soat_analysis_id": "uuid-soat",        # opcional
    "callback_url":     "https://..."       # opcional
}

Roda em thread separada, iniciado no lifespan do FastAPI.
"""
from __future__ import annotations

import base64
import json
import signal
import threading
import time
from typing import Optional

import pika
import redis as redis_lib
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
from app.infrastructure.http.webhook_sender import send_webhook
from app.infrastructure.http.soat_client import update_analysis_status
from app.shared.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_QUEUE = "ia.diagram.uploads"
_DEFAULT_ROUTING_KEY = "diagram.uploaded"
_EVENT_TTL_SECONDS = 600

# ──────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────

_shutdown_requested = False


def _handle_shutdown(signum, frame):
    global _shutdown_requested
    logger.info("rabbit.consumer.shutdown_requested", signal=signum)
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# ──────────────────────────────────────────────
# Pipeline com retry
# ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _run_pipeline_with_retry(**kwargs):
    return run_pipeline(**kwargs)


# ──────────────────────────────────────────────
# Processamento de mensagem
# ──────────────────────────────────────────────

def _process(body: bytes, redelivered: bool) -> None:
    payload = json.loads(body.decode("utf-8"))
    file_name = payload.get("file_name") or "diagrama.bin"
    file_b64 = payload.get("file_b64")
    job_id = payload.get("job_id")
    soat_analysis_id = payload.get("soat_analysis_id", "")
    callback_url = payload.get("callback_url")

    if not file_b64:
        logger.error("rabbit.consumer.missing_file_b64", payload_keys=list(payload.keys()))
        return
    if not job_id:
        logger.error("rabbit.consumer.missing_job_id", payload_keys=list(payload.keys()))
        return

    if redelivered:
        logger.warning(
            "rabbit.consumer.message_redelivered",
            job_id=job_id,
            file_name=file_name,
        )

    settings = get_settings()
    r = redis_lib.from_url(settings.redis_url)
    channel_key = f"job:{job_id}"
    events_key = f"job:{job_id}:events"

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        # ── Idempotência ─────────────────────────────────────────────
        analysis_repo = SQLAlchemyAnalysisRepository(db)
        existing = analysis_repo.get_by_sqs_message_id(job_id)
        if existing:
            logger.info(
                "rabbit.consumer.duplicate_skipped",
                job_id=job_id,
                existing_analysis_id=str(existing.id),
                existing_status=existing.status.value,
            )
            return

        file_bytes = base64.b64decode(file_b64)
        logger.info(
            "rabbit.consumer.processing",
            job_id=job_id,
            file_name=file_name,
            bytes=len(file_bytes),
        )

        # ── Notifica SOAT: início do processamento ────────────────────
        if soat_analysis_id:
            update_analysis_status(soat_analysis_id, "em_processamento")

        def on_step(step: str, status: str, data: dict) -> None:
            event = {"step": step, "status": status, "data": data}
            event_json = json.dumps(event, ensure_ascii=False, default=str)
            r.rpush(events_key, event_json)
            r.publish(channel_key, event_json)

        # ── Pipeline com retry ────────────────────────────────────────
        result = _run_pipeline_with_retry(
            db=db,
            file_bytes=file_bytes,
            file_name=file_name,
            sqs_message_id=job_id,
            on_step=on_step,
        )

        done_event = {"step": "done", "status": "complete", "data": result}
        done_json = json.dumps(done_event, ensure_ascii=False, default=str)
        r.rpush(events_key, done_json)
        r.publish(channel_key, done_json)
        r.expire(events_key, _EVENT_TTL_SECONDS)

        logger.info(
            "rabbit.consumer.pipeline_completed",
            job_id=job_id,
            analysis_id=result.get("analysis_id"),
            status=result.get("status"),
        )

        # ── Callback de sucesso para SOAT ─────────────────────────────
        send_webhook(
            callback_url=callback_url,
            analysis_id=result.get("analysis_id"),
            status=result.get("status"),
            report=result.get("report"),
            soat_analysis_id=soat_analysis_id,
        )

    except Exception as exc:
        error_event = {
            "step": "pipeline",
            "status": "error",
            "data": {"error": str(exc), "error_type": type(exc).__name__},
        }
        error_json = json.dumps(error_event, ensure_ascii=False, default=str)
        r.rpush(events_key, error_json)
        r.publish(channel_key, error_json)
        r.expire(events_key, _EVENT_TTL_SECONDS)
        logger.error("rabbit.consumer.pipeline_failed", job_id=job_id, error=str(exc))

        # ── Callback de erro para SOAT ────────────────────────────────
        send_webhook(
            callback_url=callback_url,
            analysis_id=job_id,
            status="erro",
            error_message=str(exc),
            soat_analysis_id=soat_analysis_id,
            error_step=getattr(exc, "step", "pipeline"),
            error_type=type(exc).__name__,
        )

        raise
    finally:
        db.close()


# ──────────────────────────────────────────────
# Loop principal
# ──────────────────────────────────────────────

def _consume_loop(queue_name: str, exchange: str, routing_key: str) -> None:
    settings = get_settings()
    rabbit_url = settings.rabbitmq_url

    while not _shutdown_requested:
        try:
            params = pika.URLParameters(rabbit_url)
            params.heartbeat = 60
            params.blocked_connection_timeout = 30
            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            channel.exchange_declare(exchange=exchange, exchange_type="topic", durable=True)
            channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_bind(queue=queue_name, exchange=exchange, routing_key=routing_key)
            channel.basic_qos(prefetch_count=1)

            logger.info(
                "rabbit.consumer.started",
                queue=queue_name,
                exchange=exchange,
                routing_key=routing_key,
            )

            def _on_message(ch, method, properties, body):
                try:
                    _process(body, redelivered=method.redelivered)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as exc:
                    logger.error("rabbit.consumer.process_error", error=str(exc))
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=queue_name, on_message_callback=_on_message)

            # Loop não-bloqueante: processa eventos AMQP por 1s e verifica shutdown
            while not _shutdown_requested:
                connection.process_data_events(time_limit=1)

            channel.stop_consuming()
            connection.close()

        except Exception as exc:
            if not _shutdown_requested:
                logger.error("rabbit.consumer.connection_error", error=str(exc))
                time.sleep(5)

    logger.info("rabbit.consumer.stopped")


def start(
    queue_name: str = _DEFAULT_QUEUE,
    routing_key: str = _DEFAULT_ROUTING_KEY,
    exchange: Optional[str] = None,
) -> None:
    """Inicia o consumer em thread daemon."""
    settings = get_settings()
    target_exchange = exchange or settings.rabbitmq_exchange

    thread = threading.Thread(
        target=_consume_loop,
        args=(queue_name, target_exchange, routing_key),
        daemon=True,
        name="rabbitmq-consumer",
    )
    thread.start()
    logger.info("rabbit.consumer.thread_started", queue=queue_name)
