"""
RabbitMQ consumer (modo teste/webhook).

Consome mensagens de uma fila bound ao exchange `reports.events` (ou outro
configurado) e executa o pipeline de análise. O payload esperado é JSON:

    {
        "file_name": "diagrama.png",
        "file_b64": "<base64 do binário>",
        "content_type": "image/png"        # opcional
    }

Roda em thread separada, iniciado no lifespan do FastAPI.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Optional

import pika
import redis as redis_lib

from app.infrastructure.config.settings import get_settings
from app.infrastructure.persistence.database import get_session_factory
from app.pipeline.analysis_orchestrator import run_pipeline
from app.shared.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_QUEUE = "ia.diagram.uploads"
_DEFAULT_ROUTING_KEY = "diagram.uploaded"
_EVENT_TTL_SECONDS = 600

_stop_event = threading.Event()


def _process(body: bytes) -> None:
    payload = json.loads(body.decode("utf-8"))
    file_name = payload.get("file_name") or "diagrama.bin"
    file_b64 = payload.get("file_b64")
    job_id = payload.get("job_id")

    if not file_b64:
        logger.error("rabbit.consumer.missing_file_b64", payload_keys=list(payload.keys()))
        return
    if not job_id:
        logger.error("rabbit.consumer.missing_job_id", payload_keys=list(payload.keys()))
        return

    file_bytes = base64.b64decode(file_b64)
    logger.info("rabbit.consumer.processing", job_id=job_id, file_name=file_name, bytes=len(file_bytes))

    settings = get_settings()
    r = redis_lib.from_url(settings.redis_url)
    channel = f"job:{job_id}"
    events_key = f"job:{job_id}:events"

    def on_step(step: str, status: str, data: dict) -> None:
        event = {"step": step, "status": status, "data": data}
        event_json = json.dumps(event, ensure_ascii=False, default=str)
        r.rpush(events_key, event_json)
        r.publish(channel, event_json)

    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        result = run_pipeline(
            db=db,
            file_bytes=file_bytes,
            file_name=file_name,
            on_step=on_step,
        )

        done_event = {"step": "done", "status": "complete", "data": result}
        done_json = json.dumps(done_event, ensure_ascii=False, default=str)
        r.rpush(events_key, done_json)
        r.publish(channel, done_json)
        r.expire(events_key, _EVENT_TTL_SECONDS)

        logger.info(
            "rabbit.consumer.pipeline_completed",
            job_id=job_id,
            analysis_id=result.get("analysis_id"),
            status=result.get("status"),
        )
    except Exception as exc:
        error_event = {
            "step": "pipeline",
            "status": "error",
            "data": {"error": str(exc), "error_type": type(exc).__name__},
        }
        error_json = json.dumps(error_event, ensure_ascii=False, default=str)
        r.rpush(events_key, error_json)
        r.publish(channel, error_json)
        r.expire(events_key, _EVENT_TTL_SECONDS)
        logger.error("rabbit.consumer.pipeline_failed", job_id=job_id, error=str(exc))
        raise
    finally:
        db.close()


def _consume_loop(queue_name: str, exchange: str, routing_key: str) -> None:
    settings = get_settings()
    rabbit_url = settings.rabbitmq_url

    while not _stop_event.is_set():
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
                    _process(body)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as exc:
                    logger.error("rabbit.consumer.process_error", error=str(exc))
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=queue_name, on_message_callback=_on_message)
            channel.start_consuming()

        except Exception as exc:
            logger.error("rabbit.consumer.connection_error", error=str(exc))
            time.sleep(5)  # backoff antes de tentar reconectar


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
