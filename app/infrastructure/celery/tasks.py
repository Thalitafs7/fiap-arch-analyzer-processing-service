"""
Celery tasks — processamento assíncrono de diagramas.

Cada task publica eventos de progresso via Redis pub/sub (tempo real)
e armazena no Redis list (reconexão/catch-up).

Canal pub/sub:   job:{task_id}
Lista de eventos: job:{task_id}:events   (TTL 10 min)
"""

import json

import redis

from app.infrastructure.celery.celery_app import celery_app
from app.infrastructure.config.settings import get_settings
from app.shared.logging import get_logger

logger = get_logger(__name__)

_EVENT_TTL_SECONDS = 600  # 10 minutos


@celery_app.task(bind=True, name="analyze_diagram")
def analyze_diagram_task(self, file_bytes_hex: str, file_name: str):
    """
    Executa o pipeline de análise de diagrama em background.

    Args:
        file_bytes_hex: conteúdo do arquivo em hex (JSON-safe).
        file_name: nome original do arquivo.
    """
    settings = get_settings()
    r = redis.from_url(settings.redis_url)
    job_id = self.request.id
    channel = f"job:{job_id}"
    events_key = f"job:{job_id}:events"

    file_bytes = bytes.fromhex(file_bytes_hex)

    logger.info("celery.task.started", job_id=job_id, file_name=file_name)

    def on_step(step: str, status: str, data: dict):
        event = {"step": step, "status": status, "data": data}
        event_json = json.dumps(event, ensure_ascii=False, default=str)
        r.rpush(events_key, event_json)
        r.publish(channel, event_json)

    from app.infrastructure.persistence.database import get_session_factory

    db = get_session_factory()()
    try:
        from app.pipeline.analysis_orchestrator import run_pipeline

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

        logger.info("celery.task.completed", job_id=job_id)
        return result

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

        logger.error("celery.task.failed", job_id=job_id, error=str(exc))
        raise

    finally:
        db.close()
