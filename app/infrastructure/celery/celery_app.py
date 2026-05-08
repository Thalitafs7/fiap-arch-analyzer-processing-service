"""
Celery application — usa Redis como broker e backend de resultados.

Uso:
    celery -A app.infrastructure.celery.celery_app worker --loglevel=info
"""

from celery import Celery
from app.infrastructure.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "ia_service",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["app.infrastructure.celery"])
