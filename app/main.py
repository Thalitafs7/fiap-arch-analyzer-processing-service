import asyncio
import json
import queue
import threading
import traceback
from typing import Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

import redis as redis_lib

from app.infrastructure.config.settings import get_settings
from app.shared.logging import configure_logging, get_logger
from app.infrastructure.persistence.database import get_db, check_db_connection
from app.pipeline.analysis_orchestrator import run_pipeline
from app.shared.exceptions import PipelineError

configure_logging()
logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "pdf"}

_SENTINEL = object()  # marca fim do stream


def _start_sqs_consumer():
    """Inicia o consumer SQS em thread separada (não bloqueia o servidor HTTP)."""
    settings = get_settings()
    if not settings.sqs_queue_url:
        logger.info("sqs.consumer.disabled", reason="SQS_QUEUE_URL não configurado")
        return

    from app.infrastructure.messaging.sqs_consumer import start as sqs_start

    thread = threading.Thread(target=sqs_start, daemon=True, name="sqs-consumer")
    thread.start()
    logger.info("sqs.consumer.thread_started")


def _start_rabbitmq_consumer():
    """Consumer RabbitMQ para receber arquivos via webhook de teste."""
    from app.infrastructure.messaging.rabbitmq_consumer import start as rabbit_start

    rabbit_start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ia_service.startup")
    _start_sqs_consumer()
    _start_rabbitmq_consumer()
    yield
    logger.info("ia_service.shutdown")


app = FastAPI(
    title="IA Service — Hackathon FIAP",
    description="Pipeline de análise de diagramas de arquitetura com LLM Vision e RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    db_ok = check_db_connection()
    return {
        "status": "healthy" if db_ok else "degraded",
        "db": "connected" if db_ok else "unavailable",
    }


@app.post("/analyze", status_code=200)
async def analyze_diagram(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Recebe um diagrama de arquitetura (imagem ou PDF) diretamente via upload
    e executa o pipeline de análise sincronamente.
    Usado para testes — em produção o fluxo principal é via SQS.
    """
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: .{ext}. Aceitos: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()

    try:
        result = run_pipeline(
            db=db,
            file_bytes=file_bytes,
            file_name=file.filename,
        )
        return JSONResponse(status_code=200, content=result)

    except PipelineError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error("analyze.unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail="Erro interno no pipeline de análise.")


@app.post("/analyze/stream")
async def analyze_diagram_stream(
    file: UploadFile = File(...),
):
    """
    Endpoint SSE — executa o pipeline e emite eventos a cada etapa.
    Formato: text/event-stream com JSON por linha.

    Nota: não usa Depends(get_db) porque o pipeline roda em thread separada
    e a sessão do FastAPI seria fechada antes do thread terminar.
    A sessão é criada dentro do thread.
    """
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: .{ext}. Aceitos: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    file_name = file.filename
    event_queue: queue.Queue = queue.Queue()

    def _on_step(step: str, status: str, data: dict):
        event_queue.put({"step": step, "status": status, "data": data})

    def _run():
        from app.infrastructure.persistence.database import get_session_factory
        SessionLocal = get_session_factory()
        db = SessionLocal()
        try:
            run_pipeline(
                db=db,
                file_bytes=file_bytes,
                file_name=file_name,
                on_step=_on_step,
            )
        except Exception as exc:
            event_queue.put({
                "step": "pipeline",
                "status": "error",
                "data": {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            })
        finally:
            db.close()
            event_queue.put(_SENTINEL)

    # Roda o pipeline em thread separada para não bloquear o event loop
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _event_generator():
        while True:
            try:
                event = await asyncio.get_event_loop().run_in_executor(
                    None, event_queue.get, True, 300,  # timeout 5min
                )
            except queue.Empty:
                break

            if event is _SENTINEL:
                break

            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/analyses/{analysis_id}/status")
def get_status(analysis_id: str, db: Session = Depends(get_db)):
    """Consulta o status de processamento de uma análise."""
    from app.infrastructure.persistence.sqlalchemy_analysis_repository import SQLAlchemyAnalysisRepository
    from app.domain.shared.analysis_id import AnalysisId
    repo = SQLAlchemyAnalysisRepository(db)
    analysis = repo.get_by_id(AnalysisId.from_string(analysis_id))
    if not analysis:
        raise HTTPException(status_code=404, detail="Análise não encontrada.")
    return {
        "analysis_id": analysis_id,
        "status": analysis.status.value,
        "file_name": analysis.file_name,
        "error_message": analysis.error_message,
    }


# ── Async endpoints (Celery + Redis) ──────────────────────────────


@app.post("/analyze/async", status_code=202)
async def analyze_diagram_async(
    file: UploadFile = File(...),
):
    """
    Submete um diagrama para análise assíncrona via Celery.
    Retorna imediatamente com o job_id para acompanhamento.

    Acompanhe o progresso via SSE: GET /jobs/{job_id}/events
    Ou via polling: GET /jobs/{job_id}/status
    """
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: .{ext}. Aceitos: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    file_name = file.filename

    from app.infrastructure.celery.tasks import analyze_diagram_task

    task = analyze_diagram_task.delay(file_bytes.hex(), file_name)

    logger.info("analyze.async.submitted", job_id=task.id, file_name=file_name)

    return {"job_id": task.id, "status": "recebido"}


@app.get("/jobs/{job_id}/events")
async def job_events_sse(
    job_id: str,
    last_index: int = Query(0, ge=0, description="Índice do último evento recebido (para reconexão)"),
):
    """
    SSE endpoint — acompanha o progresso de um job em tempo real.

    Fase 1 (catch-up): envia eventos já armazenados no Redis list.
    Fase 2 (real-time): assina o canal Redis pub/sub para novos eventos.

    Suporta reconexão: passe ?last_index=N para pular eventos já recebidos.
    """
    settings = get_settings()
    r = redis_lib.from_url(settings.redis_url)
    channel = f"job:{job_id}"
    events_key = f"job:{job_id}:events"

    async def _generate():
        # Fase 1: catch-up — eventos já armazenados
        stored = r.lrange(events_key, last_index, -1)
        for raw in stored:
            decoded = raw.decode() if isinstance(raw, bytes) else raw
            yield f"data: {decoded}\n\n"
            event = json.loads(decoded)
            if event.get("step") == "done" or event.get("status") == "error":
                return

        # Fase 2: real-time — pub/sub para novos eventos
        pubsub = r.pubsub()
        pubsub.subscribe(channel)
        try:
            timeout_counter = 0
            max_timeout = 300  # 5 minutos máximo de espera
            while timeout_counter < max_timeout:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pubsub.get_message(timeout=1.0)
                )
                if msg and msg["type"] == "message":
                    timeout_counter = 0
                    data = msg["data"].decode() if isinstance(msg["data"], bytes) else msg["data"]
                    yield f"data: {data}\n\n"
                    event = json.loads(data)
                    if event.get("step") == "done" or event.get("status") == "error":
                        break
                else:
                    timeout_counter += 1
                await asyncio.sleep(0.1)
        finally:
            pubsub.unsubscribe(channel)
            pubsub.close()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    """
    Polling endpoint — retorna o último evento e status de um job.
    Alternativa leve ao SSE para clientes que não suportam streaming.
    """
    settings = get_settings()
    r = redis_lib.from_url(settings.redis_url)
    events_key = f"job:{job_id}:events"

    events_raw = r.lrange(events_key, 0, -1)

    if not events_raw:
        from app.infrastructure.celery.celery_app import celery_app

        result = celery_app.AsyncResult(job_id)
        return {
            "job_id": job_id,
            "finished": False,
            "celery_state": result.state,
            "total_events": 0,
            "last_event": None,
        }

    last = json.loads(events_raw[-1])
    finished = last.get("step") == "done" or last.get("status") == "error"

    return {
        "job_id": job_id,
        "finished": finished,
        "last_event": last,
        "total_events": len(events_raw),
    }


# ── Test endpoint: upload + publish to RabbitMQ ─────────────────────


def _publish_to_rabbitmq(payload: Any, routing_key: str, exchange: Optional[str]) -> dict:
    import pika

    settings = get_settings()
    rabbit_url = getattr(settings, "rabbitmq_url", None) or "amqp://hackathon:hackathon123@rabbitmq:5672/"
    target_exchange = exchange or getattr(settings, "rabbitmq_exchange", "reports.events")

    try:
        params = pika.URLParameters(rabbit_url)
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        channel.exchange_declare(exchange=target_exchange, exchange_type="topic", durable=True)

        body_bytes = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        channel.basic_publish(
            exchange=target_exchange,
            routing_key=routing_key,
            body=body_bytes,
            properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
        )
        connection.close()
    except Exception as exc:
        logger.error("test.rabbitmq.publish_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Falha ao publicar no RabbitMQ: {exc}")

    logger.info(
        "test.rabbitmq.published",
        exchange=target_exchange,
        routing_key=routing_key,
        size=len(body_bytes),
    )
    return {
        "ok": True,
        "exchange": target_exchange,
        "routing_key": routing_key,
        "bytes": len(body_bytes),
    }


@app.post("/test/rabbitmq/upload", status_code=202)
async def test_upload_to_rabbitmq(
    file: UploadFile = File(..., description="Diagrama (png/jpg/jpeg/gif/webp/pdf)"),
    routing_key: str = Query("diagram.uploaded", description="Routing key da mensagem"),
    exchange: Optional[str] = Query(None, description="Exchange (default: settings.rabbitmq_exchange)"),
):
    """
    Simula um webhook: recebe um arquivo, gera um `job_id`, publica no RabbitMQ
    e retorna imediatamente. O consumer publica eventos de progresso em Redis
    no canal `job:{job_id}`.

    Acompanhe o progresso via:
      - SSE:     GET /jobs/{job_id}/events
      - Polling: GET /jobs/{job_id}/status
    """
    import base64
    import uuid

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo não suportado: .{ext}. Aceitos: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    job_id = str(uuid.uuid4())
    payload: Any = {
        "job_id": job_id,
        "file_name": file.filename,
        "content_type": file.content_type,
        "file_b64": base64.b64encode(file_bytes).decode("ascii"),
    }
    publish_result = _publish_to_rabbitmq(payload, routing_key, exchange)

    return {
        "job_id": job_id,
        "status": "recebido",
        "exchange": publish_result["exchange"],
        "routing_key": publish_result["routing_key"],
    }


@app.post("/test/rabbitmq/publish", status_code=200)
def test_publish_rabbitmq(
    payload: Any = Body(..., description="JSON arbitrário a ser publicado no exchange"),
    routing_key: str = Query("report.created", description="Routing key da mensagem"),
    exchange: Optional[str] = Query(None, description="Exchange (default: settings.rabbitmq_exchange)"),
):
    """Publica uma mensagem JSON arbitrária no exchange RabbitMQ."""
    return _publish_to_rabbitmq(payload, routing_key, exchange)
