"""
Webhook Sender — envia o resultado da análise via POST para o callback_url.

Política de retry:
  - Até 3 tentativas com exponential backoff (2s → 4s → 8s)
  - Retenta em: timeout, erros de conexão, respostas 5xx
  - NÃO retenta em: respostas 4xx (erro do cliente — log e segue)
  - Falha total nunca lança exceção: resultado já está no DB
"""

import logging
from datetime import datetime, timezone

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.shared.logging import get_logger

logger = get_logger(__name__)
_stdlib_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Exceção interna para respostas 5xx (retentável)
# ──────────────────────────────────────────────

class _ServerError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


# ──────────────────────────────────────────────
# Chamada HTTP com retry
# ──────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type((
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.NetworkError,
        _ServerError,
    )),
    before_sleep=before_sleep_log(_stdlib_logger, logging.WARNING),
    reraise=True,
)
def _post_with_retry(url: str, payload: dict) -> bool:
    """Executa o POST com timeout de 10s. Lança exceção retentável se necessário."""
    with httpx.Client(timeout=10.0) as client:
        response = client.post(url, json=payload)

    if response.status_code >= 500:
        raise _ServerError(response.status_code)

    if response.status_code >= 400:
        logger.warning(
            "webhook.client_error",
            status=response.status_code,
            url=url[:80],
        )
        return False  # 4xx: não retenta, mas não bloqueia pipeline

    return True


# ──────────────────────────────────────────────
# Interface pública
# ──────────────────────────────────────────────

def send_webhook(
    callback_url: str | None,
    analysis_id: str,
    status: str,
    report: dict | None = None,
    error_message: str | None = None,
    soat_analysis_id: str = "",
    error_step: str | None = None,
    error_type: str | None = None,
) -> bool:
    """
    Envia o resultado da análise para callback_url via POST.

    Sempre retorna True/False — nunca lança exceção.

    Args:
        callback_url:      URL de destino. Se None/vazio, no-op.
        analysis_id:       ID interno da análise processada.
        status:            "analisado" | "erro"
        report:            Relatório completo (quando status = "analisado").
        error_message:     Mensagem de erro legível (quando status = "erro").
        soat_analysis_id:  ID da análise no sistema SOAT.
        error_step:        Etapa do pipeline onde ocorreu o erro.
        error_type:        Classe da exceção que causou o erro.

    Returns:
        True se o webhook foi entregue com sucesso, False caso contrário.
    """
    if not callback_url:
        logger.debug("webhook.skipped", reason="callback_url ausente", analysis_id=analysis_id)
        return False

    payload = {
        "analysis_id": analysis_id,
        "soat_analysis_id": soat_analysis_id or None,
        "status": status,
        "report": report,
        "error_message": error_message,
        "error_step": error_step,
        "error_type": error_type,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    log = logger.bind(analysis_id=analysis_id, url=callback_url[:80])

    try:
        log.info("webhook.sending", status=status)
        success = _post_with_retry(callback_url, payload)
        if success:
            log.info("webhook.delivered")
        return success

    except Exception as exc:
        log.error("webhook.failed_all_retries", error=str(exc))
        return False
