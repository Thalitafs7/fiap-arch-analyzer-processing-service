"""
SOAT Client — atualiza o status de uma análise no sistema externo SOAT via HTTP PUT.

Política de retry:
  - Até 3 tentativas com exponential backoff (2s → 4s → 8s)
  - Retenta em: timeout, erros de conexão, respostas 5xx
  - NÃO retenta em: respostas 4xx (erro do cliente — log e segue)
  - Falha total nunca lança exceção: o pipeline não é bloqueado
"""

import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.infrastructure.config.settings import get_settings
from app.shared.logging import get_logger

logger = get_logger(__name__)
_stdlib_logger = logging.getLogger(__name__)


class _ServerError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


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
def _put_with_retry(url: str, payload: dict) -> bool:
    with httpx.Client(timeout=10.0) as client:
        response = client.put(url, json=payload)

    if response.status_code >= 500:
        raise _ServerError(response.status_code)

    if response.status_code >= 400:
        logger.warning(
            "soat.status_update.client_error",
            status=response.status_code,
            url=url[:80],
        )
        return False

    return True


def update_analysis_status(soat_analysis_id: str, status: str) -> bool:
    """
    Notifica o SOAT sobre a mudança de status de uma análise.

    PUT {SOAT_BASE_URL}/analyses/{soat_analysis_id}/status
    Body: {"status": status}

    Sempre retorna True/False — nunca lança exceção.

    Args:
        soat_analysis_id: ID da análise no sistema SOAT.
        status:           Novo status (ex: "em_processamento", "analisado", "erro").

    Returns:
        True se a atualização foi aceita, False caso contrário.
    """
    settings = get_settings()

    if not settings.soat_base_url:
        logger.debug("soat.status_update.skipped", reason="SOAT_BASE_URL não configurado")
        return False

    if not soat_analysis_id:
        logger.debug("soat.status_update.skipped", reason="soat_analysis_id ausente")
        return False

    url = f"{settings.soat_base_url.rstrip('/')}/analyses/{soat_analysis_id}/status"
    log = logger.bind(soat_analysis_id=soat_analysis_id, status=status)

    try:
        log.info("soat.status_update.sending")
        success = _put_with_retry(url, {"status": status})
        if success:
            log.info("soat.status_update.sent")
        return success

    except Exception as exc:
        log.error("soat.status_update.failed_all_retries", error=str(exc))
        return False
