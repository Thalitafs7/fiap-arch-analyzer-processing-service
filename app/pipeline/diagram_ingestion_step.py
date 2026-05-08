"""
Ingestion Agent — valida o arquivo e prepara o payload para o extraction-agent.
Suporta leitura de bytes (S3) ou caminho local (testes).
"""

import base64
import mimetypes
from pathlib import Path
from app.shared.exceptions import IngestionError
from app.shared.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_TYPES = {
    "image/png":  "png",
    "image/jpeg": "jpeg",
    "image/jpg":  "jpeg",
    "application/pdf": "pdf"
}
MAX_SIZE_MB = 20
MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024


def run(file_bytes: bytes, file_name: str) -> dict:
    """
    Valida e converte o arquivo para base64.

    Args:
        file_bytes: conteúdo binário do arquivo (vindo do S3 ou upload direto)
        file_name:  nome original do arquivo (ex: 'diagrama.png')

    Returns:
        dict com status, file_name, file_type, media_type, content_base64, file_size_kb
    """
    logger.info("ingestion.start", file_name=file_name, size_kb=round(len(file_bytes) / 1024, 2))

    # 1. Validar tamanho
    if len(file_bytes) > MAX_SIZE_BYTES:
        raise IngestionError(
            f"Arquivo excede o limite de {MAX_SIZE_MB}MB.",
            step="ingestion",
        )

    # 2. Detectar MIME type pelo nome do arquivo
    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type not in SUPPORTED_TYPES:
        raise IngestionError(
            f"Tipo de arquivo não suportado: {mime_type}. "
            f"Aceitos: {', '.join(SUPPORTED_TYPES.keys())}",
            step="ingestion",
        )

    file_type = SUPPORTED_TYPES[mime_type]
    content_base64 = base64.standard_b64encode(file_bytes).decode("utf-8")

    result = {
        "status": "recebido",
        "file_name": file_name,
        "file_type": file_type,
        "media_type": mime_type,
        "content_base64": content_base64,
        "file_size_kb": round(len(file_bytes) / 1024, 2),
    }

    logger.info("ingestion.done", file_name=file_name, file_type=file_type)
    return result
