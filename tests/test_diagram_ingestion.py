import base64
import pytest
from app.pipeline.diagram_ingestion_step import run
from app.shared.exceptions import IngestionError

PNG_1PX = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_ingestion_valid_png():
    result = run(PNG_1PX, "diagrama.png")

    assert result["status"] == "recebido"
    assert result["file_name"] == "diagrama.png"
    assert result["file_type"] == "png"
    assert result["media_type"] == "image/png"
    assert result["content_base64"] == base64.standard_b64encode(PNG_1PX).decode()
    assert result["file_size_kb"] > 0


def test_ingestion_unsupported_extension():
    with pytest.raises(IngestionError, match="não suportado"):
        run(b"data", "arquivo.txt")


def test_ingestion_file_too_large():
    big_file = b"x" * (21 * 1024 * 1024)  # 21 MB
    with pytest.raises(IngestionError, match="limite"):
        run(big_file, "grande.png")


def test_ingestion_pdf():
    minimal_pdf = b"%PDF-1.4\n%%EOF"
    result = run(minimal_pdf, "doc.pdf")
    assert result["file_type"] == "pdf"
    assert result["media_type"] == "application/pdf"
