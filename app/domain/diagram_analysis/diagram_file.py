from __future__ import annotations
from dataclasses import dataclass

from app.domain.diagram_analysis.file_type import FileType


@dataclass(frozen=True)
class DiagramFile:
    """
    Representa o arquivo de diagrama após ingestão (validado e codificado).
    Value object imutável — não tem identidade própria.
    """
    file_name: str
    file_type: FileType
    media_type: str
    content_base64: str
    file_size_kb: float

    MAX_SIZE_MB: int = 20

    @classmethod
    def create(
        cls,
        file_bytes: bytes,
        file_name: str,
    ) -> "DiagramFile":
        """
        Factory method — valida e constrói DiagramFile a partir de bytes brutos.
        Levanta ValueError para entradas inválidas.
        """
        import base64
        import mimetypes

        max_bytes = cls.MAX_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"Arquivo excede o limite de {cls.MAX_SIZE_MB}MB."
            )

        mime_type, _ = mimetypes.guess_type(file_name)
        file_type = FileType.from_mime(mime_type or "")

        return cls(
            file_name=file_name,
            file_type=file_type,
            media_type=mime_type,
            content_base64=base64.standard_b64encode(file_bytes).decode("utf-8"),
            file_size_kb=round(len(file_bytes) / 1024, 2),
        )

    def to_dict(self) -> dict:
        return {
            "status": "recebido",
            "file_name": self.file_name,
            "file_type": self.file_type.value,
            "media_type": self.media_type,
            "content_base64": self.content_base64,
            "file_size_kb": self.file_size_kb,
        }
