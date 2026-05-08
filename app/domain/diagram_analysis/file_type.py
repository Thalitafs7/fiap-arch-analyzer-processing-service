from __future__ import annotations
from enum import Enum


class FileType(str, Enum):
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"
    PDF = "pdf"

    @classmethod
    def from_mime(cls, mime_type: str) -> "FileType":
        mapping = {
            "image/png": cls.PNG,
            "image/jpeg": cls.JPEG,
            "image/jpg": cls.JPEG,
            "image/gif": cls.GIF,
            "image/webp": cls.WEBP,
            "application/pdf": cls.PDF,
        }
        result = mapping.get(mime_type)
        if result is None:
            raise ValueError(f"MIME type não suportado: {mime_type}")
        return result

    @property
    def mime_type(self) -> str:
        mapping = {
            FileType.PNG: "image/png",
            FileType.JPEG: "image/jpeg",
            FileType.GIF: "image/gif",
            FileType.WEBP: "image/webp",
            FileType.PDF: "application/pdf",
        }
        return mapping[self]
