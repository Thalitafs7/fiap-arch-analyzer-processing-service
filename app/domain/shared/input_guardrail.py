"""
Shared Domain Service — Input Guardrails.

Filtra prompt injection, valida formato dos dados e sanitiza inputs
antes que sejam interpolados em prompts de LLM.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

from app.shared.exceptions import GuardrailError
from app.shared.logging import get_logger

logger = get_logger(__name__)

# Tamanhos maximos para campos textuais
MAX_FILENAME_LENGTH = 255
MAX_TEXT_FIELD_LENGTH = 50_000
MAX_COMPONENTS = 200
MAX_RELATIONSHIPS = 500


class InputGuardrailService:
    """
    Serviço de domínio que protege o pipeline contra inputs maliciosos.

    Regras aplicadas:
    1. Detecção de prompt injection via padrões conhecidos
    2. Sanitização de nomes de arquivo (path traversal, caracteres perigosos)
    3. Sanitização de campos textuais (delimitadores de prompt, controle Unicode)
    4. Validação de formato/schema dos dados de extração
    """

    # ── Prompt Injection Patterns ──────────────────────────────────────

    INJECTION_PATTERNS: List[re.Pattern] = [
        # Instruções de override
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
        re.compile(r"forget\s+(everything|all|your)\s*(instructions|rules|prompts)?", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous|prior|above|your)\s*(instructions|rules|prompts)?", re.IGNORECASE),
        re.compile(r"override\s+(your|all|the)\s*(instructions|rules|prompts|settings)", re.IGNORECASE),
        re.compile(r"bypass\s+(your|all|the)\s*(instructions|rules|filters|safety|restrictions)", re.IGNORECASE),
        re.compile(r"do\s+not\s+follow\s+(your|the|any)\s*(instructions|rules|guidelines)", re.IGNORECASE),

        # Tentativas de role-play / identidade
        re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.IGNORECASE),
        re.compile(r"act\s+as\s+(a|an|if)\b", re.IGNORECASE),
        re.compile(r"pretend\s+(you\s+are|to\s+be)\b", re.IGNORECASE),
        re.compile(r"roleplay\s+as\b", re.IGNORECASE),
        re.compile(r"assume\s+the\s+role\s+of\b", re.IGNORECASE),

        # Tentativas de exfiltração
        re.compile(r"(reveal|show|print|output|display)\s+(your|the|system)\s*(prompt|instructions|rules)", re.IGNORECASE),
        re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)", re.IGNORECASE),
        re.compile(r"repeat\s+(your|the)\s*(system\s+)?(prompt|instructions)", re.IGNORECASE),

        # Delimitadores de prompt de modelos
        re.compile(r"<\|im_start\|>", re.IGNORECASE),
        re.compile(r"<\|im_end\|>", re.IGNORECASE),
        re.compile(r"\[INST\]", re.IGNORECASE),
        re.compile(r"\[/INST\]", re.IGNORECASE),
        re.compile(r"<<SYS>>", re.IGNORECASE),
        re.compile(r"<</SYS>>", re.IGNORECASE),
        re.compile(r"```\s*system", re.IGNORECASE),

        # Injeção via separadores
        re.compile(r"---+\s*(system|assistant|user)\s*:?\s*---+", re.IGNORECASE),
        re.compile(r"#{3,}\s*(system|new)\s*(prompt|instruction)", re.IGNORECASE),
    ]

    # ── Caracteres perigosos em nomes de arquivo ───────────────────────

    _SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._\-\s]")
    _PATH_TRAVERSAL_RE = re.compile(r"(\.\.[\\/]|[\\/]\.\.)")

    # ── Delimitadores de prompt para sanitização de texto ──────────────

    _PROMPT_DELIMITERS = [
        "<|im_start|>", "<|im_end|>",
        "[INST]", "[/INST]",
        "<<SYS>>", "<</SYS>>",
        "```system",
    ]

    # ── API Pública ────────────────────────────────────────────────────

    def check_prompt_injection(self, text: str) -> None:
        """
        Verifica se o texto contém padrões de prompt injection.
        Levanta GuardrailError se detectar.
        """
        if not text:
            return

        for pattern in self.INJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                logger.warning(
                    "input_guardrail.prompt_injection_detected",
                    matched_pattern=match.group(),
                    text_preview=text[:100],
                )
                raise GuardrailError(
                    f"Prompt injection detectado no input: '{match.group()}'",
                    step="input_guardrail",
                )

    def sanitize_filename(self, file_name: str) -> str:
        """
        Sanitiza nome de arquivo removendo path traversal e caracteres perigosos.
        Retorna o nome sanitizado.
        """
        if not file_name:
            raise GuardrailError(
                "Nome de arquivo vazio.",
                step="input_guardrail",
            )

        # Remove path traversal
        sanitized = self._PATH_TRAVERSAL_RE.sub("", file_name)

        # Extrai apenas o nome do arquivo (remove diretórios)
        sanitized = sanitized.replace("\\", "/").split("/")[-1]

        # Remove caracteres perigosos, mantendo apenas seguros
        name_part, _, ext = sanitized.rpartition(".")
        if not name_part:
            name_part = sanitized
            ext = ""

        name_part = self._SAFE_FILENAME_RE.sub("", name_part).strip()
        if ext:
            ext = self._SAFE_FILENAME_RE.sub("", ext).strip()

        if not name_part:
            raise GuardrailError(
                "Nome de arquivo inválido após sanitização.",
                step="input_guardrail",
            )

        sanitized = f"{name_part}.{ext}" if ext else name_part

        # Limita tamanho
        if len(sanitized) > MAX_FILENAME_LENGTH:
            sanitized = sanitized[:MAX_FILENAME_LENGTH]

        return sanitized

    def sanitize_text_field(self, text: str, max_length: int = MAX_TEXT_FIELD_LENGTH) -> str:
        """
        Sanitiza campos textuais removendo delimitadores de prompt
        e caracteres de controle Unicode.
        """
        if not text:
            return text

        sanitized = text

        # Remove delimitadores de prompt
        for delimiter in self._PROMPT_DELIMITERS:
            sanitized = sanitized.replace(delimiter, "")

        # Remove caracteres de controle Unicode (exceto newline, tab, space)
        sanitized = "".join(
            ch for ch in sanitized
            if ch in ("\n", "\t", " ") or not unicodedata.category(ch).startswith("C")
        )

        # Limita tamanho
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    def validate_extraction_data(self, data: dict) -> None:
        """
        Valida o schema do JSON retornado pela extração do LLM de visão.
        Garante tipos corretos e tamanhos dentro do limite.
        """
        # Chaves obrigatórias
        required_keys = {"components", "relationships", "patterns", "raw_description"}
        missing = required_keys - set(data.keys())
        if missing:
            raise GuardrailError(
                f"Dados de extração incompletos, chaves faltando: {missing}",
                step="input_guardrail",
            )

        # Tipos corretos
        if not isinstance(data["components"], list):
            raise GuardrailError(
                "Campo 'components' deve ser uma lista.",
                step="input_guardrail",
            )
        if not isinstance(data["relationships"], list):
            raise GuardrailError(
                "Campo 'relationships' deve ser uma lista.",
                step="input_guardrail",
            )
        if not isinstance(data["patterns"], list):
            raise GuardrailError(
                "Campo 'patterns' deve ser uma lista.",
                step="input_guardrail",
            )
        if not isinstance(data["raw_description"], str):
            raise GuardrailError(
                "Campo 'raw_description' deve ser uma string.",
                step="input_guardrail",
            )

        # Limites de tamanho
        if len(data["components"]) > MAX_COMPONENTS:
            raise GuardrailError(
                f"Extração com {len(data['components'])} componentes excede o limite de {MAX_COMPONENTS}.",
                step="input_guardrail",
            )
        if len(data["relationships"]) > MAX_RELATIONSHIPS:
            raise GuardrailError(
                f"Extração com {len(data['relationships'])} relacionamentos excede o limite de {MAX_RELATIONSHIPS}.",
                step="input_guardrail",
            )

        # Verifica injection nos componentes extraídos
        for component in data["components"]:
            if isinstance(component, str):
                self.check_prompt_injection(component)

        # Verifica injection na descrição
        self.check_prompt_injection(data.get("raw_description", ""))
