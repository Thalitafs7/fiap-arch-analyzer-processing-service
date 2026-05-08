"""
Shared Domain Service — Output Guardrails.

Valida e sanitiza as saídas do LLM antes de entregar ao usuário.
Detecta dados sensíveis (PII), conteúdo proibido e valida schema.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from app.shared.exceptions import GuardrailError
from app.shared.logging import get_logger

logger = get_logger(__name__)

REDACT_PLACEHOLDER = "[REDACTED]"

VALID_SEVERITIES = {"ALTO", "MÉDIO", "BAIXO"}

REPORT_REQUIRED_KEYS = {
    "components_identified",
    "architectural_risks",
    "recommendations",
    "executive_summary",
}

RISK_REQUIRED_KEYS = {
    "type",
    "description",
    "severity",
    "affected_components",
    "mitigation",
}


class OutputGuardrailService:
    """
    Serviço de domínio que valida saídas do LLM antes da entrega ao usuário.

    Regras aplicadas:
    1. Detecção e redação de dados sensíveis (PII)
    2. Filtro de conteúdo proibido
    3. Validação de schema do relatório
    """

    # ── PII Patterns ───────────────────────────────────────────────────

    PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
        # CPF brasileiro
        ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
        # CNPJ brasileiro
        ("CNPJ", re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")),
        # Email
        ("EMAIL", re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")),
        # Telefone BR (com DDD)
        ("TELEFONE", re.compile(r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-.\s]?\d{4}\b")),
        # IPv4
        ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
        # IPv6 (simplificado)
        ("IP", re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")),
        # API Keys / Tokens (padrões comuns)
        ("API_KEY", re.compile(r"\b(sk-[a-zA-Z0-9]{20,})\b")),
        ("API_KEY", re.compile(r"\b(ghp_[a-zA-Z0-9]{36,})\b")),
        ("API_KEY", re.compile(r"\b(gho_[a-zA-Z0-9]{36,})\b")),
        ("API_KEY", re.compile(r"\b(xoxb-[a-zA-Z0-9-]+)\b")),
        ("API_KEY", re.compile(r"\b(AKIA[0-9A-Z]{16})\b")),
        # Cartão de crédito (Luhn-like, 13-19 dígitos)
        ("CARTAO", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ]

    # ── Forbidden Content Patterns ─────────────────────────────────────

    FORBIDDEN_PATTERNS: List[Tuple[str, re.Pattern]] = [
        # Conteúdo discriminatório
        ("DISCRIMINATORIO", re.compile(
            r"\b(raça\s+superior|inferioridade\s+racial|supremacia)\b",
            re.IGNORECASE,
        )),
        # Instruções para atividades ilegais
        ("ILEGAL", re.compile(
            r"\b(como\s+hackear|invadir\s+sistema|exploit(ar)?\s+vulnerabilidade|"
            r"ataque\s+ddos|sql\s+injection\s+em\s+produção|"
            r"roubar\s+(dados|credenciais|senhas))\b",
            re.IGNORECASE,
        )),
        # Instruções de engenharia social
        ("ENGENHARIA_SOCIAL", re.compile(
            r"\b(phishing|enganar\s+usuários|falsificar\s+identidade)\b",
            re.IGNORECASE,
        )),
    ]

    # ── API Pública ────────────────────────────────────────────────────

    def check_sensitive_data(self, text: str) -> List[str]:
        """
        Verifica se o texto contém dados sensíveis (PII).
        Retorna lista de tipos de PII encontrados.
        """
        if not text:
            return []

        found_types: Set[str] = set()
        for pii_type, pattern in self.PII_PATTERNS:
            if pattern.search(text):
                found_types.add(pii_type)

        if found_types:
            logger.warning(
                "output_guardrail.sensitive_data_detected",
                pii_types=list(found_types),
            )

        return list(found_types)

    def redact_sensitive_data(self, text: str) -> str:
        """
        Substitui dados sensíveis por [REDACTED].
        Retorna o texto sanitizado.
        """
        if not text:
            return text

        redacted = text
        for pii_type, pattern in self.PII_PATTERNS:
            redacted = pattern.sub(REDACT_PLACEHOLDER, redacted)

        return redacted

    def redact_dict(self, data: dict) -> dict:
        """
        Aplica redação recursiva em todos os valores string de um dict.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.redact_sensitive_data(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.redact_dict(item) if isinstance(item, dict)
                    else self.redact_sensitive_data(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def check_forbidden_content(self, text: str) -> None:
        """
        Verifica se o texto contém conteúdo proibido.
        Levanta GuardrailError se detectar.
        """
        if not text:
            return

        for category, pattern in self.FORBIDDEN_PATTERNS:
            match = pattern.search(text)
            if match:
                logger.warning(
                    "output_guardrail.forbidden_content_detected",
                    category=category,
                    matched=match.group(),
                )
                raise GuardrailError(
                    f"Conteúdo proibido detectado na saída ({category}): '{match.group()}'",
                    step="output_guardrail",
                )

    def validate_report_schema(self, report_dict: dict) -> None:
        """
        Valida que o JSON do relatório segue o schema esperado.
        Verifica chaves obrigatórias, tipos e valores permitidos.
        """
        # Chaves obrigatórias no nível raiz
        missing = REPORT_REQUIRED_KEYS - set(report_dict.keys())
        if missing:
            raise GuardrailError(
                f"Relatório incompleto, chaves faltando: {missing}",
                step="output_guardrail",
            )

        # components_identified deve ser lista não-vazia de strings
        components = report_dict["components_identified"]
        if not isinstance(components, list) or not components:
            raise GuardrailError(
                "Campo 'components_identified' deve ser uma lista não-vazia.",
                step="output_guardrail",
            )
        if not all(isinstance(c, str) for c in components):
            raise GuardrailError(
                "Todos os itens de 'components_identified' devem ser strings.",
                step="output_guardrail",
            )

        # architectural_risks deve ser lista de dicts com schema correto
        risks = report_dict["architectural_risks"]
        if not isinstance(risks, list):
            raise GuardrailError(
                "Campo 'architectural_risks' deve ser uma lista.",
                step="output_guardrail",
            )
        for i, risk in enumerate(risks):
            if not isinstance(risk, dict):
                raise GuardrailError(
                    f"Risk [{i}] deve ser um dicionário.",
                    step="output_guardrail",
                )
            risk_missing = RISK_REQUIRED_KEYS - set(risk.keys())
            if risk_missing:
                raise GuardrailError(
                    f"Risk [{i}] incompleto, chaves faltando: {risk_missing}",
                    step="output_guardrail",
                )
            severity = risk.get("severity", "")
            if severity not in VALID_SEVERITIES:
                raise GuardrailError(
                    f"Risk [{i}] severidade inválida: '{severity}'. "
                    f"Valores permitidos: {VALID_SEVERITIES}",
                    step="output_guardrail",
                )

        # recommendations deve ser lista não-vazia de strings
        recs = report_dict["recommendations"]
        if not isinstance(recs, list) or not recs:
            raise GuardrailError(
                "Campo 'recommendations' deve ser uma lista não-vazia.",
                step="output_guardrail",
            )

        # executive_summary deve ser string com tamanho mínimo
        summary = report_dict.get("executive_summary", "")
        if not isinstance(summary, str) or len(summary) < 100:
            raise GuardrailError(
                f"Campo 'executive_summary' muito curto ({len(summary)} chars, mínimo 100).",
                step="output_guardrail",
            )

    def validate_output(self, report_dict: dict) -> dict:
        """
        Método principal: valida schema, checa conteúdo proibido e redacta PII.
        Retorna o dict sanitizado.
        """
        # 1. Validação de schema
        self.validate_report_schema(report_dict)

        # 2. Verificação de conteúdo proibido em campos textuais
        self.check_forbidden_content(report_dict.get("executive_summary", ""))
        for rec in report_dict.get("recommendations", []):
            if isinstance(rec, str):
                self.check_forbidden_content(rec)
        for risk in report_dict.get("architectural_risks", []):
            if isinstance(risk, dict):
                self.check_forbidden_content(risk.get("description", ""))
                self.check_forbidden_content(risk.get("mitigation", ""))

        # 3. Redação de PII
        sanitized = self.redact_dict(report_dict)

        logger.info("output_guardrail.validation_passed")
        return sanitized
