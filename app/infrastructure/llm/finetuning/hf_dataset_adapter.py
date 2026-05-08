"""
HuggingFace Dataset Adapter — ingere e adapta o dataset ajibawa-2023/Software-Architecture
para o formato de treino do nosso pipeline de relatórios.

O dataset original tem ~450K amostras no formato instruction/input/output (Q&A genérico).
Este módulo:
  1. Carrega o dataset via HuggingFace Hub (streaming, sem download completo)
  2. Filtra amostras relevantes por tópico (riscos, padrões, componentes, segurança)
  3. Extrai entidades arquiteturais do texto (componentes, padrões, riscos)
  4. Converte para o formato messages do SFTTrainer com JSON estruturado
  5. Salva em JSONL compatível com o data_formatter.py

Uso:
    python -m app.infrastructure.llm.finetuning.hf_dataset_adapter \
        --output ./data/hf_adapted.jsonl \
        --max-samples 5000
"""

import argparse
import json
import re
import sys
from pathlib import Path

from app.infrastructure.llm.finetuning.prompts import SYSTEM_PROMPT, RISK_CATEGORIES


# ──────────────────────────────────────────────
# Filtros de relevância
# ──────────────────────────────────────────────

# Palavras-chave que indicam conteúdo relevante para análise de arquitetura
_RELEVANT_KEYWORDS = {
    # Padrões arquiteturais
    "microservices", "microserviços", "monolith", "monolito", "event-driven",
    "serverless", "service mesh", "api gateway", "cqrs", "event sourcing",
    "hexagonal", "clean architecture", "layered", "pipe", "publish-subscribe",
    "saga pattern", "circuit breaker", "load balancer", "message queue",
    "bff", "backend for frontend",

    # Componentes de infraestrutura
    "database", "cache", "redis", "kafka", "rabbitmq", "nginx",
    "kubernetes", "docker", "container", "lambda", "dynamodb", "s3",
    "elasticsearch", "prometheus", "grafana", "cdn", "dns",
    "postgres", "mongodb", "mysql",

    # Riscos e qualidade
    "single point of failure", "spof", "bottleneck", "scalability",
    "security", "vulnerability", "authentication", "authorization",
    "resilience", "fault tolerance", "high availability", "disaster recovery",
    "coupling", "decoupling", "observability", "monitoring", "logging",
    "latency", "throughput", "performance",

    # Análise e design
    "architecture review", "risk assessment", "threat model",
    "trade-off", "anti-pattern", "best practice", "design pattern",
    "distributed system", "cloud architecture", "infrastructure",
}

# Palavras-chave que indicam conteúdo NÃO relevante (código puro, tutoriais básicos)
_EXCLUDE_KEYWORDS = {
    "hello world", "print(", "def main", "import os",
    "installation guide", "getting started tutorial",
    "interview question", "exam preparation",
}

# Categorias de risco do nosso sistema
_RISK_TYPES = [
    "SPOF", "Segurança", "Escalabilidade",
    "Acoplamento", "Observabilidade", "Resiliência",
]

# Mapeamento de termos em inglês para categorias de risco
_RISK_MAPPING = {
    "single point of failure": "SPOF",
    "spof": "SPOF",
    "redundancy": "SPOF",
    "failover": "SPOF",
    "security": "Segurança",
    "authentication": "Segurança",
    "authorization": "Segurança",
    "encryption": "Segurança",
    "vulnerability": "Segurança",
    "injection": "Segurança",
    "xss": "Segurança",
    "csrf": "Segurança",
    "scalability": "Escalabilidade",
    "scaling": "Escalabilidade",
    "bottleneck": "Escalabilidade",
    "throughput": "Escalabilidade",
    "load balancing": "Escalabilidade",
    "horizontal scaling": "Escalabilidade",
    "coupling": "Acoplamento",
    "decoupling": "Acoplamento",
    "dependency": "Acoplamento",
    "tight coupling": "Acoplamento",
    "loose coupling": "Acoplamento",
    "observability": "Observabilidade",
    "monitoring": "Observabilidade",
    "logging": "Observabilidade",
    "tracing": "Observabilidade",
    "metrics": "Observabilidade",
    "alerting": "Observabilidade",
    "resilience": "Resiliência",
    "fault tolerance": "Resiliência",
    "circuit breaker": "Resiliência",
    "retry": "Resiliência",
    "fallback": "Resiliência",
    "disaster recovery": "Resiliência",
    "high availability": "Resiliência",
    "graceful degradation": "Resiliência",
}


def is_relevant(instruction: str, input_text: str, output: str) -> bool:
    """Filtra amostras relevantes para análise de arquitetura."""
    combined = f"{instruction} {input_text} {output}".lower()

    # Excluir conteúdo irrelevante
    for kw in _EXCLUDE_KEYWORDS:
        if kw in combined:
            return False

    # Precisa ter pelo menos 2 keywords relevantes
    matches = sum(1 for kw in _RELEVANT_KEYWORDS if kw in combined)
    return matches >= 2


# ──────────────────────────────────────────────
# Extração de entidades do texto
# ──────────────────────────────────────────────

# Padrões comuns de componentes em texto de arquitetura
_COMPONENT_PATTERNS = [
    r"\b(API Gateway|Load Balancer|Message Queue|Service Mesh|Database|Cache)\b",
    r"\b(Redis|Kafka|RabbitMQ|Nginx|HAProxy|Envoy|Istio)\b",
    r"\b(PostgreSQL|MySQL|MongoDB|DynamoDB|Cassandra|Elasticsearch)\b",
    r"\b(Kubernetes|Docker|Lambda|EC2|S3|CloudFront|Route53)\b",
    r"\b(Prometheus|Grafana|Kibana|Jaeger|Zipkin|Datadog)\b",
    r"\b(Auth(?:entication)? Service|User Service|Order Service|Payment Service|Notification Service)\b",
    r"\b(CDN|DNS|WAF|VPN|NAT Gateway|Bastion Host)\b",
    r"\b(Event Bus|Event Store|Message Broker|Task Queue|Worker)\b",
    r"\b(CI/CD Pipeline|Jenkins|GitHub Actions|ArgoCD)\b",
    r"\b(Microservice[s]?|Monolith|Backend|Frontend|BFF)\b",
]

_PATTERN_KEYWORDS = {
    "microservices": "Microsserviços",
    "microservice": "Microsserviços",
    "monolith": "Monolito",
    "monolithic": "Monolito",
    "event-driven": "Event-Driven",
    "event driven": "Event-Driven",
    "serverless": "Serverless",
    "service mesh": "Service Mesh",
    "cqrs": "CQRS",
    "event sourcing": "Event Sourcing",
    "hexagonal": "Arquitetura Hexagonal",
    "clean architecture": "Clean Architecture",
    "layered": "Arquitetura em Camadas",
    "pipe": "Pipes and Filters",
    "publish-subscribe": "Publish-Subscribe",
    "pub/sub": "Publish-Subscribe",
    "saga": "Saga Pattern",
    "bff": "Backend For Frontend",
    "domain-driven": "Domain-Driven Design",
    "ddd": "Domain-Driven Design",
}


def extract_components(text: str) -> list[str]:
    """Extrai componentes arquiteturais mencionados no texto."""
    components = set()
    for pattern in _COMPONENT_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        components.update(m.strip() for m in matches if len(m.strip()) > 2)

    return sorted(components)[:15]  # max 15 componentes


def extract_patterns(text: str) -> list[str]:
    """Extrai padrões arquiteturais mencionados no texto."""
    patterns = set()
    text_lower = text.lower()
    for keyword, pattern_name in _PATTERN_KEYWORDS.items():
        if keyword in text_lower:
            patterns.add(pattern_name)

    return sorted(patterns)[:5]


def extract_risks(text: str) -> list[dict]:
    """Extrai riscos arquiteturais mencionados no texto."""
    text_lower = text.lower()
    found_risks = {}

    for term, category in _RISK_MAPPING.items():
        if term in text_lower:
            if category not in found_risks:
                # Extrair contexto ao redor do termo
                idx = text_lower.find(term)
                start = max(0, idx - 50)
                end = min(len(text), idx + len(term) + 150)
                context = text[start:end].strip()

                found_risks[category] = {
                    "type": category,
                    "description": f"Risco identificado relacionado a {term}: {context[:200]}",
                    "severity": _estimate_severity(text_lower, term),
                    "affected_components": extract_components(text[max(0, idx - 200):min(len(text), idx + 300)])[:3],
                    "mitigation": _extract_mitigation(text, idx),
                }

    return list(found_risks.values())[:6]


def _estimate_severity(text: str, term: str) -> str:
    """Estima severidade baseado no contexto textual."""
    high_indicators = ["critical", "severe", "major", "dangerous", "fatal", "breach", "outage"]
    low_indicators = ["minor", "low", "slight", "minimal", "negligible"]

    # Janela de contexto ao redor do termo
    idx = text.find(term)
    window = text[max(0, idx - 100):min(len(text), idx + 200)]

    for indicator in high_indicators:
        if indicator in window:
            return "ALTO"
    for indicator in low_indicators:
        if indicator in window:
            return "BAIXO"

    return "MÉDIO"


def _extract_mitigation(text: str, risk_idx: int) -> str:
    """Tenta extrair sugestão de mitigação próxima ao risco."""
    mitigation_triggers = [
        "solution", "mitigat", "resolv", "fix", "address", "implement",
        "recommend", "should", "best practice", "consider", "use",
        "apply", "adopt", "ensure", "configure",
    ]

    # Procurar na janela após o risco
    window = text[risk_idx:min(len(text), risk_idx + 500)].lower()

    for trigger in mitigation_triggers:
        idx = window.find(trigger)
        if idx >= 0:
            # Extrair a frase
            sentence_start = idx
            sentence_end = window.find(".", idx)
            if sentence_end < 0:
                sentence_end = min(idx + 200, len(window))
            mitigation = text[risk_idx + sentence_start:risk_idx + sentence_end + 1].strip()
            if len(mitigation) > 20:
                return mitigation[:300]

    return "Avaliar e implementar controles adequados para mitigar este risco."


def extract_relationships(text: str, components: list[str]) -> list[str]:
    """Infere relacionamentos entre componentes mencionados no texto."""
    relationships = []
    text_lower = text.lower()

    # Padrões de relacionamento
    relation_patterns = [
        (r"(\w[\w\s]*?)\s+(?:connects?|communicates?|sends?|calls?|queries?)\s+(?:to|with)\s+(\w[\w\s]*?)[\.,;]", "comunica com"),
        (r"(\w[\w\s]*?)\s+(?:stores?|persists?|saves?|writes?)\s+(?:to|in|into)\s+(\w[\w\s]*?)[\.,;]", "persiste em"),
        (r"(\w[\w\s]*?)\s+(?:reads?|fetches?|retrieves?|gets?)\s+(?:from)\s+(\w[\w\s]*?)[\.,;]", "lê de"),
        (r"(\w[\w\s]*?)\s+(?:publishes?|emits?|produces?)\s+(?:to|events?)\s+(\w[\w\s]*?)[\.,;]", "publica para"),
    ]

    for pattern, rel_type in relation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for src, dst in matches[:3]:
            src = src.strip()[:30]
            dst = dst.strip()[:30]
            if len(src) > 2 and len(dst) > 2:
                relationships.append(f"{src} → {dst}: {rel_type}")

    # Se não encontrou relacionamentos explícitos, inferir dos componentes adjacentes
    if not relationships and len(components) >= 2:
        for i in range(min(len(components) - 1, 4)):
            relationships.append(f"{components[i]} → {components[i + 1]}: integração")

    return relationships[:8]


# ──────────────────────────────────────────────
# Construção do formato de treino
# ──────────────────────────────────────────────

def build_training_example(
    instruction: str,
    input_text: str,
    output: str,
) -> dict | None:
    """
    Converte uma amostra do dataset HF para o formato de treino do sistema.

    Retorna None se a amostra não tiver entidades suficientes para um par válido.
    """
    full_text = f"{instruction}\n{input_text}\n{output}"

    # Extrair entidades
    components = extract_components(full_text)
    if len(components) < 2:
        return None  # precisa de pelo menos 2 componentes

    patterns = extract_patterns(full_text)
    risks = extract_risks(full_text)
    relationships = extract_relationships(full_text, components)

    if not risks:
        return None  # precisa de pelo menos 1 risco

    # Construir extraction (input do modelo)
    extraction = {
        "components": components,
        "relationships": relationships,
        "patterns": patterns if patterns else ["Não identificado"],
        "raw_description": output[:500],
    }

    # Construir risk_result
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for r in risks:
        sev = r["severity"]
        if sev == "ALTO":
            severity_counts["high"] += 1
        elif sev == "BAIXO":
            severity_counts["low"] += 1
        else:
            severity_counts["medium"] += 1

    risk_result = {
        "risks": risks,
        "severity_summary": severity_counts,
    }

    # Construir report (output esperado do modelo)
    recommendations = _build_recommendations(risks, output)

    report = {
        "components_identified": components,
        "architectural_risks": risks,
        "recommendations": recommendations,
        "executive_summary": _build_summary(output, components, patterns),
        "rag_used": False,
    }

    return {
        "extraction": extraction,
        "risks": risk_result,
        "report": report,
        "rag_context": None,
        "source": "huggingface/ajibawa-2023/Software-Architecture",
        "tier": _estimate_tier(components),
    }


def _build_recommendations(risks: list[dict], output: str) -> list[str]:
    """Gera recomendações baseadas nos riscos extraídos."""
    recommendations = []

    for risk in risks:
        mitigation = risk.get("mitigation", "")
        if mitigation and len(mitigation) > 20:
            recommendations.append(
                f"[{risk['type']}] {mitigation}"
            )

    # Garantir mínimo de 3 recomendações
    generic_recs = [
        "Implementar health checks e alertas para componentes críticos, garantindo observabilidade adequada.",
        "Adotar circuit breaker pattern para chamadas entre serviços, aumentando a resiliência do sistema.",
        "Revisar estratégia de autenticação e autorização entre componentes, aplicando princípio de menor privilégio.",
        "Implementar testes de carga para identificar gargalos de escalabilidade sob demanda real.",
        "Documentar decisões arquiteturais (ADRs) e manter diagramas atualizados para facilitar manutenção.",
    ]

    idx = 0
    while len(recommendations) < 3 and idx < len(generic_recs):
        recommendations.append(generic_recs[idx])
        idx += 1

    return recommendations[:6]


def _build_summary(output: str, components: list[str], patterns: list[str]) -> str:
    """Constrói sumário executivo a partir do output original."""
    # Usar o início do output como base
    sentences = output.split(".")
    relevant_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) > 30:
            relevant_sentences.append(s)
        if len(relevant_sentences) >= 5:
            break

    base_summary = ". ".join(relevant_sentences) + "."

    if len(base_summary) < 100:
        pattern_str = ", ".join(patterns) if patterns else "não identificados"
        base_summary = (
            f"A arquitetura analisada apresenta {len(components)} componentes principais "
            f"com padrões arquiteturais: {pattern_str}. {base_summary}"
        )

    return base_summary[:800]


def _estimate_tier(components: list[str]) -> int:
    """Estima o tier de complexidade baseado no número de componentes."""
    n = len(components)
    if n <= 5:
        return 1
    if n <= 10:
        return 2
    if n <= 15:
        return 3
    return 4


# ──────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────

def adapt_dataset(
    output_path: str = "./data/hf_adapted.jsonl",
    max_samples: int = 5000,
    min_components: int = 2,
    streaming: bool = True,
) -> int:
    """
    Carrega, filtra e adapta o dataset HuggingFace para treino.

    Args:
        output_path:     Caminho do JSONL de saída.
        max_samples:     Máximo de amostras adaptadas a gerar.
        min_components:  Mínimo de componentes para aceitar uma amostra.
        streaming:       Se True, usa streaming (não baixa tudo de uma vez).

    Returns:
        Número de amostras adaptadas.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERRO: datasets não instalado. Execute: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Carregando dataset ajibawa-2023/Software-Architecture (streaming={streaming})...")
    ds = load_dataset(
        "ajibawa-2023/Software-Architecture",
        split="train",
        streaming=streaming,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    adapted = 0
    processed = 0
    filtered_out = 0
    extraction_failed = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in ds:
            if adapted >= max_samples:
                break

            processed += 1
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")

            # Filtro de relevância
            if not is_relevant(instruction, input_text, output):
                filtered_out += 1
                continue

            # Adaptação
            example = build_training_example(instruction, input_text, output)
            if example is None:
                extraction_failed += 1
                continue

            # Verificar mínimo de componentes
            if len(example["extraction"]["components"]) < min_components:
                extraction_failed += 1
                continue

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            adapted += 1

            if adapted % 100 == 0:
                print(
                    f"  [{adapted}/{max_samples}] processados={processed}, "
                    f"filtrados={filtered_out}, falhas={extraction_failed}",
                    flush=True,
                )

    print(f"\nResultado:")
    print(f"  Processados: {processed}")
    print(f"  Filtrados (irrelevantes): {filtered_out}")
    print(f"  Falhas (extração insuficiente): {extraction_failed}")
    print(f"  Adaptados: {adapted} → {output_path}")

    return adapted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adapta dataset HuggingFace Software-Architecture para fine-tuning"
    )
    parser.add_argument("--output", default="./data/hf_adapted.jsonl", help="Arquivo JSONL de saída")
    parser.add_argument("--max-samples", type=int, default=5000, help="Máximo de amostras")
    parser.add_argument("--min-components", type=int, default=2, help="Mínimo de componentes por amostra")
    parser.add_argument("--no-streaming", action="store_true", help="Desativar streaming (baixa tudo)")
    args = parser.parse_args()

    adapt_dataset(
        output_path=args.output,
        max_samples=args.max_samples,
        min_components=args.min_components,
        streaming=not args.no_streaming,
    )
