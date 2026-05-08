"""
Data Generator — gera pares de treino (input → relatório) usando um LLM como professor.

Estratégia:
  1. Para cada template de arquitetura, gera extraction_result sintético via LLM.
  2. Gera o risk_result correspondente via LLM.
  3. Gera o relatório gold-standard usando o MESMO system prompt do report_agent.
  4. Salva os pares em formato bruto para processamento pelo data_formatter.

O LLM professor garante que os dados de treino sigam exatamente o schema
esperado pelo pipeline, evitando inconsistências de formato.

Uso:
    python -m app.finetuning.data_generator --output ./data/raw_pairs.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────
# System prompts (espelham os agentes reais)
# ──────────────────────────────────────────────

_EXTRACTION_SYSTEM = """Você é um arquiteto de software sênior simulando a saída de um sistema de análise de diagramas.
Gere dados realistas de extração de um diagrama de arquitetura de software.
Retorne APENAS um JSON válido, sem texto adicional."""

_RISK_SYSTEM = """Você é um arquiteto de software sênior especializado em identificação de riscos arquiteturais.
Gere uma análise de riscos realista com base nos componentes fornecidos.
Retorne APENAS um JSON válido, sem texto adicional."""

from app.infrastructure.llm.finetuning.prompts import SYSTEM_PROMPT as _REPORT_SYSTEM
from app.infrastructure.llm.finetuning.prompts import build_user_message as _build_report_prompt

# Descrições dos templates de arquitetura para guiar a geração
# Organizados por tier de complexidade (ver config.DataGenerationConfig)
_TEMPLATE_DESCRIPTIONS = {
    # ── Tier 1: Simples (5-7 componentes) ──────────────────────────
    "static_website_cdn": "Site estático com S3, CloudFront CDN e Route53 para DNS",
    "simple_crud_api": "API REST simples com único banco PostgreSQL e cache Redis",
    "single_container_app": "Aplicação em container Docker único com banco gerenciado RDS",
    "basic_queue_worker": "API com fila SQS e worker único para processamento assíncrono",
    "wordpress_lamp": "Stack LAMP clássica com Apache, PHP, MySQL e servidor de arquivos",

    # ── Tier 2: Intermediário (8-12 componentes) ───────────────────
    "microservices_api_gateway": "Arquitetura de microsserviços com API Gateway, serviços de autenticação, catálogo de produtos e banco de dados por serviço",
    "monolith_single_db": "Aplicação monolítica com único banco de dados relacional, servidor web e cache Redis",
    "event_driven_kafka": "Arquitetura orientada a eventos com Kafka, múltiplos consumers, serviço de notificação e storage",
    "serverless_lambda_dynamodb": "Arquitetura serverless com AWS Lambda, API Gateway, DynamoDB e S3",
    "kubernetes_service_mesh": "Cluster Kubernetes com Istio service mesh, múltiplos deployments e Prometheus para observabilidade",
    "cqrs_event_sourcing": "Padrão CQRS com event sourcing, projeções read-only, event store e múltiplos read models",
    "bff_mobile_web": "Backend For Frontend com BFF separado para mobile e web, serviços compartilhados downstream",
    "data_pipeline_etl": "Pipeline ETL com ingestão de dados, processamento batch com Spark, data warehouse e dashboard",
    "hexagonal_clean_arch": "Aplicação com arquitetura hexagonal, ports e adapters, múltiplos adaptadores de entrada e saída",
    "multi_region_failover": "Arquitetura multi-região com failover automático, Route53, replicação de banco e CDN",
    "saga_pattern_distributed": "Transações distribuídas com padrão Saga, compensações, orquestrador central e múltiplos serviços",
    "api_composition_gateway": "API Gateway com composição de múltiplos serviços, agregação de respostas e circuit breaker",
    "strangler_fig_migration": "Migração gradual de monolito para microsserviços com Strangler Fig Pattern e proxy reverso",
    "blue_green_deployment": "Deploy blue-green com load balancer, health checks, rollback automático e canary releases",
    "feature_flag_service": "Plataforma de feature flags com serviço central, SDK client, cache distribuído e analytics",

    # ── Tier 3: Complexo (12-20 componentes) ───────────────────────
    "multi_cloud_hybrid": "Arquitetura multi-cloud AWS + Azure com VPN, replicação de dados, failover entre clouds e gateway unificado",
    "streaming_platform_kafka_flink": "Plataforma de streaming em tempo real com Kafka, Apache Flink, Elasticsearch, Kibana, schema registry e connectors",
    "ml_inference_pipeline": "Pipeline de ML com feature store, model registry, A/B testing, serving de modelos, monitoring de drift e retraining",
    "zero_trust_network": "Rede Zero Trust com mTLS, service mesh, OIDC provider, policy engine, WAF e segmentação de rede",
    "event_mesh_choreography": "Event mesh com múltiplos bounded contexts, coreografia de eventos, CQRS por contexto e eventual consistency",
    "polyglot_persistence": "Microsserviços com persistência poliglota: PostgreSQL, MongoDB, Redis, Elasticsearch, Neo4j e S3",
    "global_edge_computing": "Computação na edge com CDN, edge functions, central control plane, sincronização multi-região e cache distribuído",

    # ── Tier 4: Expert (20+ componentes) ───────────────────────────
    "banking_core_modernization": "Modernização de core bancário com mainframe legado, microsserviços, event sourcing, CQRS, compliance, auditoria e API banking",
    "healthcare_hipaa_platform": "Plataforma de saúde HIPAA-compliant com PHI handling, HL7 FHIR, multi-tenant, audit trail, criptografia end-to-end e backups",
    "autonomous_vehicle_platform": "Plataforma de veículos autônomos com edge computing, real-time processing, ML inference, telemetria, OTA updates e safety systems",
}


def _call_llm(client, prompt: str, system: str, model: str) -> dict:
    """Chama o LLM e parseia o JSON retornado."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return json.loads(raw)


def _generate_extraction(client, template: str, description: str, variation: int, model: str) -> dict:
    prompt = f"""Simule a extração de um diagrama de arquitetura do tipo: {description}
Variação {variation}/5 — use componentes e nomes ligeiramente diferentes das variações anteriores.

Retorne JSON com exatamente estas chaves:
{{
  "components": ["lista de 5-12 componentes identificados no diagrama"],
  "relationships": ["lista de 4-8 relacionamentos no formato 'ComponenteA → ComponenteB: descrição'"],
  "patterns": ["lista de 1-4 padrões arquiteturais identificados"],
  "raw_description": "descrição textual completa do diagrama em 2-3 parágrafos"
}}"""
    return _call_llm(client, prompt, _EXTRACTION_SYSTEM, model)


def _generate_risks(client, extraction: dict, model: str) -> dict:
    prompt = f"""Com base nos componentes e padrões abaixo, identifique riscos arquiteturais:

Componentes: {json.dumps(extraction['components'], ensure_ascii=False)}
Relacionamentos: {json.dumps(extraction['relationships'], ensure_ascii=False)}
Padrões: {json.dumps(extraction['patterns'], ensure_ascii=False)}

Retorne JSON com exatamente estas chaves:
{{
  "risks": [
    {{
      "type": "SPOF|Segurança|Escalabilidade|Acoplamento|Observabilidade|Resiliência",
      "description": "descrição do risco",
      "severity": "ALTO|MÉDIO|BAIXO",
      "affected_components": ["componentes afetados"],
      "mitigation": "sugestão de mitigação"
    }}
  ],
  "severity_summary": {{"high": 0, "medium": 0, "low": 0}}
}}

Identifique entre 2 e 6 riscos. severity_summary deve contar os totais."""
    return _call_llm(client, prompt, _RISK_SYSTEM, model)


def _generate_report(client, extraction: dict, risks: dict, model: str, rag_result: dict | None = None) -> dict:
    prompt = _build_report_prompt(extraction, risks, rag_result)
    return _call_llm(client, prompt, _REPORT_SYSTEM, model)


def generate(
    api_key: str,
    model: str,
    output_path: str,
    num_samples: int = 500,
    delay_seconds: float = 1.0,
) -> list[dict]:
    """
    Gera pares de treino sintéticos e salva em JSONL.

    Usa DataGenerationConfig para determinar variações por tier.
    30% das amostras Tier 3-4 incluem contexto RAG sintético.

    Args:
        api_key:       Chave de API do LLM professor.
        model:         ID do modelo a usar como professor.
        output_path:   Caminho do arquivo JSONL de saída.
        num_samples:   Número máximo de pares a gerar.
        delay_seconds: Pausa entre chamadas (respeitar rate limit).

    Returns:
        Lista de pares gerados.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise RuntimeError("anthropic não instalado. Execute: pip install anthropic")

    from app.infrastructure.llm.finetuning.config import DataGenerationConfig
    import random

    data_cfg = DataGenerationConfig()
    templates = list(_TEMPLATE_DESCRIPTIONS.items())
    pairs = []
    count = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for template_name, template_desc in templates:
            if count >= num_samples:
                break

            max_variations = data_cfg.get_variations(template_name)
            variations = min(max_variations, num_samples - count)
            tier = data_cfg.get_tier(template_name)

            for v in range(1, variations + 1):
                print(f"[{count + 1}/{num_samples}] {template_name} (tier {tier}) — variação {v}/{variations}", flush=True)

                try:
                    extraction = _generate_extraction(client, template_name, template_desc, v, model)
                    time.sleep(delay_seconds)

                    risks = _generate_risks(client, extraction, model)
                    time.sleep(delay_seconds)

                    # 30% das amostras de tier 3-4 incluem RAG sintético
                    rag_result = None
                    if tier >= 3 and random.random() < data_cfg.rag_sample_ratio:
                        rag_result = {
                            "has_context": True,
                            "rag_enrichment": (
                                "Análises similares anteriores identificaram riscos de SPOF "
                                "em componentes centralizados e recomendaram implementar "
                                "circuit breaker e retry com backoff exponencial."
                            ),
                        }

                    report = _generate_report(client, extraction, risks, model, rag_result)
                    time.sleep(delay_seconds)

                    pair = {
                        "template": template_name,
                        "variation": v,
                        "tier": tier,
                        "extraction": extraction,
                        "risks": risks,
                        "rag_context": rag_result,
                        "report": report,
                    }
                    pairs.append(pair)
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    f.flush()

                    count += 1

                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"  [WARN] Erro ao gerar par ({template_name} v{v}): {exc}", flush=True)
                    continue

    print(f"\n✅ {count} pares gerados → {output_path}", flush=True)
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de dados de treino para fine-tuning")
    parser.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"), help="Chave de API")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Modelo professor")
    parser.add_argument("--output", default="./data/raw_pairs.jsonl", help="Arquivo de saída")
    parser.add_argument("--samples", type=int, default=500, help="Número de pares")
    parser.add_argument("--delay", type=float, default=1.0, help="Pausa entre chamadas (s)")
    args = parser.parse_args()

    if not args.api_key:
        print("ERRO: defina --api-key ou a variável ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    generate(
        api_key=args.api_key,
        model=args.model,
        output_path=args.output,
        num_samples=args.samples,
        delay_seconds=args.delay,
    )
