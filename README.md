# ia-service — Pipeline de Análise de Diagramas com IA

Serviço principal do time IADT. Recebe diagramas de arquitetura (imagem ou PDF), processa em um pipeline de 5 etapas com guardrails, RAG e QA, e devolve um relatório técnico estruturado via webhook.

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura Hexagonal](#2-arquitetura-hexagonal)
3. [Domínio (DDD)](#3-domínio-ddd)
4. [Estrutura de Pastas](#4-estrutura-de-pastas)
5. [Modos de Execução](#5-modos-de-execução)
6. [SQS Consumer](#6-sqs-consumer)
7. [Celery + Redis](#7-celery--redis)
8. [Pipeline de IA — 5 Etapas](#8-pipeline-de-ia--5-etapas)
9. [RAG com pgvector](#9-rag-com-pgvector)
10. [Guardrails e QA](#10-guardrails-e-qa)
11. [Webhook de Devolutiva](#11-webhook-de-devolutiva)
12. [Fine-Tuning](#12-fine-tuning)
13. [Schema do Banco](#13-schema-do-banco)
14. [Configuração de Ambiente](#14-configuração-de-ambiente)
15. [Como Executar](#15-como-executar)
16. [API Reference](#16-api-reference)
17. [Streamlit](#17-streamlit)
18. [Testes](#18-testes)
19. [Segurança](#19-segurança)
20. [Limitações e Decisões](#20-limitações-e-decisões)

---

## 1. Visão Geral

### O Problema

Empresas com sistemas distribuídos possuem dezenas de diagramas armazenados como imagens ou PDFs. Sua análise é feita manualmente, demanda tempo, depende de especialistas e não escala.

### A Solução

Pipeline de IA que:
- **Lê** o diagrama visualmente (LLM Vision multimodal — sem OCR)
- **Extrai** componentes, relacionamentos e padrões arquiteturais
- **Enriquece** com contexto de diagramas similares já processados (RAG via pgvector)
- **Classifica** riscos em 6 categorias com severidade
- **Gera** relatório técnico estruturado em JSON
- **Valida** com QA em duas fases (determinística + LLM)
- **Devolve** resultado via webhook

### Stack

| Camada | Tecnologia |
|---|---|
| Web framework | FastAPI + Uvicorn |
| ORM / DB | SQLAlchemy 2.x + PostgreSQL 16 |
| Vector DB | pgvector + LangChain PGVector |
| LLM | OpenAI (`gpt-4o`) ou compatível via `LLM_BASE_URL` |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Fila externa | AWS SQS (`boto3`) |
| Fila interna | Celery 5 + Redis 7 |
| Resiliência | `tenacity` (retry com backoff) |
| Logging | `structlog` (JSON) |
| UI de validação | Streamlit |
| Fine-tuning | `peft` (QLoRA) + `transformers` |
| Testes | `pytest` + Playwright (E2E TypeScript) |

---

## 2. Arquitetura Hexagonal

Adota **Arquitetura Hexagonal** (Ports & Adapters) com **modelagem tática DDD**.

```
+------------------------------------------------------+
|              INFRASTRUCTURE (Adapters)               |
|                                                      |
|  OpenAIVisionAdapter    SQLAlchemy*Repository        |
|  OpenAITextAdapter      PGVectorAdapter              |
|  SQSConsumer            WebhookSender                |
|  CeleryApp / Tasks      RedisClient                  |
|                                                      |
|    +----------------------------------------------+  |
|    |         APPLICATION (Use Cases + Ports)      |  |
|    |                                              |  |
|    |  AnalyzeDiagramUseCase                       |  |
|    |  RetrieveReportUseCase                       |  |
|    |                                              |  |
|    |  Ports: IVisionLLM, ITextLLM                 |  |
|    |         IVectorStore                         |  |
|    |         IAnalysisRepository                  |  |
|    |         IReportRepository                    |  |
|    |                                              |  |
|    |    +----------------------------------+      |  |
|    |    |     DOMAIN (Modelo Tático DDD)   |      |  |
|    |    |                                  |      |  |
|    |    |  AnalysisAggregate               |      |  |
|    |    |  ReportAggregate                 |      |  |
|    |    |  GuardrailService                |      |  |
|    |    |  Value Objects, Events           |      |  |
|    |    +----------------------------------+      |  |
|    +----------------------------------------------+  |
+------------------------------------------------------+
```

**Regra:** Infrastructure depende de Application, que depende de Domain. Domain não importa nada externo.

### Portas (interfaces)

```python
class IVisionLLM(ABC):
    def extract_components(self, diagram_file: DiagramFile) -> ExtractionResult: ...

class ITextLLM(ABC):
    def generate_report(self, extraction, rag_context) -> TechnicalReport: ...
    def evaluate_quality(self, extraction, report) -> QAScore: ...

class IVectorStore(ABC):
    def index(self, analysis_id, extraction) -> None: ...
    def retrieve_context(self, extraction, exclude_analysis_id) -> RagContext: ...
```

### Composition Root (DI)

Arquivo único de montagem: [app/infrastructure/composition_root.py](app/infrastructure/composition_root.py)

```python
def build_analyze_use_case(db: Session) -> AnalyzeDiagramUseCase:
    return AnalyzeDiagramUseCase(
        analysis_repo=SQLAlchemyAnalysisRepository(db),
        report_repo=SQLAlchemyReportRepository(db),
        vision_llm=OpenAIVisionAdapter(),
        text_llm=OpenAITextAdapter(),
        vector_store=PGVectorAdapter(db),
        guardrail_svc=GuardrailService(),
        input_guardrail=InputGuardrailService(),
        output_guardrail=OutputGuardrailService(),
    )
```

---

## 3. Domínio (DDD)

### Bounded Contexts

| Contexto | Responsabilidade | Agregado |
|---|---|---|
| **DiagramAnalysis** | Ciclo de vida da análise | `AnalysisAggregate` |
| **ReportGeneration** | Geração e validação do relatório | `ReportAggregate` |

### AnalysisAggregate — Máquina de Estados

```
RECEIVED --start_ingestion()--> PROCESSING --complete()--> ANALYZED
                                     |
                                     +--fail()--> ERROR
```

### Value Objects

| VO | Contexto | Descrição |
|---|---|---|
| `DiagramFile` | DiagramAnalysis | Arquivo validado (tipo, tamanho, base64) — imutável |
| `Component`, `Relationship`, `ArchitecturalPattern` | DiagramAnalysis | Elementos extraídos |
| `AnalysisId`, `ReportId` | Shared | UUIDs tipados |
| `RiskItem` | ReportGeneration | Risco com severidade e mitigação |
| `QAScore` | ReportGeneration | Score >= 0.6 para aprovação |
| `RagContext` | ReportGeneration | Contexto histórico do pgvector |

### Domain Events

| Evento | Quando |
|---|---|
| `DiagramReceivedEvent` | Análise criada |
| `DiagramIngestedEvent` | Arquivo validado |
| `ComponentsExtractedEvent` | LLM Vision extraiu componentes |
| `AnalysisCompletedEvent` | Pipeline finalizou com sucesso |
| `AnalysisFailedEvent` | Qualquer etapa falhou |
| `ReportGeneratedEvent` | Relatório gerado |
| `QAValidationCompletedEvent` | QA executado |

---

## 4. Estrutura de Pastas

```
ia-service/
├── Dockerfile
├── requirements.txt
├── finetuning-requirements.txt
├── .env.example
├── infrastructure/
│   ├── database/
│   │   ├── docker-compose.yml
│   │   └── init/                        # SQL de inicialização do banco
│   └── redis/
│       └── docker-compose.yml
├── docs/                                # Diagramas e documentação visual
├── streamlit-app/                       # UI de validação
├── tests/
│   ├── test_diagram_ingestion.py
│   ├── test_component_extraction.py
│   ├── test_risk_assessment.py
│   ├── test_quality_validation.py
│   └── e2e/                             # Playwright (TypeScript)
└── app/
    ├── main.py                          # FastAPI — 8 endpoints
    ├── domain/
    │   ├── diagram_analysis/            # Bounded Context
    │   ├── report_generation/           # Bounded Context
    │   └── shared/
    ├── application/
    │   ├── ports/                       # IVisionLLM, ITextLLM, IVectorStore
    │   └── use_cases/
    ├── infrastructure/
    │   ├── composition_root.py          # DI
    │   ├── config/settings.py
    │   ├── llm/
    │   │   ├── openai_adapter.py
    │   │   └── finetuning/              # QLoRA pipeline
    │   ├── vector_store/pgvector_adapter.py
    │   ├── persistence/
    │   ├── messaging/sqs_consumer.py
    │   ├── http/webhook_sender.py
    │   └── celery/
    ├── pipeline/                        # Delegação ao use case
    └── shared/                          # Exceções e logging
```

---

## 5. Modos de Execução

| Modo | Entrada | Quando usar |
|---|---|---|
| **SQS consumer** | mensagem SQS | Fluxo principal em produção |
| **Síncrono** | `POST /analyze` | Testes rápidos |
| **SSE streaming** | `POST /analyze/stream` | UI com progresso em tempo real |
| **Assíncrono (Celery)** | `POST /analyze/async` + `GET /jobs/{id}/events` | Alta carga |

---

## 6. SQS Consumer

Arquivo: [app/infrastructure/messaging/sqs_consumer.py](app/infrastructure/messaging/sqs_consumer.py)

Roda como **thread daemon** no startup do FastAPI.

```
SQS Queue
   | long polling (WaitTimeSeconds=20)
   v
Consumer Thread
   +-- Idempotência: sqs_message_id já existe? -> skip
   +-- Poison pill: ApproximateReceiveCount > 3? -> warn
   +-- Download S3: retry 3x (backoff exponencial)
   +-- run_pipeline(file_bytes, file_name)
   +-- delete_message()  <- somente após sucesso
   +-- send_webhook(callback_url, result)
```

**Mensagem SQS esperada:**
```json
{
  "file_name": "diagrama.png",
  "s3_url": "https://s3.amazonaws.com/...",
  "callback_url": "https://soat-api.example.com/webhook"
}
```

---

## 7. Celery + Redis

```
Cliente          ia-service         Redis           celery-worker
   |                 |                |                   |
   | POST /async     |                |                   |
   |---------------> | task.delay()   |                   |
   |                 |--------------->| push(task)        |
   | 202 {job_id}    |                |<------------------| pop + run_pipeline
   |<--------------- |                | pub job:{id}      |
   |                 |                | rpush job:{id}:events
   | GET /jobs/{id}/events            |                   |
   |---------------> | subscribe + catch-up               |
   | SSE stream      |<-------------- |                   |
```

**Canais Redis:**
- `job:{task_id}` — pub/sub (tempo real)
- `job:{task_id}:events` — list com TTL 10min (catch-up/reconexão)

---

## 8. Pipeline de IA — 5 Etapas

```
[arquivo binário]
      |
(0) Input Guardrails   -> sanitiza filename, detecta prompt injection
      |
(1) Ingestion          -> valida tipo/tamanho, converte para base64
      |
(2) Extraction         -> LLM Vision -> componentes, relacionamentos, padrões
      |
(2.5) Input Guardrail  -> valida schema da extração
      |
(3) RAG                -> indexa no pgvector, busca similares (non-blocking)
      |
(4) Report + Riscos    -> LLM + Output Guardrails -> relatório JSON
      |
(5) QA                 -> 2 fases de validação -> score >= 0.6
      |
[PostgreSQL] + [Webhook]
```

### Etapa 2 — Extraction

Envia o arquivo diretamente ao LLM como conteúdo multimodal (sem OCR). O modelo interpreta setas, caixas, relacionamentos e padrões semanticamente.

**Saída:**
```json
{
  "components": ["API Gateway", "Auth Service", "User DB"],
  "relationships": ["API Gateway -> Auth Service: valida JWT"],
  "patterns": ["Microservices", "API Gateway Pattern"],
  "raw_description": "O diagrama apresenta..."
}
```

### Etapa 4 — Riscos (6 categorias)

| Categoria | O que avalia |
|---|---|
| **SPOF** | Pontos únicos de falha sem redundância |
| **Segurança** | Ausência de autenticação, endpoints expostos |
| **Escalabilidade** | Gargalos, ausência de cache/filas |
| **Acoplamento** | Dependências síncronas excessivas |
| **Observabilidade** | Ausência de logs, métricas, tracing |
| **Resiliência** | Sem circuit breaker, retry, fallback |

### Etapa 5 — QA

**Fase 1 (determinística):** campos obrigatórios, grounding >= 80%.  
**Fase 2 (LLM):** completude (30%), consistência (40%), coerência (20%), qualidade (10%).  
**Score mínimo:** 0.6.

---

## 9. RAG com pgvector

```
ExtractionResult
  -> LangChain Document
  -> Embeddings (text-embedding-3-small)
  -> PGVector.add_documents()

query = raw_description + components + patterns
  -> similarity_search_with_score(k=3)
  -> distância coseno < 0.3 -> incluído no contexto
```

**Non-blocking:** falha no pgvector retorna `RagContext.empty()` e o pipeline continua.

**Índice HNSW** (criar após primeira análise):
```sql
CREATE INDEX idx_langchain_hnsw
  ON langchain_pg_embedding
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

---

## 10. Guardrails e QA

### Input
- Sanitização de filename (path traversal, caracteres perigosos)
- Detecção de prompt injection via regex
- Validação de schema da extração (max 200 componentes, 500 relacionamentos)

### Output
- Validação de schema (chaves obrigatórias, severidades válidas)
- Detecção de PII (CPF, CNPJ, email, IP, API keys) → `[REDACTED]`
- Filtro de conteúdo proibido

### Grounding
- Fase Report: max 20% de componentes inventados
- Fase QA: >= 80% dos componentes do relatório na extração

---

## 11. Webhook de Devolutiva

Arquivo: [app/infrastructure/http/webhook_sender.py](app/infrastructure/http/webhook_sender.py)

Retry: 3 tentativas com backoff (2s → 4s → 8s). Falha total não bloqueia o pipeline — resultado já persistido.

**Payload de sucesso:**
```json
{
  "analysis_id": "uuid",
  "status": "analisado",
  "report": { "components_identified": [...], "architectural_risks": [...], "recommendations": [...], "executive_summary": "...", "rag_used": true },
  "error_message": null,
  "completed_at": "2026-04-02T21:30:00Z"
}
```

---

## 12. Fine-Tuning

Base: [app/infrastructure/llm/finetuning/](app/infrastructure/llm/finetuning/)

```bash
# 1. Dependências (máquina com GPU)
pip install -r finetuning-requirements.txt

# 2. Gerar dados sintéticos
python -m app.infrastructure.llm.finetuning.data_generator \
  --api-key $ANTHROPIC_API_KEY --samples 50 --output ./data/raw_pairs.jsonl

# 3. Formatar
python -m app.infrastructure.llm.finetuning.data_formatter \
  --input ./data/raw_pairs.jsonl --output ./data --split 0.9

# 4. Treinar (GPU)
python -m app.infrastructure.llm.finetuning.train \
  --epochs 3 --output-dir ./output/report-lora-adapter \
  --push-to-hub --hub-model-id "seu-usuario/report-lora"
```

**Backends disponíveis (`REPORT_MODEL_BACKEND`):**

| Valor | Descrição |
|---|---|
| `langchain` | LangChain + LLM via `LLM_MODEL` (padrão) |
| `finetuned_api` | HuggingFace Inference API |
| `finetuned_local` | Adapter QLoRA local (requer GPU) |

---

## 13. Schema do Banco

```sql
CREATE TABLE analyses (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status          VARCHAR(20) NOT NULL DEFAULT 'recebido'
                        CHECK (status IN ('recebido', 'em_processamento', 'analisado', 'erro')),
    file_name       VARCHAR(255) NOT NULL,
    file_type       VARCHAR(10)  NOT NULL,
    s3_key          VARCHAR(512),
    sqs_message_id  VARCHAR(255),
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE extraction_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id     UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    components      JSONB NOT NULL DEFAULT '[]',
    relationships   JSONB NOT NULL DEFAULT '[]',
    patterns        JSONB NOT NULL DEFAULT '[]',
    raw_description TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE reports (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id           UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    components_identified JSONB NOT NULL DEFAULT '[]',
    architectural_risks   JSONB NOT NULL DEFAULT '[]',
    recommendations       JSONB NOT NULL DEFAULT '[]',
    executive_summary     TEXT,
    rag_used              BOOLEAN NOT NULL DEFAULT FALSE,
    qa_is_valid           BOOLEAN,
    qa_completeness_score FLOAT,
    qa_issues_found       JSONB DEFAULT '[]',
    qa_quality_notes      TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 14. Configuração de Ambiente

```bash
cp .env.example .env
# edite .env e preencha OPENAI_API_KEY
```

| Variável | Padrão | Descrição |
|---|---|---|
| `OPENAI_API_KEY` | — | **Obrigatória** — Vision, texto e embeddings |
| `ANTHROPIC_API_KEY` | `""` | Fine-tuning (data_generator) |
| `POSTGRES_CONNECTION_STRING` | `postgresql+psycopg://hackathon:hackathon123@localhost:5432/hackathon_db` | Connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Broker Celery + pub/sub |
| `SQS_QUEUE_URL` | `""` | Se vazio, consumer não inicia |
| `REPORT_MODEL_BACKEND` | `langchain` | `langchain`, `finetuned_api`, `finetuned_local` |
| `LLM_MODEL` | `gpt-4o` | Modelo LLM |
| `LLM_BASE_URL` | `""` | Override URL (Groq, etc.) |
| `AWS_REGION` | `us-east-1` | Região AWS |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` |

---

## 15. Como Executar

### Standalone (com banco e redis inclusos)

```bash
cd ia-service
cp .env.example .env   # preencha OPENAI_API_KEY
docker compose -f docker-compose.standalone.yml up --build
```

Sobe: `pgvector` + `redis` + `ia-service` + `celery-worker`.

### Desenvolvimento local (fora do Docker)

```bash
# 1. Suba as dependências
docker compose -f docker-compose.standalone.yml up pgvector redis

# 2. ia-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 3. Celery worker (outro terminal)
celery -A app.infrastructure.celery.celery_app worker --loglevel=info
```

### Verificar saúde

```bash
curl http://localhost:8000/health
# {"status": "healthy", "db": "connected"}
```

### Testar o pipeline

```bash
# Síncrono
curl -X POST http://localhost:8000/analyze -F "file=@docs/diagrama-exemplo1.png"

# Assíncrono
JOB=$(curl -s -X POST http://localhost:8000/analyze/async \
  -F "file=@docs/diagrama-exemplo1.png" | jq -r '.job_id')
curl -N http://localhost:8000/jobs/$JOB/events
```

---

## 16. API Reference

### `GET /health`
```json
{"status": "healthy", "db": "connected"}
```

### `POST /analyze` (síncrono)
**200:**
```json
{
  "analysis_id": "uuid",
  "status": "analisado",
  "report": { "components_identified": [...], "architectural_risks": [...], "recommendations": [...], "executive_summary": "...", "rag_used": false },
  "qa": { "is_valid": true, "completeness_score": 0.92, "issues_found": [] }
}
```

### `POST /analyze/stream` (SSE)
```
data: {"step": "ingestion", "status": "done", "data": {"file_type": "png", "elapsed": 0.1}}
data: {"step": "extraction", "status": "done", "data": {"components_count": 8, "elapsed": 3.2}}
...
```

### `POST /analyze/async` (Celery)
**202:** `{"job_id": "f3a2...", "status": "recebido"}`

### `GET /jobs/{job_id}/events?last_index=0` (SSE)
Fase 1: catch-up via Redis list. Fase 2: pub/sub em tempo real.

### `GET /jobs/{job_id}/status`
```json
{"job_id": "f3a2...", "finished": true, "last_event": {...}, "total_events": 12}
```

### `GET /analyses/{analysis_id}/status`
```json
{"analysis_id": "uuid", "status": "analisado", "file_name": "diagrama.png", "error_message": null}
```

---

## 17. Streamlit

Interface visual de validação em [streamlit-app/](streamlit-app/). Consome `POST /analyze/stream` diretamente.

```bash
# Já incluído no docker-compose.standalone.yml
# Acesso: http://localhost:8501
```

Funcionalidades: upload drag-and-drop, progresso SSE em tempo real, relatório renderizado, download JSON, histórico via sidebar.

---

## 18. Testes

### Unitários (pytest)

```bash
pip install -r requirements.txt
pytest tests/ -v
pytest tests/ --cov=app --cov-report=term-missing
```

| Arquivo | Cobre |
|---|---|
| `test_diagram_ingestion.py` | Validação de tipo, tamanho, base64 |
| `test_component_extraction.py` | Parsing JSON do LLM, campos obrigatórios |
| `test_risk_assessment.py` | Classificação de severidade |
| `test_quality_validation.py` | Grounding, score mínimo, fallback |

### E2E (Playwright + TypeScript)

```bash
cd tests/e2e
npm install
npx playwright test
npx playwright test --ui   # modo visual
```

---

## 19. Segurança

- Prompt injection detectada via regex antes de qualquer LLM call
- Sanitização de filename (path traversal, tamanho)
- Grounding check: max 20% componentes inventados
- QA score mínimo 0.6
- PII detectada e substituída por `[REDACTED]`
- Variáveis sensíveis via env vars — nunca hardcoded
- Graceful shutdown: `SIGTERM`/`SIGINT` aguardam processamento atual

---

## 20. Limitações e Decisões

**Por que não OCR?** LLMs Vision interpretam setas, caixas e padrões semanticamente. OCR extrairia apenas texto.

**Por que o LLM é configurável?** `LLM_MODEL` + `LLM_BASE_URL` permite trocar provider sem alterar código.

**Por que RAG é non-blocking?** Um diagrama pode ser analisado sem histórico (cold start). Bloquear quebraria o pipeline em cenários válidos.

**Por que riscos junto com o relatório?** Compartilham o mesmo contexto (extração + RAG). Unificar reduz latência, custo e garante coerência.

**Por que QA tem duas fases?** Fase 1 (determinística) é instantânea e sem custo de API. Fase 2 (LLM) avalia nuances. Separar evita chamar LLM para relatórios claramente inválidos.

**Por que Celery além do SQS?** SQS é o canal externo (SOAT → IADT). Celery/Redis é o canal interno para UIs que acompanham progresso em tempo real via SSE.
