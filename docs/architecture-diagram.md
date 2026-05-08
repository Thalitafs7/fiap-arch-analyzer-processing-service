# Diagrama de Arquitetura — Hackathon FIAP

## Visão Geral do Sistema

```mermaid
graph TB
    subgraph CLIENT["🖥️ Cliente"]
        USER["👤 Usuário"]
    end

    subgraph FRONTEND["🎨 Frontend — Streamlit :8501"]
        ST_APP["Streamlit App<br/>(app.py)"]
        ST_UPLOAD["Upload de Diagrama<br/>PNG / JPEG / PDF"]
        ST_SSE["SSE Client<br/>(progresso real-time)"]
        ST_HISTORY["Histórico de Análises"]
        ST_REPORT["Renderização<br/>de Relatório"]
    end

    subgraph IA_SERVICE["⚙️ IA Service — FastAPI :8000"]
        direction TB
        API_SYNC["POST /analyze<br/>(síncrono)"]
        API_STREAM["POST /analyze/stream<br/>(SSE)"]
        API_ASYNC["POST /analyze/async<br/>(Celery)"]
        API_JOBS["GET /jobs/{id}/events<br/>GET /jobs/{id}/status"]
        API_STATUS["GET /analyses/{id}/status"]
        HEALTH_IA["GET /health"]
    end

    subgraph PIPELINE["🔄 Pipeline de Análise (DDD)"]
        direction TB
        ORCHESTRATOR["AnalyzeDiagramUseCase<br/>(Application Layer)"]
        
        subgraph STEPS["Etapas do Pipeline"]
            direction LR
            S0["🛡️ Input<br/>Guardrails"]
            S1["📁 Ingestão"]
            S1_5["🏷️ Classificação<br/>(Vision LLM)"]
            S2["🔍 Extração<br/>(Vision LLM)"]
            S2_5["🛡️ Validação<br/>Extração"]
            S3["🔗 RAG<br/>(pgvector)"]
            S4["📝 Relatório<br/>+ Riscos<br/>(Text LLM)"]
            S4_5["🛡️ Output<br/>Guardrails"]
            S5["✅ QA<br/>(Text LLM)"]
        end

        S0 --> S1 --> S1_5 --> S2 --> S2_5 --> S3 --> S4 --> S4_5 --> S5
        S5 -.->|"❌ Rejeitado<br/>(max 2x)"| S4
    end

    subgraph INFRA_LLM["🤖 LLM Adapters"]
        VISION_LLM["OpenAI Vision Adapter<br/>(classificação + extração)"]
        TEXT_LLM["OpenAI Text Adapter<br/>(relatório + QA)"]
    end

    subgraph REPORT_API["📊 Report API — FastAPI :8001"]
        REPORT_LIST["GET /reports"]
        REPORT_GET["GET /reports/{id}"]
        HEALTH_REPORT["GET /health"]
    end

    subgraph ASYNC_LAYER["⚡ Processamento Assíncrono"]
        CELERY_WORKER["Celery Worker<br/>(concurrency=2)"]
        REDIS[("Redis :6379<br/>Broker + Pub/Sub<br/>+ Event Store")]
    end

    subgraph DATA_LAYER["🗄️ Persistência"]
        PGVECTOR[("PostgreSQL + pgvector :5432")]
        subgraph TABLES["Tabelas"]
            T_ANALYSES["analyses<br/>(status, file_name, ...)"]
            T_EXTRACTION["extraction_results<br/>(components, relationships, patterns)"]
            T_REPORTS["reports<br/>(risks, recommendations, QA)"]
            T_EMBEDDINGS["embeddings<br/>(vector similarity search)"]
        end
    end

    subgraph EXTERNAL["☁️ Integrações Externas (Opcional)"]
        SQS["AWS SQS<br/>(fila de entrada)"]
        S3["AWS S3<br/>(armazenamento)"]
        WEBHOOK["Webhook Callback<br/>(devolutiva)"]
    end

    subgraph DOMAIN["📦 Domain Layer (DDD)"]
        direction LR
        AGG_ANALYSIS["AnalysisAggregate"]
        AGG_REPORT["ReportAggregate"]
        VO["Value Objects<br/>DiagramFile, ExtractionResult,<br/>TechnicalReport, QAScore,<br/>RagContext, Risk"]
        EVENTS["Domain Events<br/>DiagramReceived, ComponentsExtracted,<br/>ReportGenerated, QACompleted"]
        GUARDRAILS["Guardrail Services<br/>Input / Output / Report"]
    end

    %% ── Fluxos principais ──
    USER --> ST_APP
    ST_APP --> ST_UPLOAD
    ST_UPLOAD -->|"POST /analyze/async"| API_ASYNC
    API_ASYNC -->|"dispatch task"| CELERY_WORKER
    CELERY_WORKER -->|"executa"| ORCHESTRATOR
    ST_SSE -->|"GET /jobs/{id}/events"| API_JOBS
    API_JOBS -->|"pub/sub"| REDIS
    CELERY_WORKER -->|"publish events"| REDIS

    ST_HISTORY -->|"GET /reports"| REPORT_LIST
    REPORT_API --> PGVECTOR

    %% ── Pipeline interno ──
    ORCHESTRATOR --> STEPS
    ORCHESTRATOR --> VISION_LLM
    ORCHESTRATOR --> TEXT_LLM
    ORCHESTRATOR --> PGVECTOR

    %% ── Fluxo SQS (opcional) ──
    SQS -.->|"consumer thread"| IA_SERVICE
    S3 -.->|"download pré-assinado"| IA_SERVICE
    IA_SERVICE -.->|"webhook resultado"| WEBHOOK

    %% ── Persistência ──
    IA_SERVICE --> PGVECTOR
    CELERY_WORKER --> REDIS

    %% ── LLM externo ──
    VISION_LLM -->|"API Call"| OPENAI_API["OpenAI API<br/>(gpt-4o)"]
    TEXT_LLM -->|"API Call"| OPENAI_API

    %% ── Estilos ──
    classDef frontend fill:#4CAF50,stroke:#333,color:#fff
    classDef backend fill:#2196F3,stroke:#333,color:#fff
    classDef pipeline fill:#FF9800,stroke:#333,color:#fff
    classDef data fill:#9C27B0,stroke:#333,color:#fff
    classDef external fill:#607D8B,stroke:#333,color:#fff
    classDef llm fill:#E91E63,stroke:#333,color:#fff

    class ST_APP,ST_UPLOAD,ST_SSE,ST_HISTORY,ST_REPORT frontend
    class API_SYNC,API_STREAM,API_ASYNC,API_JOBS,API_STATUS,HEALTH_IA backend
    class REPORT_LIST,REPORT_GET,HEALTH_REPORT backend
    class S0,S1,S1_5,S2,S2_5,S3,S4,S4_5,S5,ORCHESTRATOR pipeline
    class PGVECTOR,REDIS,T_ANALYSES,T_EXTRACTION,T_REPORTS,T_EMBEDDINGS data
    class SQS,S3,WEBHOOK external
    class VISION_LLM,TEXT_LLM,OPENAI_API llm
```

## Fluxo Principal (Upload via Streamlit)

```mermaid
sequenceDiagram
    actor User as 👤 Usuário
    participant ST as 🎨 Streamlit
    participant API as ⚙️ IA Service
    participant Celery as ⚡ Celery Worker
    participant Redis as 🔴 Redis
    participant LLM as 🤖 OpenAI (gpt-4o)
    participant PG as 🗄️ PostgreSQL + pgvector
    participant Report as 📊 Report API

    User->>ST: Upload diagrama (PNG/PDF)
    ST->>API: POST /analyze/async (file)
    API->>Celery: dispatch task (file_hex, name)
    API-->>ST: 202 { job_id }

    ST->>API: GET /jobs/{job_id}/events (SSE)
    API->>Redis: SUBSCRIBE job:{id}

    rect rgb(255, 243, 224)
        Note over Celery,PG: Pipeline de Análise
        
        Celery->>Redis: publish(input_guardrail: running)
        Celery->>Celery: Sanitize filename + check injection
        Celery->>Redis: publish(input_guardrail: done)

        Celery->>Redis: publish(ingestion: running)
        Celery->>Celery: Validar arquivo (tipo, tamanho)
        Celery->>PG: INSERT analyses
        Celery->>Redis: publish(ingestion: done)

        Celery->>Redis: publish(classification: running)
        Celery->>LLM: Vision → classificar imagem
        LLM-->>Celery: { is_architecture_diagram, confidence }
        Celery->>Redis: publish(classification: done)

        Celery->>Redis: publish(extraction: running)
        Celery->>LLM: Vision → extrair componentes
        LLM-->>Celery: { components, relationships, patterns }
        Celery->>PG: INSERT extraction_results
        Celery->>Redis: publish(extraction: done)

        Celery->>Redis: publish(rag: running)
        Celery->>PG: pgvector similarity search
        PG-->>Celery: contexto histórico similar
        Celery->>Redis: publish(rag: done)

        Celery->>Redis: publish(report: running)
        Celery->>LLM: Text → gerar relatório + riscos
        LLM-->>Celery: { summary, risks, recommendations }
        Celery->>Celery: Output guardrails (schema, PII, conteúdo)
        Celery->>Redis: publish(report: done)

        Celery->>Redis: publish(qa: running)
        Celery->>Celery: Verificações determinísticas
        Celery->>LLM: Text → avaliar qualidade
        LLM-->>Celery: { is_valid, score, issues }
        
        alt QA rejeitou (tentativa < 2)
            Celery->>Redis: publish(qa: refinement)
            Celery->>LLM: Re-gerar com feedback
        end

        Celery->>PG: INSERT reports (com QA)
        Celery->>Redis: publish(qa: done)
    end

    Celery->>Redis: publish(done: complete, result)
    Redis-->>API: event stream
    API-->>ST: SSE events (real-time)
    ST-->>User: Renderiza relatório

    User->>ST: Ver histórico
    ST->>Report: GET /reports
    Report->>PG: SELECT reports
    PG-->>Report: lista
    Report-->>ST: JSON
    ST-->>User: Exibe histórico
```

## Fluxo Alternativo (AWS SQS)

```mermaid
sequenceDiagram
    participant SOAT as 🏢 Time SOAT
    participant SQS as 📨 AWS SQS
    participant S3 as 📦 AWS S3
    participant Consumer as ⚙️ SQS Consumer Thread
    participant Pipeline as 🔄 Pipeline
    participant Webhook as 🔔 Callback URL

    SOAT->>SQS: Envia mensagem { file_name, s3_url, callback_url }
    Consumer->>SQS: Long polling (20s)
    SQS-->>Consumer: Mensagem recebida
    Consumer->>S3: Download via URL pré-assinada
    S3-->>Consumer: file_bytes
    Consumer->>Pipeline: run_pipeline(file_bytes, file_name)
    Pipeline-->>Consumer: resultado
    Consumer->>Webhook: POST callback_url (resultado)
    Consumer->>SQS: delete_message (ACK)
```

## Modelo de Dados

```mermaid
erDiagram
    ANALYSES {
        uuid id PK
        varchar status "recebido | em_processamento | analisado | erro"
        varchar file_name
        varchar file_type
        varchar s3_key "nullable"
        varchar sqs_message_id "nullable"
        text error_message "nullable"
        timestamptz created_at
        timestamptz updated_at
    }

    EXTRACTION_RESULTS {
        uuid id PK
        uuid analysis_id FK
        jsonb components
        jsonb relationships
        jsonb patterns
        text raw_description
        timestamptz created_at
    }

    REPORTS {
        uuid id PK
        uuid analysis_id FK
        jsonb components_identified
        jsonb architectural_risks
        jsonb recommendations
        text executive_summary
        boolean rag_used
        boolean qa_is_valid
        float qa_completeness_score
        jsonb qa_issues_found
        text qa_quality_notes
        timestamptz created_at
        timestamptz updated_at
    }

    ANALYSES ||--o| EXTRACTION_RESULTS : "has"
    ANALYSES ||--o| REPORTS : "has"
```

## Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| Frontend | Streamlit (Python) |
| Backend API | FastAPI + Uvicorn |
| Processamento Async | Celery + Redis (broker + pub/sub) |
| LLM | OpenAI gpt-4o (Vision + Text) |
| Vector Store / RAG | PostgreSQL + pgvector |
| Banco de Dados | PostgreSQL 16 |
| Mensageria (opcional) | AWS SQS + S3 |
| Containerização | Docker Compose (6 serviços) |
| Arquitetura | DDD (Domain-Driven Design) com Hexagonal/Ports & Adapters |
