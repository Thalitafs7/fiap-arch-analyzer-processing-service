-- ============================================================
-- Tabela principal: registro de cada análise de diagrama
-- ============================================================
CREATE TABLE IF NOT EXISTS analyses (
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

-- ============================================================
-- Tabela: resultado da extração (extraction-agent)
-- ============================================================
CREATE TABLE IF NOT EXISTS extraction_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id     UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    components      JSONB  NOT NULL DEFAULT '[]',
    relationships   JSONB  NOT NULL DEFAULT '[]',
    patterns        JSONB  NOT NULL DEFAULT '[]',
    raw_description TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Tabela: relatório técnico gerado (report-agent)
-- ============================================================
CREATE TABLE IF NOT EXISTS reports (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_id           UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    components_identified JSONB NOT NULL DEFAULT '[]',
    architectural_risks   JSONB NOT NULL DEFAULT '[]',
    recommendations       JSONB NOT NULL DEFAULT '[]',
    executive_summary     TEXT,
    rag_used              BOOLEAN NOT NULL DEFAULT FALSE,
    -- QA fields
    qa_is_valid           BOOLEAN,
    qa_completeness_score FLOAT,
    qa_issues_found       JSONB  DEFAULT '[]',
    qa_quality_notes      TEXT,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Trigger: atualiza updated_at automaticamente
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER analyses_updated_at
    BEFORE UPDATE ON analyses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER reports_updated_at
    BEFORE UPDATE ON reports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
