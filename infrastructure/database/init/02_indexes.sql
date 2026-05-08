-- ============================================================
-- Índices relacionais
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_analyses_status
    ON analyses(status);

CREATE INDEX IF NOT EXISTS idx_analyses_created_at
    ON analyses(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_extraction_analysis_id
    ON extraction_results(analysis_id);

CREATE INDEX IF NOT EXISTS idx_reports_analysis_id
    ON reports(analysis_id);

-- ============================================================
-- Índice HNSW para a tabela gerenciada pelo LangChain (pgvector)
-- Criado APÓS o LangChain inicializar a tabela langchain_pg_embedding.
-- Execute este bloco manualmente após o primeiro uso do rag-agent.
-- ============================================================

-- DO $$
-- BEGIN
--     IF EXISTS (
--         SELECT 1 FROM information_schema.tables
--         WHERE table_name = 'langchain_pg_embedding'
--     ) THEN
--         CREATE INDEX IF NOT EXISTS idx_langchain_hnsw
--             ON langchain_pg_embedding
--             USING hnsw (embedding vector_cosine_ops)
--             WITH (m = 16, ef_construction = 64);
--     END IF;
-- END $$;

-- ============================================================
-- Nota: para criar o índice HNSW manualmente após subir o stack:
--
--   docker exec -it hackathon_pgvector psql -U hackathon -d hackathon_db -c "
--     CREATE INDEX idx_langchain_hnsw
--       ON langchain_pg_embedding
--       USING hnsw (embedding vector_cosine_ops)
--       WITH (m = 16, ef_construction = 64);"
-- ============================================================
