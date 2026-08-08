-- Chunk-level embeddings of weather_documents.narrative_text.
-- Run after 01_setup_weather_documents.sql.
--
-- The VECTOR(n) width must match the embedding model exactly:
--   sentence-transformers/all-MiniLM-L6-v2   -> 384  (default for this app)
--   sentence-transformers/all-mpnet-base-v2  -> 768
--   BAAI/bge-small-en-v1.5                   -> 384
--   BAAI/bge-base-en-v1.5                    -> 768
--   BAAI/bge-large-en-v1.5                   -> 1024
-- Changing the model means changing this width and re-embedding everything.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id           TEXT PRIMARY KEY,          -- "<document_id>::<chunk_index>"
    document_id  TEXT NOT NULL REFERENCES weather_documents (id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    chunk_text   TEXT NOT NULL,
    embedding    VECTOR(384) NOT NULL,
    model_name   TEXT NOT NULL,
    content_hash TEXT NOT NULL,             -- hash of the source narrative at embed time
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index, model_name)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

-- HNSW with cosine ops, matching the `<=>` operator used by /weather/search.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding
    ON weather_embeddings USING hnsw (embedding vector_cosine_ops);

-- Verify
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
