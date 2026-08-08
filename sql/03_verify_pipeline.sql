-- Sanity checks to run after a sync + embedding pass.

-- 1. Documents by source type
SELECT source_type, count(*) AS documents, max(synced_at) AS last_sync
FROM weather_documents
GROUP BY source_type
ORDER BY documents DESC;

-- 2. Coverage: documents with and without current vectors
SELECT
    count(*) FILTER (WHERE e.document_id IS NOT NULL) AS embedded_documents,
    count(*) FILTER (WHERE e.document_id IS NULL)     AS pending_documents
FROM weather_documents d
LEFT JOIN (SELECT DISTINCT document_id FROM weather_embeddings) e
       ON e.document_id = d.id;

-- 3. Chunk counts per model, and the stored vector width
SELECT model_name, count(*) AS chunks, min(vector_dims(embedding)) AS dims
FROM weather_embeddings
GROUP BY model_name;

-- 4. Confirm the HNSW index exists
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'weather_embeddings';
