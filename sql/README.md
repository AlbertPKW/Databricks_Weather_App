# SQL setup for Lakebase

Two ways to create the schema — pick one, they produce the same tables:

- **From Python:** `python lakebase.py`, or just start the app (every route that
  writes calls `lakebase.ensure_weather_tables()` first). All statements are
  `IF NOT EXISTS`, so it's safe to re-run.
- **By hand:** run the files below in a SQL editor connected to your Lakebase
  instance, in order.

| File | What it creates |
|---|---|
| `01_setup_weather_documents.sql` | `weather_documents` + lookup indexes |
| `02_setup_weather_embeddings.sql` | `pgvector` extension, `weather_embeddings` with `VECTOR(384)`, HNSW cosine index |
| `03_verify_pipeline.sql` | Read-only checks: row counts, embedding coverage, vector width, index presence |

## Vector width

`VECTOR(384)` matches `sentence-transformers/all-MiniLM-L6-v2`. If you swap the
model, change the width in `02_setup_weather_embeddings.sql`, set `EMBEDDING_DIM`
to match, drop the table, and re-embed — pgvector rejects a vector whose width
differs from the column.

## No post-hoc cast needed

Embeddings are written as `%s::vector` through psycopg2, so they land in a real
`vector` column on insert. There's no array-to-vector `UPDATE` step to remember
after each run.
