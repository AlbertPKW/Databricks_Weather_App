# Weather Intelligence — unstructured text → Lakebase vector search → REST API

Harvests free-text weather bulletins from the National Weather Service, stores
them in Lakebase (Databricks-managed Postgres), embeds them into `pgvector`
columns, and serves semantic search over them from a Flask API.

```
api.weather.gov ──► POST /weather/sync ──► weather_documents
                                                 │
                              embedding job ─────┤ chunk → embed (384-dim)
                                                 ▼
                                          weather_embeddings  (pgvector)
                                                 │
                    POST /weather/search ◄───────┘  cosine `<=>` + HNSW
```

## Files

| File | What it does |
|---|---|
| `weather_client.py` | NWS API client: resolves locations to forecast grid cells, fetches alerts and forecasts, normalizes both into one document shape |
| `lakebase.py` | Postgres connection helper (`get_connection()` context manager, psycopg2 + `RealDictCursor`) and the idempotent DDL for both tables |
| `weather_embeddings.py` | Chunking, embedding, and `%s::vector` writes — shared by the app and the CLI script |
| `app.py` | Flask API: `/weather/sync`, `/weather/embed`, `/weather/search`, `/weather/documents`, plus two UI pages |
| `notebooks/ingest_weather_embeddings.py` | Self-contained Databricks notebook: harvest + embed, for scheduled runs |
| `scripts/ingest_weather_embeddings.py` | Same job as a plain CLI, for a laptop or cron box |
| `sql/` | The same DDL as `.sql` files, plus verification queries |
| `templates/` | Sync console and search UI |
| `databricks.yml`, `resources/` | Asset Bundle that schedules the notebook as a Workflow |

## Data source: the National Weather Service API

`api.weather.gov` was the right fit for three reasons:

1. **No API key.** Nothing to provision, rotate, or leak — the only credential
   in this project is the database URL. NWS asks instead for a descriptive
   `User-Agent` with contact details; requests without one get a 403, so set
   `NWS_USER_AGENT` before running anything.
2. **The text is genuinely unstructured.** Alerts carry a `description` (what
   is happening) and an `instruction` (what to do about it), both written by a
   forecaster. Forecast periods carry a `detailedForecast` narrative. This is
   prose written for humans, which is exactly what embeddings are for — the
   numeric fields alongside it would be better served by a WHERE clause.
3. **Two document types with different shapes**, which makes retrieval
   interesting: short forward-looking forecasts and long, urgent alerts.

Three endpoints are used:

| Endpoint | Purpose |
|---|---|
| `GET /points/{lat},{lon}` | Resolve coordinates to a forecast office and grid cell |
| `GET /alerts/active?point={lat},{lon}` | Active watches, warnings, and advisories for that point |
| `GET /gridpoints/{office}/{x},{y}/forecast` | Multi-day narrative forecast, one period per half-day |

Locations are accepted as `"City, ST"` or `"lat,lon"`. City names resolve
through a small built-in gazetteer first, falling back to a public geocoder —
NWS itself does no geocoding, and it only covers US territory.

## Schema

### `weather_documents` — raw harvested text

| Column | Notes |
|---|---|
| `id` | NWS alert id, or `sha256(grid + period start)[:40]` for forecasts |
| `location`, `latitude`, `longitude` | Resolved place name plus the coordinates behind it |
| `source_type` | `alert` or `forecast` |
| `event`, `headline` | e.g. `Flash Flood Warning`; the headline is display text |
| `narrative_text` | **The text that gets embedded** |
| `content_hash` | `sha256(narrative_text)` — drives re-embedding |
| `severity`, `area_desc` | NWS severity (`Severe`, `Moderate`, …) and covered counties |
| `issued_at`, `effective_at`, `expires_at` | Alert lifecycle / forecast period bounds |
| `payload` | Raw API JSON, for provenance |
| `synced_at` | Last upsert time |

**Two decisions worth calling out.**

*Forecast ids hash the period's **start** time, not its issue time.* NWS
reissues the same forecast several times a day with a new `updated` stamp; if
that stamp were in the key, every sync would mint duplicate rows for the same
Tuesday afternoon. Hashing the grid cell plus the period start makes re-syncs
idempotent — the row updates in place.

*Alert `description` and `instruction` are concatenated into one
`narrative_text`.* They answer different questions ("what is happening" vs.
"what should you do"), and a query like *"what do I do in a flash flood"*
should be able to reach either half. The forecast period name is prepended to
its narrative for the same reason — a vector has no other way to encode which
day it describes.

### `weather_embeddings` — chunk vectors

| Column | Notes |
|---|---|
| `id` | `"<document_id>::<chunk_index>"` |
| `document_id` | FK to `weather_documents.id`, `ON DELETE CASCADE` |
| `chunk_index`, `chunk_text` | Position and the exact text that produced the vector |
| `embedding` | `VECTOR(384)` |
| `model_name`, `content_hash` | What produced this vector, and from which text |
| `created_at` | |

Indexed with `USING hnsw (embedding vector_cosine_ops)`, matching the `<=>`
cosine operator the search query uses. `chunk_text` is stored alongside the
vector so results can quote the exact passage that matched rather than the
whole document.

### Chunking and the model

`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` characters, sliding window — the same
values as the reference pipeline this project mirrors, so both corpora chunk
identically.

In practice most NWS text fits in a single chunk: a forecast period runs two or
three sentences. The window only bites on long alerts, where the description
and the safety instruction together run to several paragraphs — and those are
exactly the documents where a specific passage ("do not drive through flooded
roadways") should be retrievable without the whole bulletin diluting it. The
100-character overlap keeps a sentence straddling a boundary findable from
both sides.

Embeddings come from `sentence-transformers/all-MiniLM-L6-v2` (384-dim), the
same model as the reference pipeline, so both corpora stay comparable under the
same distance operator. Swapping models means changing `EMBEDDING_MODEL`,
`EMBEDDING_DIM`, the `VECTOR(n)` width in
`sql/02_setup_weather_embeddings.sql`, and re-embedding everything — pgvector
rejects a vector whose width doesn't match its column.

**Vectors are written as real `vector` values on insert**, via
`execute_values` with a `%s::vector` template. There is no array-to-vector
`UPDATE` step to remember after each run. All database access is psycopg2 —
`spark.write.jdbc` is not used anywhere, since it isn't supported against this
Lakebase instance and can't produce a pgvector column regardless.

## Running the pipeline end to end

### 1. Store the Lakebase URL

```python
python setup_secrets.py     # prompts for the URL, writes database/lakebase-url
```

Only one secret is needed — the NWS API is unauthenticated.

### 2. Create the schema

```bash
python lakebase.py
```

Or run `sql/01_setup_weather_documents.sql` and
`sql/02_setup_weather_embeddings.sql` by hand. Both paths produce the same
tables, and everything is `IF NOT EXISTS`, so re-running is safe.

### 3. Configure and install

```bash
cp .env.example .env       # set LAKEBASE_URL and NWS_USER_AGENT
pip install -r requirements.txt
```

### 4. Run it

```bash
python app.py              # http://localhost:8000
```

```bash
# Harvest
curl -X POST localhost:8000/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'
# -> {"synced": 31, "locations": {...}, "errors": {}}

# Embed
curl -X POST localhost:8000/weather/embed -H 'Content-Type: application/json' -d '{}'
# -> {"documents_embedded": 31, "chunks_written": 44, ...}

# Search
curl -X POST localhost:8000/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

Or use the two pages: `/` to sync, `/search` to query.

For real volume, run the embedding pass outside the request cycle:

```bash
python scripts/ingest_weather_embeddings.py --sync --locations "Chicago, IL;Austin, TX"
```

### 5. Schedule it

```bash
databricks bundle deploy -t dev
databricks bundle run ingest_weather_embeddings_job -t dev
```

Once a manual run succeeds, flip `pause_status: PAUSED` to `UNPAUSED` in
`resources/ingest_weather_embeddings_job.yml` and redeploy. The default cadence
is every 30 minutes, because alerts are the time-sensitive half of the corpus —
a flash flood warning retrieved an hour late is worthless. Documents whose text
hasn't changed are skipped, so most runs do almost no work.

## API

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | Health check |
| `POST /weather/sync` | Harvest and upsert. Body: `{"locations": [...], "limit": 50, "source_types": ["alert","forecast"]}` |
| `POST /weather/embed` | Embed documents without current vectors. Body: `{"limit": 500}` (optional) |
| `POST /weather/search` | Semantic search. Body: `{"query": "...", "top_k": 5, "location": "...", "source_type": "...", "summarize": false}` |
| `GET /weather/search?query=…&top_k=5&summarize=true` | Same, as a GET |
| `GET /weather/documents?limit=50&source_type=alert` | Browse stored documents |

Search runs one query:

```sql
SELECT d.id, d.location, d.headline, d.narrative_text, e.chunk_text,
       1 - (e.embedding <=> %s::vector) AS similarity
FROM weather_embeddings e
JOIN weather_documents d ON d.id = e.document_id
ORDER BY e.embedding <=> %s::vector
LIMIT %s;
```

`<=>` is cosine distance, so similarity is `1 - distance`. The same expression
appears in `ORDER BY` so the HNSW index can serve the ranking (confirmed with
`EXPLAIN`: `Index Scan using idx_weather_embeddings_embedding`).

**Edge cases handled:** a missing or blank query returns 400; a non-numeric
`top_k` returns 400; `top_k` is clamped to 1–20; an unknown `source_type`
returns 400; and an empty result set distinguishes *nothing has been embedded
yet* from *nothing matched your query*, because those need different fixes. The
embedding model is loaded once at module scope (set
`PRELOAD_EMBEDDING_MODEL=1` to warm it at boot), never per request.

Setting `WEATHER_SUMMARY_ENDPOINT` to a Databricks model-serving endpoint
enables `summarize: true`, which feeds the retrieved passages to that model for
a short natural-language answer. It fails soft — with no endpoint configured,
search still returns ranked passages.

## Known limitations

- **Nothing expires.** Alerts carry an `expires_at`, but no job deletes stale
  rows, so a warning from last week still competes for retrieval slots. The
  next thing I'd add is a sweep that drops expired alerts, or a recency decay
  applied to the similarity score.
- **Retrieval is pure vector search.** A query naming a city relies on that
  city appearing in the embedded text; the `location` filter exists precisely
  because pure semantics is unreliable for proper nouns. Hybrid search — BM25
  over `narrative_text` fused with the vector ranking — would fix that class of
  miss properly.
- **Geocoding is best-effort.** The built-in gazetteer covers common cities;
  everything else depends on a public geocoder with its own rate limits. A
  production version would use a proper geocoding service, or take coordinates
  only.
- **The notebook duplicates the chunking and normalization logic** from
  `weather_client.py` / `weather_embeddings.py`, so it runs on a job cluster
  without the repo root on `sys.path` — matching how the reference pipeline is
  structured. The duplication is a real hazard: change the chunk parameters in
  one place and search quality quietly degrades. Packaging the shared code as a
  wheel and installing it on the cluster would remove the copy.
- **HNSW index tuning is untouched.** Defaults (`m=16`, `ef_construction=64`)
  are fine at this corpus size; at millions of chunks, both those and
  `hnsw.ef_search` would need benchmarking against recall.
- **No authentication on the write endpoints.** `POST /weather/sync` and
  `/weather/embed` are open to anyone who can reach the app. Databricks Apps
  put SSO in front of the whole thing, but within the app there's no
  distinction between a reader and someone who can trigger an ingest.
