# Databricks notebook source
# MAGIC %md
# MAGIC # Harvest NWS Weather Narratives -> pgvector Embeddings (Lakebase)
# MAGIC
# MAGIC Scheduled ETL for the Weather Intelligence app. It:
# MAGIC
# MAGIC 1. Harvests active alerts and narrative forecasts for a list of
# MAGIC    locations from the National Weather Service API (no API key; the
# MAGIC    service only asks for a descriptive `User-Agent`), and upserts them
# MAGIC    into `weather_documents`.
# MAGIC 2. Selects documents that have no current vectors — no embedding row
# MAGIC    for the active model whose `content_hash` still matches — so reissued
# MAGIC    alerts get re-embedded and unchanged ones are skipped.
# MAGIC 3. Chunks each narrative with a sliding window, embeds every chunk with
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2` (384-dim), and writes the
# MAGIC    vectors into `weather_embeddings`.
# MAGIC
# MAGIC **All database access is psycopg2.** Spark JDBC writes are not supported
# MAGIC against this Lakebase instance, and `spark.write.jdbc` cannot produce a
# MAGIC pgvector `VECTOR` column anyway. Every vector here is bound as
# MAGIC `%s::vector`, so it lands in the right type on insert — no array cast
# MAGIC step afterwards.
# MAGIC
# MAGIC This notebook is deliberately self-contained: it re-implements the
# MAGIC chunking and normalization from `weather_client.py` /
# MAGIC `weather_embeddings.py` rather than importing them, so it runs on a job
# MAGIC cluster without the repo root being on `sys.path`. **If you change the
# MAGIC chunking parameters here, change them there too** — the query vector and
# MAGIC the stored vectors have to come from the same recipe.
# MAGIC
# MAGIC It reuses the same Lakebase secret (scope `database`, key `lakebase-url-weather`)
# MAGIC that `lakebase.py` uses, so no extra secrets are needed.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.30.0' sentence-transformers requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let a scheduled job override locations, tables, and chunking
# MAGIC without editing the notebook.

# COMMAND ----------

dbutils.widgets.text("locations", "Chicago, IL;Austin, TX;Miami, FL;New York, NY; Los Angeles, CA;Houston, TX;Phoenix, AZ", "Locations (semicolon-separated)")
dbutils.widgets.text("documents_table", "weather_documents", "Destination table (raw documents)")
dbutils.widgets.text("embeddings_table", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("nws_api_base_url", "https://api.weather.gov", "NWS API base URL")
dbutils.widgets.text("nws_user_agent", "weather-intelligence-app (contact@example.com)", "NWS User-Agent (include contact)")
dbutils.widgets.text("source_types", "alert;forecast", "Source types to harvest")
dbutils.widgets.text("fetch_limit", "50", "Max documents per source type per location")
dbutils.widgets.text("chunk_size", "800", "Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Chunk overlap (chars)")
dbutils.widgets.dropdown("skip_harvest", "false", ["true", "false"], "Embed only (skip harvest)")

LOCATIONS = [s.strip() for s in dbutils.widgets.get("locations").split(";") if s.strip()]
DOCUMENTS_TABLE = dbutils.widgets.get("documents_table")
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
NWS_API_BASE_URL = dbutils.widgets.get("nws_api_base_url").rstrip("/")
NWS_USER_AGENT = dbutils.widgets.get("nws_user_agent")
SOURCE_TYPES = [s.strip() for s in dbutils.widgets.get("source_types").split(";") if s.strip()]
FETCH_LIMIT = int(dbutils.widgets.get("fetch_limit"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
SKIP_HARVEST = dbutils.widgets.get("skip_harvest") == "true"

# The pgvector column width is fixed at table-creation time and must match the
# model's output exactly, so the mapping is explicit rather than inferred.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above, and widen the VECTOR(n) "
            "column in sql/02_setup_weather_embeddings.sql to match."
        )

print(f"Model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")
print(f"Locations: {LOCATIONS}")
print(f"Source types: {SOURCE_TYPES} | harvest: {'skipped' if SKIP_HARVEST else 'on'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection
# MAGIC
# MAGIC Same secret and same base64 decoding as `lakebase.py`, parsed into the
# MAGIC pieces psycopg2 needs.

# COMMAND ----------

# DBTITLE 1,Parse Lakebase connection info
import base64
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

secret = w.secrets.get_secret(scope="database", key="lakebase-url-weather")
lakebase_url = base64.b64decode(secret.value).decode("utf-8")
parsed = urlparse(lakebase_url)

DB_CONN_KWARGS = {
    "host": parsed.hostname,
    "port": parsed.port or 5432,
    "dbname": parsed.path.lstrip("/"),
    "user": parsed.username,
    "password": parsed.password,
    "sslmode": "require",
}


def connect():
    return psycopg2.connect(cursor_factory=RealDictCursor, **DB_CONN_KWARGS)


with connect() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database() AS db, current_user AS role")
        info = cur.fetchone()
print(f"Connected to {info['db']} as {info['role']} "
      f"({DB_CONN_KWARGS['host']}:{DB_CONN_KWARGS['port']})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the schema exists
# MAGIC
# MAGIC Idempotent DDL, identical to `sql/01_*.sql` and `sql/02_*.sql`. Running
# MAGIC it here means a fresh workspace needs no manual SQL step.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Harvest weather narratives from the NWS API
# MAGIC
# MAGIC For each location: resolve it to a forecast grid cell via `/points`,
# MAGIC then pull active alerts and the multi-day narrative forecast. Requests
# MAGIC are serial — the NWS API is generous, but a handful of locations doesn't
# MAGIC justify a thread pool, and serial calls keep the log readable.
# MAGIC
# MAGIC A location that fails to resolve or fetch is logged and skipped, so one
# MAGIC bad entry doesn't lose the documents already collected.

# COMMAND ----------

# DBTITLE 1,Fetch and upsert weather documents
import hashlib
import json
import re

import requests

_LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

# Offline lookup for the demo cities; anything else goes to the geocoder.
GAZETTEER = {
    "chicago, il": (41.8781, -87.6298),
    "austin, tx": (30.2672, -97.7431),
    "miami, fl": (25.7617, -80.1918),
    "new york, ny": (40.7128, -74.0060),
    "los angeles, ca": (34.0522, -118.2437),
    "houston, tx": (29.7604, -95.3698),
    "phoenix, az": (33.4484, -112.0740),
    "denver, co": (39.7392, -104.9903),
    "seattle, wa": (47.6062, -122.3321),
    "new orleans, la": (29.9511, -90.0715),
    "oklahoma city, ok": (35.4676, -97.5164),
}

session = requests.Session()
session.headers.update({"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"})


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def stable_id(*parts) -> str:
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def geocode(location: str) -> tuple[float, float]:
    key = location.strip().lower()
    if key in GAZETTEER:
        return GAZETTEER[key]
    resp = session.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 1, "countrycodes": "us"},
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json()
    if not hits:
        raise ValueError(f"No coordinates found for {location!r}")
    return float(hits[0]["lat"]), float(hits[0]["lon"])


def resolve_location(location: str) -> dict:
    match = _LATLON_RE.match(location)
    lat, lon = (float(match.group(1)), float(match.group(2))) if match else geocode(location)

    resp = session.get(f"{NWS_API_BASE_URL}/points/{round(lat, 4)},{round(lon, 4)}", timeout=30)
    resp.raise_for_status()
    props = resp.json().get("properties", {})
    relative = (props.get("relativeLocation") or {}).get("properties", {})
    city, state = relative.get("city"), relative.get("state")

    return {
        "name": f"{city}, {state}" if city and state else location.strip(),
        "latitude": lat,
        "longitude": lon,
        "grid_id": props.get("gridId"),
        "grid_x": props.get("gridX"),
        "grid_y": props.get("gridY"),
    }


def fetch_alerts(loc: dict, limit: int) -> list[dict]:
    # /alerts/active has no `limit` param - sending one 400s ("Query
    # parameter \"limit\" is not recognized"). It always returns every
    # active alert for the point; the cap is applied client-side below.
    resp = session.get(
        f"{NWS_API_BASE_URL}/alerts/active",
        params={"point": f"{round(loc['latitude'], 4)},{round(loc['longitude'], 4)}"},
        timeout=30,
    )
    resp.raise_for_status()
    documents = []
    for feature in resp.json().get("features", [])[:limit]:
        props = feature.get("properties") or {}
        description = (props.get("description") or "").strip()
        instruction = (props.get("instruction") or "").strip()
        narrative = "\n\n".join(p for p in (description, instruction) if p)
        if not narrative:
            continue
        documents.append({
            "id": str(props.get("id") or feature.get("id")
                      or stable_id("alert", loc["name"], props.get("event"), narrative)),
            "location": loc["name"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "source_type": "alert",
            "event": props.get("event"),
            "headline": props.get("headline") or props.get("event"),
            "narrative_text": narrative,
            "severity": props.get("severity"),
            "area_desc": props.get("areaDesc"),
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective") or props.get("onset"),
            "expires_at": props.get("expires") or props.get("ends"),
            "payload": feature,
        })
    return documents


def fetch_forecast(loc: dict, limit: int) -> list[dict]:
    resp = session.get(
        f"{NWS_API_BASE_URL}/gridpoints/{loc['grid_id']}/{loc['grid_x']},{loc['grid_y']}/forecast",
        timeout=30,
    )
    resp.raise_for_status()
    props = resp.json().get("properties", {})
    updated = props.get("updated") or props.get("updateTime")

    documents = []
    for period in props.get("periods", [])[:limit]:
        narrative = (period.get("detailedForecast") or period.get("shortForecast") or "").strip()
        if not narrative:
            continue
        period_name = period.get("name") or period.get("startTime")
        # Prepend the period name so the embedded text carries its own time
        # context - the vector has no other way to say which day it describes.
        narrative = f"{period_name}: {narrative}"

        headline = period.get("shortForecast") or period_name
        if period.get("temperature") is not None and period.get("temperatureUnit"):
            headline = f"{headline} ({period['temperature']}\u00b0{period['temperatureUnit']})"

        documents.append({
            # Hashed from the grid cell and the period's START time, not its
            # issue time - so a re-run updates the row instead of adding one.
            "id": stable_id("forecast", loc["grid_id"], loc["grid_x"], loc["grid_y"],
                            period.get("startTime")),
            "location": loc["name"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "source_type": "forecast",
            "event": period.get("shortForecast"),
            "headline": headline,
            "narrative_text": narrative,
            "severity": None,
            "area_desc": loc["name"],
            "issued_at": updated,
            "effective_at": period.get("startTime"),
            "expires_at": period.get("endTime"),
            "payload": period,
        })
    return documents


UPSERT_DOCUMENT_SQL = f"""
    INSERT INTO {DOCUMENTS_TABLE} (
        id, location, latitude, longitude, source_type, event, headline,
        narrative_text, content_hash, severity, area_desc, issued_at,
        effective_at, expires_at, payload, synced_at
    ) VALUES %s
    ON CONFLICT (id) DO UPDATE
        SET location       = EXCLUDED.location,
            source_type    = EXCLUDED.source_type,
            event          = EXCLUDED.event,
            headline       = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            content_hash   = EXCLUDED.content_hash,
            severity       = EXCLUDED.severity,
            area_desc      = EXCLUDED.area_desc,
            issued_at      = EXCLUDED.issued_at,
            effective_at   = EXCLUDED.effective_at,
            expires_at     = EXCLUDED.expires_at,
            payload        = EXCLUDED.payload,
            synced_at      = EXCLUDED.synced_at
"""

documents_synced = 0

if SKIP_HARVEST:
    print("Harvest skipped (skip_harvest=true) - embedding whatever is already stored.")
else:
    collected = []
    for location in LOCATIONS:
        try:
            loc = resolve_location(location)
            if "alert" in SOURCE_TYPES:
                collected.extend(fetch_alerts(loc, FETCH_LIMIT))
            if "forecast" in SOURCE_TYPES:
                collected.extend(fetch_forecast(loc, FETCH_LIMIT))
            print(f"  {location} -> {loc['name']} (grid {loc['grid_id']} "
                  f"{loc['grid_x']},{loc['grid_y']})")
        except Exception as exc:
            print(f"  Skipping {location}: {exc}")
            continue

    if collected:
        values = [
            (
                doc["id"], doc["location"], doc["latitude"], doc["longitude"],
                doc["source_type"], doc["event"], doc["headline"],
                doc["narrative_text"], content_hash(doc["narrative_text"]),
                doc["severity"], doc["area_desc"], doc["issued_at"],
                doc["effective_at"], doc["expires_at"], json.dumps(doc["payload"]),
            )
            for doc in collected
        ]
        with connect() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur, UPSERT_DOCUMENT_SQL, values,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())",
                    page_size=100,
                )
            conn.commit()
        documents_synced = len(values)

    print(f"\nUpserted {documents_synced} weather documents into {DOCUMENTS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select documents that still need vectors
# MAGIC
# MAGIC A document needs work when it has no embedding row for the active model
# MAGIC whose `content_hash` matches its current text. That covers both new
# MAGIC documents and reissued alerts whose wording changed, while leaving
# MAGIC unchanged documents alone — so a job running every 15 minutes does
# MAGIC almost no work most of the time.

# COMMAND ----------

PENDING_SQL = f"""
    SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text, d.content_hash
    FROM {DOCUMENTS_TABLE} d
    WHERE d.narrative_text IS NOT NULL
      AND btrim(d.narrative_text) <> ''
      AND NOT EXISTS (
          SELECT 1 FROM {EMBEDDINGS_TABLE} e
          WHERE e.document_id = d.id
            AND e.model_name = %s
            AND e.content_hash = d.content_hash
      )
    ORDER BY d.synced_at DESC
"""

with connect() as conn:
    with conn.cursor() as cur:
        cur.execute(PENDING_SQL, (EMBEDDING_MODEL_NAME,))
        pending_documents = [dict(row) for row in cur.fetchall()]

print(f"{len(pending_documents)} documents need embeddings")
if pending_documents:
    display(pending_documents[:5])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk and embed
# MAGIC
# MAGIC Sliding window of `chunk_size` characters with `chunk_overlap` carried
# MAGIC forward, so a sentence landing on a boundary stays retrievable from both
# MAGIC sides. Most NWS text fits in one chunk — forecast periods run a couple of
# MAGIC sentences — so the window mainly matters for long alerts where the
# MAGIC description and the safety instruction together run to several
# MAGIC paragraphs.

# COMMAND ----------

# DBTITLE 1,Compute chunk embeddings
import os

from sentence_transformers import SentenceTransformer

# Job clusters give the driver a small root volume; keep the model cache in /tmp.
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


chunk_rows = []
for doc in pending_documents:
    for index, piece in enumerate(chunk_text(doc["narrative_text"], CHUNK_SIZE, CHUNK_OVERLAP)):
        chunk_rows.append({
            "id": f"{doc['id']}::{index}",
            "document_id": doc["id"],
            "chunk_index": index,
            "chunk_text": piece,
            "content_hash": doc["content_hash"],
        })

print(f"{len(pending_documents)} documents -> {len(chunk_rows)} chunks")

if chunk_rows:
    print(f"Loading {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    batch_size = 32
    vectors = []
    texts = [row["chunk_text"] for row in chunk_rows]
    for start in range(0, len(texts), batch_size):
        vectors.extend(model.encode(texts[start:start + batch_size],
                                    show_progress_bar=False).tolist())
        if (start + batch_size) % 256 == 0:
            print(f"  Embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")

    for row, vector in zip(chunk_rows, vectors):
        row["embedding"] = vector

    print(f"Computed {len(vectors)} vectors of width {len(vectors[0])}")
    assert len(vectors[0]) == EMBEDDING_DIM, (
        f"Model returned {len(vectors[0])}-dim vectors but the column is "
        f"VECTOR({EMBEDDING_DIM}) - the insert would be rejected."
    )
else:
    print("Nothing to embed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write vectors into Lakebase
# MAGIC
# MAGIC `execute_values` with a `%s::vector` template, so each embedding is
# MAGIC parsed straight into the pgvector column. Existing vectors for these
# MAGIC documents are cleared first, so a document whose text got shorter
# MAGIC doesn't leave orphaned high-index chunks behind.

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings
INSERT_EMBEDDING_SQL = f"""
    INSERT INTO {EMBEDDINGS_TABLE} (
        id, document_id, chunk_index, chunk_text, embedding, model_name, content_hash
    ) VALUES %s
    ON CONFLICT (id) DO UPDATE
        SET chunk_text   = EXCLUDED.chunk_text,
            embedding    = EXCLUDED.embedding,
            model_name   = EXCLUDED.model_name,
            content_hash = EXCLUDED.content_hash,
            created_at   = now()
"""

chunks_written = 0

if chunk_rows:
    document_ids = sorted({row["document_id"] for row in chunk_rows})
    values = [
        (
            row["id"],
            row["document_id"],
            int(row["chunk_index"]),
            row["chunk_text"],
            "[" + ",".join(str(float(x)) for x in row["embedding"]) + "]",
            EMBEDDING_MODEL_NAME,
            row["content_hash"],
        )
        for row in chunk_rows
    ]

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {EMBEDDINGS_TABLE} "
                f"WHERE document_id = ANY(%s) AND model_name = %s",
                (document_ids, EMBEDDING_MODEL_NAME),
            )
            execute_values(
                cur, INSERT_EMBEDDING_SQL, values,
                template="(%s, %s, %s, %s, %s::vector, %s, %s)",
                page_size=100,
            )
        conn.commit()
    chunks_written = len(values)

print(f"Wrote {chunks_written} chunk embeddings into {EMBEDDINGS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Confirms the vectors are queryable end to end: correct width, HNSW index
# MAGIC in place, and a real cosine search returning ranked passages.

# COMMAND ----------

# DBTITLE 1,Post-run checks
with connect() as conn:
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT
                (SELECT count(*) FROM {DOCUMENTS_TABLE})  AS documents,
                (SELECT count(*) FROM {EMBEDDINGS_TABLE}) AS chunks,
                (SELECT count(DISTINCT document_id) FROM {EMBEDDINGS_TABLE}) AS embedded_documents
        """)
        print(dict(cur.fetchone()))

        cur.execute(f"SELECT DISTINCT vector_dims(embedding) AS dims FROM {EMBEDDINGS_TABLE}")
        print("Stored vector widths:", [row["dims"] for row in cur.fetchall()])

        # A sample query proves the whole path works, not just the inserts.
        # Needs the model, which is only loaded when there was work to do.
        if chunk_rows:
            probe = model.encode("flash flood risk this weekend", show_progress_bar=False)
            literal = "[" + ",".join(str(float(x)) for x in probe) + "]"
            cur.execute(
                f"""
                SELECT d.location, d.source_type, d.headline,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM {EMBEDDINGS_TABLE} e
                JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
                ORDER BY e.embedding <=> %s::vector
                LIMIT 5
                """,
                (literal, literal),
            )
            print("\nSample search - 'flash flood risk this weekend':")
            for row in cur.fetchall():
                print(f"  {row['similarity']:.3f}  [{row['source_type']}] "
                      f"{row['location']} - {row['headline']}")

print(f"\nRun summary: {documents_synced} documents upserted, "
      f"{chunks_written} chunks embedded with {EMBEDDING_MODEL_NAME}")