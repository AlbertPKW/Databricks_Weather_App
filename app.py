"""
Weather Intelligence - Databricks App.

- Harvests unstructured weather narratives from the National Weather Service
  API and upserts them into Lakebase (Databricks-managed Postgres).
- Embeds those narratives into pgvector columns (see weather_embeddings.py).
- Serves semantic search over them with pgvector's `<=>` cosine operator.

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import json
import logging
import os

import requests
from flask import Flask, jsonify, render_template, request

import lakebase
import weather_embeddings
from weather_client import (
    SOURCE_ALERT,
    SOURCE_FORECAST,
    SOURCE_HOURLY,
    LocationNotFound,
    WeatherClient,
    content_hash,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

DOCUMENTS_TABLE = lakebase.DOCUMENTS_TABLE
EMBEDDINGS_TABLE = lakebase.EMBEDDINGS_TABLE

DEFAULT_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get(
        "WEATHER_LOCATIONS", "Chicago, IL;Austin, TX;Miami, FL"
    ).split(";")
    if loc.strip()
]

VALID_SOURCE_TYPES = {SOURCE_ALERT, SOURCE_FORECAST, SOURCE_HOURLY}

MAX_TOP_K = 20
MIN_TOP_K = 1

# Optional: a Databricks model-serving endpoint used to summarize search hits.
# Unset means /weather/search just returns the ranked passages.
SUMMARY_ENDPOINT = os.environ.get("WEATHER_SUMMARY_ENDPOINT", "").strip()

# The embedding model is loaded at module scope rather than per request. Set
# PRELOAD_EMBEDDING_MODEL=1 to pay that cost at boot instead of on the first
# search (worth it for a deployed app; skip it for a fast local restart loop).
if os.environ.get("PRELOAD_EMBEDDING_MODEL", "").lower() in ("1", "true", "yes"):
    weather_embeddings.get_embedding_model()


# --------------------------------------------------------------------------
# Health, errors, UI
# --------------------------------------------------------------------------

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Return JSON for every unhandled error so the UI's resp.json() holds up."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Sync console: pull weather documents for a list of locations."""
    return render_template("index.html", default_locations="; ".join(DEFAULT_LOCATIONS))


@app.route("/search")
def search_ui():
    """Semantic search over the ingested weather narratives."""
    return render_template("search.html")


# --------------------------------------------------------------------------
# Part 1 - Harvest
# --------------------------------------------------------------------------

@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """Fetch active alerts and forecasts for a set of locations and upsert them.

    Body (optional JSON):
        {"locations": ["Chicago, IL", "Austin, TX"],
         "limit": 50,
         "source_types": ["alert", "forecast"]}

    Locations may be "City, ST" or "lat,lon". A location that fails to resolve
    or fetch is reported in `errors` rather than failing the whole request -
    one bad entry shouldn't discard the documents already collected.
    """
    lakebase.ensure_weather_tables()

    body = request.json if request.is_json else {}

    # An absent `locations` key falls back to the configured defaults; an
    # explicitly empty one is a mistake worth reporting, not a silent default.
    requested = body.get("locations")
    if requested is None:
        requested = DEFAULT_LOCATIONS
    locations = [loc.strip() for loc in requested if isinstance(loc, str) and loc.strip()]
    if not locations:
        return jsonify({"error": "Provide at least one location"}), 400

    limit = max(1, min(int(body.get("limit", 50)), 500))

    requested_types = body.get("source_types") or [SOURCE_ALERT, SOURCE_FORECAST]
    source_types = [t for t in requested_types if t in VALID_SOURCE_TYPES]
    if not source_types:
        return jsonify(
            {"error": f"source_types must be any of {sorted(VALID_SOURCE_TYPES)}"}
        ), 400

    client = WeatherClient()
    synced = 0
    per_location = {}
    errors = {}

    for location in locations:
        try:
            documents = client.fetch_documents(
                location, limit=limit, source_types=source_types
            )
        except LocationNotFound as exc:
            errors[location] = str(exc)
            continue
        except requests.HTTPError as exc:
            errors[location] = f"NWS API error: {exc}"
            continue
        except requests.RequestException as exc:
            errors[location] = f"Could not reach the NWS API: {exc}"
            continue

        written = _upsert_documents(documents)
        per_location[location] = written
        synced += written

    return jsonify(
        {
            "synced": synced,
            "locations": per_location,
            "source_types": source_types,
            "errors": errors,
        }
    )


def _upsert_documents(documents: list[dict]) -> int:
    """Upsert normalized weather documents, one statement per row.

    Re-running a sync updates in place rather than duplicating: alert ids come
    from NWS, and forecast ids are hashed from the grid cell plus the period
    start time, so both are stable across runs.
    """
    if not documents:
        return 0

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {DOCUMENTS_TABLE} (
                        id, location, latitude, longitude, source_type, event,
                        headline, narrative_text, content_hash, severity,
                        area_desc, issued_at, effective_at, expires_at,
                        payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET location       = EXCLUDED.location,
                            latitude       = EXCLUDED.latitude,
                            longitude      = EXCLUDED.longitude,
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
                    """,
                    (
                        doc["id"],
                        doc["location"],
                        doc.get("latitude"),
                        doc.get("longitude"),
                        doc["source_type"],
                        doc.get("event"),
                        doc.get("headline"),
                        doc["narrative_text"],
                        content_hash(doc["narrative_text"]),
                        doc.get("severity"),
                        doc.get("area_desc"),
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        doc.get("expires_at"),
                        json.dumps(doc.get("payload", {})),
                    ),
                )
                count += 1
        conn.commit()
    return count


@app.route("/weather/documents")
def list_documents():
    """Read weather documents already synced into Lakebase."""
    limit = max(1, min(int(request.args.get("limit", 50)), 500))
    source_type = request.args.get("source_type")

    sql = f"""
        SELECT id, location, source_type, event, headline, narrative_text,
               severity, issued_at, effective_at, expires_at, synced_at
        FROM {DOCUMENTS_TABLE}
    """
    params: list = []
    if source_type:
        if source_type not in VALID_SOURCE_TYPES:
            return jsonify({"error": f"Unknown source_type: {source_type}"}), 400
        sql += " WHERE source_type = %s"
        params.append(source_type)
    sql += " ORDER BY synced_at DESC LIMIT %s"
    params.append(limit)

    return jsonify(lakebase.run_query(sql, tuple(params)))


# --------------------------------------------------------------------------
# Part 2 - Vectorize (convenience trigger; the scheduled job is the main path)
# --------------------------------------------------------------------------

@app.route("/weather/embed", methods=["POST"])
def embed_documents():
    """Embed any documents that don't have current vectors yet.

    Handy for a laptop demo. For real volume, run the notebook or
    scripts/ingest_weather_embeddings.py on a cluster instead - embedding
    thousands of chunks inside a request thread will time out.
    """
    body = request.json if request.is_json else {}
    limit = body.get("limit")
    result = weather_embeddings.ingest_pending(limit=int(limit) if limit else None)
    return jsonify(result)


# --------------------------------------------------------------------------
# Part 3 - Retrieve
# --------------------------------------------------------------------------

@app.route("/weather/search", methods=["POST", "GET"])
def weather_search():
    """Semantic search over embedded weather narratives.

    POST body: {"query": "risk of flooding near rivers", "top_k": 5}
    GET:       /weather/search?query=...&top_k=5&summarize=true

    Optional filters: `location`, `source_type`.
    """
    if request.method == "POST":
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        body = request.json or {}
        summarize = bool(body.get("summarize", False))
    else:
        body = request.args
        summarize = str(request.args.get("summarize", "")).lower() in ("1", "true", "yes")

    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Provide a query to search for"}), 400

    try:
        top_k = int(body.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be a whole number"}), 400
    top_k = max(MIN_TOP_K, min(top_k, MAX_TOP_K))

    location = (body.get("location") or "").strip() or None
    source_type = (body.get("source_type") or "").strip() or None
    if source_type and source_type not in VALID_SOURCE_TYPES:
        return jsonify({"error": f"Unknown source_type: {source_type}"}), 400

    query_vector = weather_embeddings.embed_query(query)

    # `<=>` is pgvector's cosine distance, so similarity is 1 - distance.
    # The same expression is used in ORDER BY so the HNSW index is usable.
    sql = f"""
        SELECT d.id,
               d.location,
               d.source_type,
               d.event,
               d.headline,
               d.narrative_text,
               d.severity,
               d.effective_at,
               d.expires_at,
               e.chunk_index,
               e.chunk_text,
               e.model_name,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
    """
    params: list = [query_vector]

    filters = []
    if location:
        filters.append("d.location ILIKE %s")
        params.append(f"%{location}%")
    if source_type:
        filters.append("d.source_type = %s")
        params.append(source_type)
    if filters:
        sql += " WHERE " + " AND ".join(filters)

    sql += " ORDER BY e.embedding <=> %s::vector LIMIT %s"
    params.extend([query_vector, top_k])

    rows = lakebase.run_query(sql, tuple(params))

    results = [
        {
            "document_id": row["id"],
            "location": row["location"],
            "source_type": row["source_type"],
            "event": row["event"],
            "headline": row["headline"],
            "chunk_index": row["chunk_index"],
            "chunk_text": row["chunk_text"],
            "narrative_text": row["narrative_text"],
            "severity": row["severity"],
            "effective_at": row["effective_at"],
            "expires_at": row["expires_at"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]

    payload = {"query": query, "top_k": top_k, "count": len(results), "results": results}

    if not results:
        # An empty corpus and a query with no near neighbours look identical
        # from the client side, so say which one happened.
        total = lakebase.run_query(f"SELECT count(*) AS n FROM {EMBEDDINGS_TABLE}")
        payload["message"] = (
            "No weather documents have been embedded yet. Run POST /weather/sync, "
            "then the embedding job."
            if total and total[0]["n"] == 0
            else "No matches for that query. Try broader wording or drop the filters."
        )

    if summarize and results:
        payload["summary"] = _summarize(query, results)

    return jsonify(payload)


def _summarize(query: str, results: list[dict]) -> str | None:
    """Ask a Databricks serving endpoint to summarize the retrieved passages.

    Optional by design: with no endpoint configured, search still works and
    this returns None rather than erroring.
    """
    if not SUMMARY_ENDPOINT:
        return None

    context = "\n\n".join(
        f"[{r['location']} | {r['source_type']}] {r['headline']}\n{r['chunk_text']}"
        for r in results
    )
    prompt = (
        "Answer the question using only the weather bulletins below. "
        "Name the locations you drew from, and say so plainly if the bulletins "
        "don't cover the question.\n\n"
        f"Question: {query}\n\nBulletins:\n{context}"
    )

    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        client = WorkspaceClient()
        response = client.serving_endpoints.query(
            name=SUMMARY_ENDPOINT,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception:
        logger.exception("Summary endpoint call failed; returning results only")
        return None


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", 8000))
    app.run(debug=True, host=host, port=port)
