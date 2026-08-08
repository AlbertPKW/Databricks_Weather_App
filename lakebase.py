"""
Lakebase (Databricks-managed Postgres) connection helper for the Weather
Intelligence app, plus the DDL/migration for the two destination tables.

Connection strategy (first match wins):
  1. LAKEBASE_URL environment variable - convenient for local development
     (see .env.example).
  2. A Databricks secret (scope `database`, key `lakebase-url`) holding the
     same standard Postgres URL, e.g.
     postgresql://role:password@host:5432/databricks_postgres?sslmode=require

The role behind that URL is a native Postgres role with a static,
non-expiring password, so there is no token-refresh logic here.
"""

import base64
import os
import re
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

# --- Schema configuration -------------------------------------------------
# The embedding model and the pgvector column width must agree. Both the
# ingestion job and the search endpoint read these same two values, so a model
# swap is a one-place change (plus a migration).
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")

# Table names are interpolated into SQL strings, so they are validated as plain
# identifiers rather than trusted blindly from the environment.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_workspace_client = None


def _validate_identifier(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


_validate_identifier(DOCUMENTS_TABLE)
_validate_identifier(EMBEDDINGS_TABLE)


def _get_workspace_client():
    """Import and construct the Databricks SDK client lazily.

    Keeps `import lakebase` working on a plain laptop that only has
    LAKEBASE_URL set and no Databricks auth configured.
    """
    global _workspace_client
    if _workspace_client is None:
        from databricks.sdk import WorkspaceClient

        _workspace_client = WorkspaceClient()
    return _workspace_client


def _lakebase_url() -> str:
    """Resolve the Lakebase connection URL from env, else the secret scope."""
    env_url = os.environ.get("LAKEBASE_URL")
    if env_url:
        return env_url

    secret = _get_workspace_client().secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase, return affected rows."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


# --------------------------------------------------------------------------
# DDL / migration
# --------------------------------------------------------------------------

DOCUMENTS_DDL = f"""
CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
    id             TEXT PRIMARY KEY,
    location       TEXT NOT NULL,
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION,
    source_type    TEXT NOT NULL,
    event          TEXT,
    headline       TEXT,
    narrative_text TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    severity       TEXT,
    area_desc      TEXT,
    issued_at      TIMESTAMPTZ,
    effective_at   TIMESTAMPTZ,
    expires_at     TIMESTAMPTZ,
    payload        JSONB NOT NULL,
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

EMBEDDINGS_DDL = f"""
CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL REFERENCES {DOCUMENTS_TABLE} (id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    chunk_text   TEXT NOT NULL,
    embedding    VECTOR({EMBEDDING_DIM}) NOT NULL,
    model_name   TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index, model_name)
)
"""

_MIGRATION_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    DOCUMENTS_DDL,
    f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_location "
    f"ON {DOCUMENTS_TABLE} (location)",
    f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_source_type "
    f"ON {DOCUMENTS_TABLE} (source_type)",
    f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_issued_at "
    f"ON {DOCUMENTS_TABLE} (issued_at DESC)",
    EMBEDDINGS_DDL,
    f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_document_id "
    f"ON {EMBEDDINGS_TABLE} (document_id)",
    # HNSW + cosine ops, matching the `<=>` operator used by the search query.
    f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_embedding "
    f"ON {EMBEDDINGS_TABLE} USING hnsw (embedding vector_cosine_ops)",
]


def ensure_weather_tables() -> None:
    """Create the pgvector extension, both tables, and their indexes.

    Idempotent - every statement is IF NOT EXISTS - so it is safe to call on
    every app start and at the top of the ingestion job. The equivalent
    statements are also in sql/ for anyone who prefers to run them by hand.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            for statement in _MIGRATION_STATEMENTS:
                cur.execute(statement)
        conn.commit()


if __name__ == "__main__":
    ensure_weather_tables()
    print(f"Ready: {DOCUMENTS_TABLE}, {EMBEDDINGS_TABLE} "
          f"(vector({EMBEDDING_DIM}), hnsw/cosine index)")
