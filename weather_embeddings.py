"""
Chunking, embedding, and pgvector writes for the weather pipeline.

Everything that turns `weather_documents.narrative_text` into rows in
`weather_embeddings` lives here, so the Flask app, the standalone script, and
the scheduled notebook all chunk and embed identically. If they diverged, the
query vector and the stored vectors would stop being comparable.

Writes go through psycopg2 `execute_values` with an explicit `%s::vector`
cast - Spark JDBC writes are not supported against this Lakebase instance.
"""

import logging
import os
import threading

from psycopg2.extras import execute_values

import lakebase
from weather_client import content_hash

logger = logging.getLogger(__name__)

# NWS narratives are short - a forecast period is usually 1-3 sentences, and
# most alerts fit in a single chunk. The window only bites on long alert bodies
# where description + instruction run to several paragraphs. 800/100 matches the
# reference pipeline so both corpora chunk the same way.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

EMBEDDING_MODEL = lakebase.EMBEDDING_MODEL
EMBEDDING_DIM = lakebase.EMBEDDING_DIM
DOCUMENTS_TABLE = lakebase.DOCUMENTS_TABLE
EMBEDDINGS_TABLE = lakebase.EMBEDDINGS_TABLE

_model = None
_model_lock = threading.Lock()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def get_embedding_model():
    """Load the sentence-transformers model once per process.

    Guarded by a lock because Flask serves requests on multiple threads and two
    concurrent first-requests would otherwise both pay the load cost.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s", EMBEDDING_MODEL)
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a list of strings, in batches, into plain Python float lists."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vectors.extend(model.encode(batch, show_progress_bar=False).tolist())
    return vectors


def to_pgvector_literal(vector) -> str:
    """Format a vector the way pgvector's text input expects: [1.0,2.0,...]."""
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def embed_query(query: str) -> str:
    """Embed a search string and return it ready to bind as `%s::vector`."""
    model = get_embedding_model()
    return to_pgvector_literal(model.encode(query, show_progress_bar=False))


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping windows of `chunk_size` characters.

    The overlap keeps a sentence that straddles a boundary retrievable from
    both sides. Text shorter than one window comes back as a single chunk.
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
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


# --------------------------------------------------------------------------
# Read pending work
# --------------------------------------------------------------------------

def fetch_pending_documents(conn, limit: int | None = None) -> list[dict]:
    """Documents with no current vectors for the active model.

    "Current" means same model *and* same content hash, so an alert that gets
    reissued with updated text is picked up again on the next run instead of
    keeping its stale vectors forever.
    """
    sql = f"""
        SELECT d.id, d.location, d.source_type, d.headline,
               d.narrative_text, d.content_hash
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
    params: list = [EMBEDDING_MODEL]
    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return [dict(row) for row in cur.fetchall()]


# --------------------------------------------------------------------------
# Write vectors
# --------------------------------------------------------------------------

def write_embeddings(conn, rows: list[dict], page_size: int = 100) -> int:
    """Upsert embedding rows, casting each vector with `%s::vector`.

    Stale vectors for a re-embedded document are cleared first, so a document
    whose text got shorter doesn't leave orphaned high-index chunks behind.
    """
    if not rows:
        return 0

    document_ids = sorted({row["document_id"] for row in rows})
    values = [
        (
            row["id"],
            row["document_id"],
            int(row["chunk_index"]),
            row["chunk_text"],
            to_pgvector_literal(row["embedding"]),
            EMBEDDING_MODEL,
            row["content_hash"],
        )
        for row in rows
    ]

    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {EMBEDDINGS_TABLE} "
            f"WHERE document_id = ANY(%s) AND model_name = %s",
            (document_ids, EMBEDDING_MODEL),
        )
        execute_values(
            cur,
            f"""
            INSERT INTO {EMBEDDINGS_TABLE} (
                id, document_id, chunk_index, chunk_text,
                embedding, model_name, content_hash
            ) VALUES %s
            ON CONFLICT (id) DO UPDATE
                SET chunk_text   = EXCLUDED.chunk_text,
                    embedding    = EXCLUDED.embedding,
                    model_name   = EXCLUDED.model_name,
                    content_hash = EXCLUDED.content_hash,
                    created_at   = now()
            """,
            values,
            template="(%s, %s, %s, %s, %s::vector, %s, %s)",
            page_size=page_size,
        )
    conn.commit()
    return len(values)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def build_chunk_rows(documents: list[dict]) -> list[dict]:
    """Chunk documents and embed every chunk in one batched pass."""
    pending = []
    for doc in documents:
        for index, piece in enumerate(chunk_text(doc["narrative_text"])):
            pending.append(
                {
                    "id": f"{doc['id']}::{index}",
                    "document_id": doc["id"],
                    "chunk_index": index,
                    "chunk_text": piece,
                    # Recompute rather than trust the stored hash, so a row
                    # written by an older ingestion still lines up.
                    "content_hash": doc.get("content_hash")
                    or content_hash(doc["narrative_text"]),
                }
            )

    if not pending:
        return []

    vectors = embed_texts([row["chunk_text"] for row in pending])
    for row, vector in zip(pending, vectors):
        row["embedding"] = vector
    return pending


def ingest_pending(limit: int | None = None, batch_documents: int = 200) -> dict:
    """Embed every document that doesn't have current vectors yet.

    Returns counts rather than printing, so the Flask endpoint, the CLI, and
    the notebook can each report in their own way.
    """
    with lakebase.get_connection() as conn:
        documents = fetch_pending_documents(conn, limit=limit)
        logger.info("Found %d documents needing embeddings", len(documents))

        chunks_written = 0
        for start in range(0, len(documents), batch_documents):
            batch = documents[start:start + batch_documents]
            rows = build_chunk_rows(batch)
            chunks_written += write_embeddings(conn, rows)
            logger.info(
                "Embedded %d/%d documents", min(start + batch_documents, len(documents)),
                len(documents),
            )

    return {
        "documents_embedded": len(documents),
        "chunks_written": chunks_written,
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIM,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = ingest_pending()
    print(
        f"Embedded {result['documents_embedded']} documents into "
        f"{result['chunks_written']} chunks using {result['model']} "
        f"({result['dimensions']}-dim)"
    )
