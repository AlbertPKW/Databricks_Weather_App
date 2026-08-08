#!/usr/bin/env python
"""
Plain-Python version of the embedding pass, for a laptop or a cron box where
spinning up a Databricks job is overkill.

    python scripts/ingest_weather_embeddings.py              # embed pending docs
    python scripts/ingest_weather_embeddings.py --sync       # harvest first
    python scripts/ingest_weather_embeddings.py --sync --locations "Denver, CO"

Same logic as notebooks/ingest_weather_embeddings.py, but it imports the shared
modules instead of re-implementing them, so there is one source of truth for
chunking and embedding whenever this path is available.
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lakebase                       # noqa: E402
import weather_embeddings             # noqa: E402
from weather_client import WeatherClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true",
                        help="harvest fresh documents from the NWS API first")
    parser.add_argument("--locations", default=os.environ.get("WEATHER_LOCATIONS", ""),
                        help='semicolon-separated, e.g. "Chicago, IL;Austin, TX"')
    parser.add_argument("--limit", type=int, default=None,
                        help="cap how many documents to embed in this run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    lakebase.ensure_weather_tables()

    if args.sync:
        # Imported here so the embed-only path doesn't need app.py's Flask deps.
        from app import _upsert_documents

        locations = [s.strip() for s in args.locations.split(";") if s.strip()]
        if not locations:
            parser.error("--sync needs --locations or WEATHER_LOCATIONS")

        client = WeatherClient()
        for location in locations:
            try:
                documents = client.fetch_documents(location)
            except Exception as exc:
                logging.warning("Skipping %s: %s", location, exc)
                continue
            logging.info("%s -> %d documents", location, _upsert_documents(documents))

    result = weather_embeddings.ingest_pending(limit=args.limit)
    print(
        f"Embedded {result['documents_embedded']} documents into "
        f"{result['chunks_written']} chunks using {result['model']} "
        f"({result['dimensions']}-dim)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
