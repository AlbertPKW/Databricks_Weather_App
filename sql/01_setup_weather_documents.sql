-- Raw weather documents harvested from the National Weather Service API.
-- Run once against your Lakebase Postgres database before the first sync.
-- (lakebase.ensure_weather_tables() runs the same statements from Python.)

CREATE TABLE IF NOT EXISTS weather_documents (
    id             TEXT PRIMARY KEY,        -- NWS alert id, or hash(grid + period start)
    location       TEXT NOT NULL,           -- resolved place name, e.g. "Chicago, IL"
    latitude       DOUBLE PRECISION,
    longitude      DOUBLE PRECISION,
    source_type    TEXT NOT NULL,           -- 'alert' | 'forecast' | 'forecast_hourly'
    event          TEXT,                    -- e.g. "Flash Flood Warning"
    headline       TEXT,
    narrative_text TEXT NOT NULL,           -- the free text that gets embedded
    content_hash   TEXT NOT NULL,           -- sha256(narrative_text); detects stale vectors
    severity       TEXT,
    area_desc      TEXT,
    issued_at      TIMESTAMPTZ,
    effective_at   TIMESTAMPTZ,
    expires_at     TIMESTAMPTZ,
    payload        JSONB NOT NULL,          -- raw API response, for provenance
    synced_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

CREATE INDEX IF NOT EXISTS idx_weather_documents_issued_at
    ON weather_documents (issued_at DESC);

-- Verify
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
