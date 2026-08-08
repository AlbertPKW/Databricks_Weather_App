"""
Client for the National Weather Service API (https://api.weather.gov).

The NWS API needs no API key, so there is no secret plumbing here - the only
requirement is a descriptive User-Agent header with contact details, which the
service uses to reach you if a client misbehaves. Requests without one are
rejected with a 403, so set NWS_USER_AGENT before running anything.

The client's job is to turn three NWS endpoints into one flat document shape
that the rest of the pipeline can treat uniformly:

    GET /points/{lat},{lon}                          -> grid + place name
    GET /alerts/active?point={lat},{lon}             -> active warnings
    GET /gridpoints/{office}/{x},{y}/forecast        -> narrative forecast
"""

import hashlib
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import requests

_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT", "weather-intelligence-app (contact@example.com)"
)
_GEOCODER_URL = os.environ.get(
    "GEOCODER_URL", "https://nominatim.openstreetmap.org/search"
)
_DEFAULT_TIMEOUT = 30

SOURCE_ALERT = "alert"
SOURCE_FORECAST = "forecast"
SOURCE_HOURLY = "forecast_hourly"

# "41.88,-87.63" or "41.88, -87.63"
_LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")

# A small offline gazetteer covers the cities used in demos and tests without
# a network round-trip. Anything else falls through to the geocoder below.
_GAZETTEER: dict[str, tuple[float, float]] = {
    "chicago, il": (41.8781, -87.6298),
    "austin, tx": (30.2672, -97.7431),
    "new york, ny": (40.7128, -74.0060),
    "los angeles, ca": (34.0522, -118.2437),
    "houston, tx": (29.7604, -95.3698),
    "phoenix, az": (33.4484, -112.0740),
    "philadelphia, pa": (39.9526, -75.1652),
    "san antonio, tx": (29.4241, -98.4936),
    "san diego, ca": (32.7157, -117.1611),
    "dallas, tx": (32.7767, -96.7970),
    "san francisco, ca": (37.7749, -122.4194),
    "seattle, wa": (47.6062, -122.3321),
    "denver, co": (39.7392, -104.9903),
    "boston, ma": (42.3601, -71.0589),
    "miami, fl": (25.7617, -80.1918),
    "atlanta, ga": (33.7490, -84.3880),
    "new orleans, la": (29.9511, -90.0715),
    "minneapolis, mn": (44.9778, -93.2650),
    "kansas city, mo": (39.0997, -94.5786),
    "oklahoma city, ok": (35.4676, -97.5164),
    "tampa, fl": (27.9506, -82.4572),
    "portland, or": (45.5152, -122.6784),
    "detroit, mi": (42.3314, -83.0458),
    "st. louis, mo": (38.6270, -90.1994),
    "buffalo, ny": (42.8864, -78.8784),
    "anchorage, ak": (61.2181, -149.9003),
    "honolulu, hi": (21.3069, -157.8583),
}


class LocationNotFound(Exception):
    """Raised when a location string cannot be resolved to coordinates."""


@dataclass
class ResolvedLocation:
    """A location pinned to an NWS forecast grid cell."""

    query: str          # what the caller asked for, e.g. "Chicago, IL"
    name: str           # NWS's own label, e.g. "Chicago, IL"
    latitude: float
    longitude: float
    state: str | None
    grid_id: str        # forecast office, e.g. "LOT"
    grid_x: int
    grid_y: int

    def as_dict(self) -> dict:
        return asdict(self)


def _stable_id(*parts: Any) -> str:
    """Deterministic id for records the API doesn't give one for.

    Forecast periods have no natural key, so the id is hashed from the grid
    cell plus the period's start time - values that stay put across re-syncs.
    Hashing the *issue* time instead would mint a new row on every run.
    """
    raw = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def content_hash(text: str) -> str:
    """Hash of the text that gets embedded, used to detect stale vectors."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class WeatherClient:
    """Thin wrapper around api.weather.gov with location resolution built in."""

    def __init__(
        self,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent or _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )
        self._location_cache: dict[str, ResolvedLocation] = {}

    # -- transport ---------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # -- location resolution ----------------------------------------------

    def _geocode(self, location: str) -> tuple[float, float]:
        """Turn a place name into coordinates: gazetteer first, then geocoder."""
        key = location.strip().lower()
        if key in _GAZETTEER:
            return _GAZETTEER[key]

        resp = self._session.get(
            _GEOCODER_URL,
            params={"q": location, "format": "json", "limit": 1, "countrycodes": "us"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        hits = resp.json()
        if not hits:
            raise LocationNotFound(
                f"No coordinates found for {location!r}. Pass 'lat,lon' instead."
            )
        return float(hits[0]["lat"]), float(hits[0]["lon"])

    def resolve_location(self, location: str) -> ResolvedLocation:
        """Resolve 'City, ST' or 'lat,lon' to an NWS grid point.

        NWS only serves US territory, and only accepts coordinates rounded to
        four decimal places on /points.
        """
        cache_key = location.strip().lower()
        if cache_key in self._location_cache:
            return self._location_cache[cache_key]

        match = _LATLON_RE.match(location)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
        else:
            lat, lon = self._geocode(location)

        point = self.get(f"/points/{round(lat, 4)},{round(lon, 4)}")
        props = point.get("properties", {})
        relative = (props.get("relativeLocation") or {}).get("properties", {})

        city, state = relative.get("city"), relative.get("state")
        name = f"{city}, {state}" if city and state else location.strip()

        resolved = ResolvedLocation(
            query=location.strip(),
            name=name,
            latitude=lat,
            longitude=lon,
            state=state,
            grid_id=props.get("gridId"),
            grid_x=props.get("gridX"),
            grid_y=props.get("gridY"),
        )
        self._location_cache[cache_key] = resolved
        return resolved

    # -- raw endpoint calls ------------------------------------------------

    def get_active_alerts(self, loc: ResolvedLocation, limit: int = 50) -> list[dict]:
        """Active alerts covering this point (watches, warnings, advisories)."""
        data = self.get(
            "/alerts/active",
            params={"point": f"{round(loc.latitude, 4)},{round(loc.longitude, 4)}",
                    "limit": limit},
        )
        return data.get("features", [])[:limit]

    def get_forecast_periods(self, loc: ResolvedLocation, limit: int = 50) -> list[dict]:
        """Multi-day narrative forecast: one period per half-day."""
        data = self.get(f"/gridpoints/{loc.grid_id}/{loc.grid_x},{loc.grid_y}/forecast")
        props = data.get("properties", {})
        periods = props.get("periods", [])[:limit]
        for period in periods:
            period["_updated"] = props.get("updated") or props.get("updateTime")
        return periods

    def get_hourly_forecast_periods(
        self, loc: ResolvedLocation, limit: int = 24
    ) -> list[dict]:
        """Hourly forecast periods. Terse - useful mainly as extra volume."""
        data = self.get(
            f"/gridpoints/{loc.grid_id}/{loc.grid_x},{loc.grid_y}/forecast/hourly"
        )
        props = data.get("properties", {})
        periods = props.get("periods", [])[:limit]
        for period in periods:
            period["_updated"] = props.get("updated") or props.get("updateTime")
        return periods

    # -- normalization -----------------------------------------------------

    @staticmethod
    def normalize_alert(feature: dict, loc: ResolvedLocation) -> dict | None:
        """Flatten one alert GeoJSON feature into a document row.

        description and instruction are joined because they answer different
        questions ("what is happening" vs "what should you do") and a query
        like "what do I do in a flash flood" should be able to hit either.
        """
        props = feature.get("properties") or {}
        description = (props.get("description") or "").strip()
        instruction = (props.get("instruction") or "").strip()

        narrative = "\n\n".join(part for part in (description, instruction) if part)
        if not narrative:
            return None

        alert_id = props.get("id") or feature.get("id")
        if not alert_id:
            alert_id = _stable_id(SOURCE_ALERT, loc.name, props.get("event"),
                                  props.get("sent"), narrative)

        return {
            "id": str(alert_id),
            "location": loc.name,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "source_type": SOURCE_ALERT,
            "event": props.get("event"),
            "headline": props.get("headline") or props.get("event"),
            "narrative_text": narrative,
            "severity": props.get("severity"),
            "area_desc": props.get("areaDesc"),
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective") or props.get("onset"),
            "expires_at": props.get("expires") or props.get("ends"),
            "payload": feature,
        }

    @staticmethod
    def normalize_forecast_period(
        period: dict, loc: ResolvedLocation, source_type: str = SOURCE_FORECAST
    ) -> dict | None:
        """Flatten one forecast period into a document row."""
        narrative = (period.get("detailedForecast") or period.get("shortForecast") or "").strip()
        if not narrative:
            return None

        period_name = period.get("name") or period.get("startTime")
        # The period name ("Tuesday Night") is prepended so the embedded text
        # carries its own time context - the vector has no other way to know
        # which day it describes.
        narrative = f"{period_name}: {narrative}" if period_name else narrative

        doc_id = _stable_id(
            source_type, loc.grid_id, loc.grid_x, loc.grid_y, period.get("startTime")
        )

        temperature = period.get("temperature")
        unit = period.get("temperatureUnit")
        headline = period.get("shortForecast") or period_name
        if temperature is not None and unit:
            headline = f"{headline} ({temperature}\u00b0{unit})"

        return {
            "id": doc_id,
            "location": loc.name,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "source_type": source_type,
            "event": period.get("shortForecast"),
            "headline": headline,
            "narrative_text": narrative,
            "severity": None,
            "area_desc": loc.name,
            "issued_at": period.get("_updated"),
            "effective_at": period.get("startTime"),
            "expires_at": period.get("endTime"),
            "payload": period,
        }

    # -- the one call the app makes ---------------------------------------

    def fetch_documents(
        self,
        location: str,
        limit: int = 50,
        source_types: Iterable[str] = (SOURCE_ALERT, SOURCE_FORECAST),
    ) -> list[dict]:
        """Fetch and normalize every requested document type for one location.

        `limit` caps each source type independently, so a location with both
        alerts and a forecast can return up to 2 * limit documents.
        """
        loc = self.resolve_location(location)
        source_types = set(source_types)
        documents: list[dict] = []

        if SOURCE_ALERT in source_types:
            for feature in self.get_active_alerts(loc, limit=limit):
                doc = self.normalize_alert(feature, loc)
                if doc:
                    documents.append(doc)

        if SOURCE_FORECAST in source_types:
            for period in self.get_forecast_periods(loc, limit=limit):
                doc = self.normalize_forecast_period(period, loc, SOURCE_FORECAST)
                if doc:
                    documents.append(doc)

        if SOURCE_HOURLY in source_types:
            for period in self.get_hourly_forecast_periods(loc, limit=limit):
                doc = self.normalize_forecast_period(period, loc, SOURCE_HOURLY)
                if doc:
                    documents.append(doc)

        return documents
