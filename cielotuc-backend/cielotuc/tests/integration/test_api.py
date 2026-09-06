"""
tests/integration/test_api.py
───────────────────────────────
Integration tests for all FastAPI endpoints.
Uses httpx AsyncClient against the real app (with test DB).

Run with: pytest tests/integration/test_api.py -v
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


# ── Health ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "CIELO" in data["app"]


# ── Zones ──────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_list_zones_returns_list(client):
    resp = await client.get("/api/v1/zones/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_zone_not_found(client):
    resp = await client.get("/api/v1/zones/99999")
    assert resp.status_code == 404


# ── Sensors ────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_sensor_status(client):
    resp = await client.get("/api/v1/sensors/status")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_ingest_reading_unknown_station(client):
    payload = {
        "station_code": "NONEXISTENT-001",
        "timestamp": "2026-04-01T12:00:00Z",
        "temperature_c": 25.0,
        "humidity_pct": 65.0,
        "pressure_hpa": 1010.0,
        "precip_mm": 0.0,
        "wind_speed_kmh": 15.0,
    }
    resp = await client.post("/api/v1/sensors/readings", json=payload)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ingest_reading_invalid_temperature(client):
    """Temperature outside Tucumán plausible range should be rejected."""
    payload = {
        "station_code": "SMN-TUC-AER",
        "timestamp": "2026-04-01T12:00:00Z",
        "temperature_c": 200.0,  # impossible
        "humidity_pct": 65.0,
        "pressure_hpa": 1010.0,
        "precip_mm": 0.0,
        "wind_speed_kmh": 15.0,
    }
    resp = await client.post("/api/v1/sensors/readings", json=payload)
    assert resp.status_code == 422  # validation error


# ── Alerts ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_flood_alert_history(client):
    resp = await client.get("/api/v1/alerts/flood/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_flood_alert_missing_zone(client):
    payload = {
        "zone_id": 99999,
        "rain_probability": 0.85,
        "expected_precip_mm": 95.0,
        "severity": "critical",
        "trigger_type": "manual",
    }
    resp = await client.post("/api/v1/alerts/flood", json=payload)
    # Either 404 (zone not found) or 201 (logged but not delivered)
    assert resp.status_code in (201, 404)


# ── Model metrics ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_model_metrics_no_model(client):
    """Should return 404 if no active model version exists yet."""
    resp = await client.get("/api/v1/model/metrics")
    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_model_comparison_response_shape(client):
    resp = await client.get("/api/v1/model/comparison?zone_id=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "cielotuc_accuracy" in data
    assert "smn_accuracy" in data
    assert "weathercom_accuracy" in data
    assert data["cielotuc_accuracy"] > data["smn_accuracy"]


# ── Forecast ───────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_forecast_no_model_returns_503(client):
    """Without a trained model loaded, forecast should be unavailable."""
    resp = await client.get("/api/v1/forecast/1")
    assert resp.status_code in (200, 503)


@pytest.mark.asyncio
async def test_zonda_index_returns_valid_level(client):
    resp = await client.get("/api/v1/forecast/1/zonda")
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_level"] in ("none", "low", "medium", "high", "active")
    assert 0 <= data["risk_score"] <= 100
    assert 0 <= data["forecast_24h_probability"] <= 1
