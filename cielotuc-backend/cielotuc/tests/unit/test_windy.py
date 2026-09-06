"""
tests/unit/test_windy.py
──────────────────────────
Unit tests for the Windy Point Forecast API client (response parsing,
feature mapping and graceful degradation without an API key).
"""

import pandas as pd
import pytest

from app.services.windy_client import WindyClient, PARAMS


@pytest.fixture
def client() -> WindyClient:
    return WindyClient(api_key="test-key", model="ecmwf")


def sample_payload() -> dict:
    """Simulated Windy Point Forecast response for 2 timesteps at 3 levels."""
    ts = ["2026-01-01T00:00:00Z", "2026-01-01T03:00:00Z"]
    headers = {
        "wind": [["surface", "850h", "700h"]],
        "temp": [["surface", "850h", "700h"]],
        "rh": [["surface"]],
        "dewpoint": [["surface"]],
        "prmsl": [["surface"]],
        "cape": [["surface"]],
        "precip": [["surface"]],
    }
    # row values follow header param order, with "wind" contributing 2 values.
    # order: wind(surface)=[speed,dir], temp, rh, dewpoint, prmsl, cape, precip
    surface_t0 = [5.0, 90.0, 25.0, 60.0, 15.0, 1013.0, 1500.0, 0.5]
    surface_t1 = [5.0, 90.0, 25.0, 60.0, 15.0, 1013.0, 1500.0, 1.1]
    lvl850 = [12.0, 260.0, 10.0]      # wind(850h), temp(850h)
    lvl700 = [15.0, 270.0, 5.0]       # wind(700h), temp(700h)

    rows = [
        {"time_idx": 0, "level_idx": 0, "lat": -26.82, "lon": -65.22, "values": surface_t0},
        {"time_idx": 0, "level_idx": 1, "lat": -26.82, "lon": -65.22, "values": lvl850},
        {"time_idx": 0, "level_idx": 2, "lat": -26.82, "lon": -65.22, "values": lvl700},
        {"time_idx": 1, "level_idx": 0, "lat": -26.82, "lon": -65.22, "values": surface_t1},
        {"time_idx": 1, "level_idx": 1, "lat": -26.82, "lon": -65.22, "values": lvl850},
        {"time_idx": 1, "level_idx": 2, "lat": -26.82, "lon": -65.22, "values": lvl700},
    ]
    return {"ts": ts, "headers": headers, "rows": rows}


# ── Configuration ───────────────────────────────────────────────
def test_params_include_wind_and_multilevel():
    assert "wind" in PARAMS
    assert "850h" in PARAMS["wind"]
    assert "700h" in PARAMS["wind"]
    assert "surface" in PARAMS["temp"]


def test_n_values_wind_returns_two():
    from app.services.windy_client import _n_values
    assert _n_values("wind") == 2
    assert _n_values("temp") == 1


def test_not_configured_without_key():
    c = WindyClient(api_key="")
    assert c.is_configured is False


# ── Parsing ─────────────────────────────────────────────────────
def test_parse_response_basic_columns(client):
    df = client._parse_response(sample_payload(), model="ecmwf")
    assert len(df) == 2  # one row per timestamp

    for col in [
        "timestamp", "temperature_c", "humidity_pct", "dew_point_c",
        "pressure_hpa", "wind_speed_kmh", "wind_direction_deg",
        "cape_j_kg", "precip_mm",
        "wind_speed_850h_kmh", "wind_dir_850h_deg",
        "wind_speed_700h_kmh", "wind_dir_700h_deg",
        "temp_850h_c", "temp_700h_c",
    ]:
        assert col in df.columns, f"missing column: {col}"


def test_parse_response_unit_conversions(client):
    df = client._parse_response(sample_payload(), model="ecmwf")
    # wind 5.0 m/s → 18.0 km/h
    assert (df["wind_speed_kmh"] - 18.0).abs().max() < 1e-6
    # surface temperature kept in °C
    assert (df["temperature_c"] - 25.0).abs().max() < 1e-6
    # cape passed through
    assert (df["cape_j_kg"] - 1500.0).abs().max() < 1e-6
    # precip cumulative 0.5 → 1.1 → per-step (0.6 / timeStep 3) = 0.2 mm/h
    assert df["precip_mm"].iloc[0] == pytest.approx(0.0)
    assert df["precip_mm"].iloc[1] == pytest.approx(0.6 / 3, rel=1e-6)


def test_parse_response_pressure_gradient(client):
    df = client._parse_response(sample_payload(), model="ecmwf")
    assert "cordillera_pressure_hpa" in df.columns
    assert "andes_plain_pressure_diff" in df.columns
    assert df["andes_plain_pressure_diff"].notna().all()


def test_parse_empty_payload(client):
    assert client._parse_response({}, model="ecmwf").empty


def test_parse_payload_without_rows(client):
    assert client._parse_response({"ts": ["2026-01-01T00:00:00Z"]}, model="ecmwf").empty


# ── Merge ───────────────────────────────────────────────────────
def test_merge_into_fills_gaps():
    base = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]),
        "temperature_c": [25.0, None],
        "wind_speed_kmh": [None, 10.0],
    })
    windy = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]),
        "temperature_c": [24.0, 26.0],
        "wind_speed_kmh": [20.0, 30.0],
    })
    merged = WindyClient.merge_into(base, windy)
    assert merged["temperature_c"].iloc[0] == 25.0  # base wins
    assert merged["temperature_c"].iloc[1] == 26.0  # gap filled
    assert merged["wind_speed_kmh"].iloc[0] == 20.0  # filled
    assert merged["wind_speed_kmh"].iloc[1] == 10.0  # base wins
