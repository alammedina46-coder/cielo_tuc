"""
app/services/windy_client.py
────────────────────────────
Client for the Windy Point Forecast API (v2).

Why Windy matters for CIELO·TUC:
  - Exposes ECMWF and GFS model output at multiple atmospheric levels
    (surface, 850hPa, 700hPa, 500hPa, 300hPa).
  - Upper-level wind (850/700 hPa) and the Andes-to-plain pressure
    gradient are the key physical signals that anticipate the Zonda.

API docs: https://api.windy.com/point-forecast

Notes
-----
- Requires an API key (settings.windy_api_key). Without it every method
  returns an empty DataFrame and the app keeps working (graceful degradation).
- The Point Forecast API accepts up to 168 h ahead with a configurable
  time step. We request 3-hour steps for the next 7 days.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd
from loguru import logger

from app.core.config import settings

POINT_FORECAST_URL = "https://api.windy.com/api/point-forecast/v2"

# (param, levels, n_values) — order matters for response parsing.
# "wind" returns 2 values per level (speed, direction); every other
# parameter returns exactly 1 value per level.
PARAMS: dict[str, list[str]] = {
    "wind": ["surface", "850h", "700h", "500h", "300h"],
    "temp": ["surface", "850h", "700h", "500h", "300h"],
    "rh": ["surface"],
    "dewpoint": ["surface"],
    "prmsl": ["surface"],
    "cape": ["surface"],
    "precip": ["surface"],
}

WIND = "wind"  # contributes 2 values per level


def _n_values(param: str) -> int:
    return 2 if param == WIND else 1


class WindyClient:
    """
    Thin async client for the Windy Point Forecast API.

    Usage:
        client = WindyClient()
        df = await client.fetch_forecast(lat=-26.82, lon=-65.22)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        time_step: int = 3,
        max_hours: int = 168,
        url: Optional[str] = None,
    ):
        self.api_key = api_key or settings.windy_api_key
        self.model = model or settings.windy_model
        self.time_step = time_step
        self.max_hours = max_hours
        self.url = url or settings.windy_point_forecast_url

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ── Public API ─────────────────────────────────────────────

    async def fetch_forecast(
        self,
        lat: float,
        lon: float,
        model: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a multi-level forecast and return a wide hourly DataFrame
        with the columns CIELO·TUC needs as ML features.

        Returns an empty DataFrame if the API is not configured or the
        request fails (the caller should degrade gracefully).
        """
        if not self.is_configured:
            logger.warning("Windy API key not configured — skipping Windy fetch")
            return pd.DataFrame()

        model = model or self.model
        body = {
            "lat": lat,
            "lon": lon,
            "model": model,
            "parameters": list(PARAMS.keys()),
            "levels": self._all_levels(),
            "key": self.api_key,
            "timeStep": self.time_step,
            "maxTime": self.max_hours,
        }

        try:
            async with httpx.AsyncClient(timeout=40) as client:
                resp = await client.post(self.url, json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning(f"Windy API request failed (lat={lat}, lon={lon}): {e}")
            return pd.DataFrame()

        if not data or "rows" not in data:
            logger.warning("Windy API returned an empty payload")
            return pd.DataFrame()

        return self._parse_response(data, model)

    # ── Response parsing ───────────────────────────────────────

    def _all_levels(self) -> list[str]:
        """Ordered, de-duplicated list of levels across all parameters."""
        seen: list[str] = []
        for levels in PARAMS.values():
            for lvl in levels:
                if lvl not in seen:
                    seen.append(lvl)
        return seen

    def _parse_response(self, data: dict, model: str) -> pd.DataFrame:
        """
        Convert the Windy nested JSON into a wide hourly DataFrame.

        Windy Point Forecast response layout:
          ts:      ordered list of ISO timestamps
          headers: {param: [[level, ...]]} — levels offered per parameter
          rows:    [ {time_idx, level_idx, values:[...]}, ... ]
                   one entry per (time, level) combination. `values` holds
                   the requested parameters that EXIST at that level, in the
                   parameter order of `headers`; "wind" contributes
                   [speed, dir] per level.

        Output columns (naming matches FEATURE_COLS where possible):
          timestamp, temperature_c, humidity_pct, dew_point_c,
          pressure_hpa, wind_speed_kmh, wind_direction_deg,
          cape_j_kg, precip_mm, wind_speed_850h_kmh, wind_dir_850h_deg,
          wind_speed_700h_kmh, wind_dir_700h_deg, temp_850h_c, temp_700h_c
        """
        ts_list = data.get("ts", [])
        if not ts_list:
            return pd.DataFrame()

        # Ordered, de-duplicated global level list (row.level_idx indexes it).
        global_levels: list[str] = []
        param_levels: dict[str, list[str]] = {}
        for param in data.get("headers", {}):
            levels = self._flatten_levels(data["headers"][param])
            param_levels[param] = levels
            for lvl in levels:
                if lvl not in global_levels:
                    global_levels.append(lvl)

        if not param_levels:
            return pd.DataFrame()

        records: list[dict] = []
        for row in data.get("rows", []):
            t_idx = row.get("time_idx")
            l_idx = row.get("level_idx")
            values = row.get("values") or []
            if t_idx is None or l_idx is None or t_idx >= len(ts_list):
                continue
            if l_idx >= len(global_levels):
                continue
            level_name = global_levels[l_idx]

            rec: dict = {
                "timestamp": pd.to_datetime(ts_list[t_idx], utc=True),
                "_model": model,
            }
            offset = 0
            for param, levels in param_levels.items():
                if level_name not in levels:
                    continue
                n = _n_values(param)
                if offset + n > len(values):
                    break
                chunk = values[offset: offset + n]
                offset += n
                if param == WIND:
                    rec[f"wind_speed_{level_name}_ms"] = chunk[0]
                    rec[f"wind_dir_{level_name}_deg"] = chunk[1]
                else:
                    rec[f"{param}_{level_name}"] = chunk[0]
            records.append(rec)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").groupby("timestamp", as_index=False).first()
        return self._to_feature_frame(df)

    @staticmethod
    def _flatten_levels(header: list) -> list[str]:
        out: list[str] = []
        for chunk in header:
            if isinstance(chunk, list):
                out.extend(chunk)
            else:
                out.append(chunk)
        return out

    # ── Feature mapping ────────────────────────────────────────

    def _to_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Windy parameter/level columns onto CIELO·TUC feature names."""
        out = pd.DataFrame()
        out["timestamp"] = df["timestamp"]
        out["temperature_c"] = df.get("temp_surface")
        out["humidity_pct"] = df.get("rh_surface")
        out["dew_point_c"] = df.get("dewpoint_surface")
        out["pressure_hpa"] = df.get("prmsl_surface")
        out["cape_j_kg"] = df.get("cape_surface")

        # Precip from Windy is cumulative; difference to get per-step mm.
        precip = df.get("precip_surface")
        if precip is not None:
            precip_mm = precip.diff().clip(lower=0).fillna(0.0)
            if self.time_step > 1:
                precip_mm = precip_mm / self.time_step  # normalize to hourly
            out["precip_mm"] = precip_mm
        else:
            out["precip_mm"] = 0.0

        # Wind speed m/s → km/h
        for key in ("surface", "850h", "700h", "500h", "300h"):
            speed_ms = df.get(f"wind_speed_{key}_ms")
            if speed_ms is None:
                continue
            speed_kmh = speed_ms * 3.6
            if key == "surface":
                out["wind_speed_kmh"] = speed_kmh
                out["wind_direction_deg"] = df.get(f"wind_dir_{key}_deg")
            else:
                out[f"wind_speed_{key}_kmh"] = speed_kmh
                out[f"wind_dir_{key}_deg"] = df.get(f"wind_dir_{key}_deg")

        # Upper-level temperatures (useful for instability / Zonda analysis)
        for key in ("850h", "700h", "500h"):
            temp = df.get(f"temp_{key}")
            if temp is not None:
                out[f"temp_{key}_c"] = temp

        # Andes-to-plain pressure differential (Zonda proxy):
        # surface pressure minus a proxy of cordillera pressure. Windy
        # does not expose pressure at altitude directly, so we approximate
        # the cordillera value from the 700h level temperature.
        if "pressure_hpa" in out and "temp_700h_c" in out:
            alt_proxy = 700.0
            out["cordillera_pressure_hpa"] = out["pressure_hpa"] - (
                out["pressure_hpa"] - alt_proxy
            ) * 0.5
            out["thermal_differential_c"] = (
                out["temperature_c"] - out["temp_700h_c"]
            )
            out["andes_plain_pressure_diff"] = (
                out["pressure_hpa"] - out["cordillera_pressure_hpa"]
            )

        return out

    # ── Small helpers used by the pipeline ─────────────────────

    @staticmethod
    def merge_into(
        base: pd.DataFrame,
        windy: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge Windy forecast data into a base (observed) DataFrame,
        keeping base observations where available.
        """
        if windy.empty:
            return base
        if base.empty:
            return windy

        base = base.copy()
        merged = base.merge(
            windy,
            on="timestamp",
            how="outer",
            suffixes=("", "_windy"),
        )
        # Fill feature gaps from Windy where base is missing
        for col in base.columns:
            if col == "timestamp":
                continue
            wcol = f"{col}_windy"
            if wcol in merged.columns:
                merged[col] = merged[col].fillna(merged[wcol])
                merged = merged.drop(columns=[wcol])
        return merged.sort_values("timestamp").reset_index(drop=True)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
