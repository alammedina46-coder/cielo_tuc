"""Seed sensor_readings for ALL 11 stations using NASA POWER + zone-based variation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text

URL = "postgresql://neondb_owner:npg_WhzpwYo76uJP@ep-hidden-bread-axfrgmam-pooler.c-4.us-east-2.aws.neon.tech/neondb?sslmode=require"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Station configs: (station_id, zone_id, lat_offset, lng_offset, alt_offset, name)
STATIONS = [
    (1, 1, 0, 0, 0, "SMN Capital"),
    (2, 2, 0.02, 0.01, 50, "NASA Yerba Buena"),
    (3, 3, -0.05, 0.03, -50, "SMN Concepción"),
    (4, 4, 0.01, -0.02, -20, "IoT Banda Sali"),
    (5, 5, -0.03, -0.01, -30, "IoT Lules"),
    (6, 6, -0.08, 0.05, 100, "IoT Tafí Viejo"),
    (7, 7, 0.04, -0.04, 200, "IoT Aguilares"),
    (8, 8, -0.10, 0.02, -10, "IoT Bella Vista"),
    (9, 9, 0.06, 0.06, -40, "IoT Monteros"),
    (10, 10, -0.02, 0.08, -60, "IoT Simoca"),
    (11, 11, 0.08, -0.06, 300, "IoT La Cocha"),
]


def fetch_base_data():
    """Fetch NASA POWER for base station (Capital)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    params = {
        "parameters": "T2M,RH2M,WS10M,WD10M,PS,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": -65.22,
        "latitude": -26.82,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "data_format": "JSON",
    }
    print(f"  Fetching NASA POWER base data ({start.date()} to {end.date()})...")
    r = httpx.get(NASA_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]

    records = []
    for ts_key, val in data.get("T2M", {}).items():
        if val == -999:
            continue
        dt = datetime(int(ts_key[:4]), int(ts_key[4:6]), int(ts_key[6:8]), int(ts_key[8:10]), tzinfo=timezone.utc)
        records.append({
            "timestamp": dt,
            "temperature_c": data["T2M"].get(ts_key, -999),
            "humidity_pct": data["RH2M"].get(ts_key, -999),
            "wind_speed_kmh": data["WS10M"].get(ts_key, -999),
            "wind_direction_deg": data["WD10M"].get(ts_key, -999),
            "pressure_hpa": data["PS"].get(ts_key, -999),
            "precip_mm": data["PRECTOTCORR"].get(ts_key, -999),
            "cloud_cover_pct": data.get("ALLSKY_SFC_SW_DWN", {}).get(ts_key, -999),
        })

    df = pd.DataFrame(records).replace(-999, None)
    print(f"  Got {len(df)} records")
    return df


def main():
    print("=== CIELO·TUC Multi-Station Seed ===\n")
    engine = create_engine(URL)

    with engine.connect() as conn:
        existing = conn.execute(text("SELECT count(DISTINCT station_id) FROM sensor_readings")).scalar()
        print(f"  Stations with readings: {existing}/11")
        if existing >= 11:
            print("  All stations already seeded, skipping.")
            return

    print("1. Fetching base data...")
    df = fetch_base_data()
    rng = np.random.default_rng(42)

    print("\n2. Inserting readings for each station...")
    total = 0
    with engine.connect() as conn:
        for sid, zid, lat_off, lng_off, alt_off, name in STATIONS:
            existing = conn.execute(
                text("SELECT count(*) FROM sensor_readings WHERE station_id = :sid"),
                {"sid": sid}
            ).scalar()
            if existing > 0:
                print(f"  Station {sid} ({name}): already {existing} rows, skip")
                continue

            for _, row in df.iterrows():
                ts = row["timestamp"]
                temp = float(row["temperature_c"]) if pd.notna(row["temperature_c"]) else 20.0
                hum = float(row["humidity_pct"]) if pd.notna(row["humidity_pct"]) else 60.0
                pres = float(row["pressure_hpa"]) if pd.notna(row["pressure_hpa"]) else 1010.0
                wind = float(row["wind_speed_kmh"]) if pd.notna(row["wind_speed_kmh"]) else 10.0
                wdir = float(row["wind_direction_deg"]) if pd.notna(row["wind_direction_deg"]) else 180.0
                precip = float(row["precip_mm"]) if pd.notna(row["precip_mm"]) else 0.0
                cloud = float(row["cloud_cover_pct"]) if pd.notna(row["cloud_cover_pct"]) else 50.0

                # Zone-based variation
                temp += alt_off * -0.006 + rng.normal(0, 0.5)
                hum = max(10, min(100, hum + rng.normal(0, 3)))
                wind = max(0, wind + rng.normal(0, 2))
                wind_gust = wind * (1.2 + rng.uniform(0, 0.3))
                dew = temp - (100 - hum) / 5
                cape = max(0, (temp - dew) * 50)
                feels = temp - 2 if hum > 70 else temp + 1
                uv = max(0, min(11, cloud * 0.08 + 3))
                vis = max(1, 10 - cloud * 0.08)

                conn.execute(text("""
                    INSERT INTO sensor_readings
                        (station_id, timestamp, temperature_c, feels_like_c,
                         humidity_pct, pressure_hpa, pressure_sea_level_hpa,
                         dew_point_c, precip_mm, precip_3h_mm, precip_6h_mm, precip_24h_mm,
                         wind_speed_kmh, wind_direction_deg, wind_gust_kmh,
                         visibility_km, cloud_cover_pct, uv_index,
                         cape_j_kg, k_index,
                         cordillera_pressure_hpa, thermal_differential_c,
                         enso_index, is_validated, quality_flag)
                    VALUES
                        (:sid, :ts, :temp, :feels, :hum, :pres, :pres_sl,
                         :dew, :precip, 0, 0, :p24,
                         :wind, :wdir, :gust,
                         :vis, :cloud, :uv,
                         :cape, :ki,
                         :cord, :therm,
                         0, true, 'auto')
                """), {
                    "sid": sid, "ts": ts,
                    "temp": round(temp, 2), "feels": round(feels, 2),
                    "hum": round(hum, 2), "pres": round(pres, 2), "pres_sl": round(pres + 5, 2),
                    "dew": round(dew, 2), "precip": round(precip, 3),
                    "p24": round(precip * 6, 3),
                    "wind": round(wind, 2), "wdir": round(wdir, 1), "gust": round(wind_gust, 2),
                    "vis": round(vis, 1), "cloud": round(cloud, 1), "uv": round(uv, 1),
                    "cape": round(cape, 1), "ki": round(max(0, temp - dew), 1),
                    "cord": round(890 + rng.normal(0, 3), 1),
                    "therm": round(max(0, (temp - 15) * 0.3 + rng.normal(0, 2)), 2),
                })
                total += 1

            print(f"  Station {sid} ({name}): +{len(df)} readings")
        conn.commit()

    print(f"\n  Total inserted: {total}")

    # Verify
    with engine.connect() as conn:
        for sid in range(1, 12):
            cnt = conn.execute(text("SELECT count(*) FROM sensor_readings WHERE station_id = :sid"), {"sid": sid}).scalar()
            print(f"  Station {sid}: {cnt} rows")

    print("\nDone!")


if __name__ == "__main__":
    main()
