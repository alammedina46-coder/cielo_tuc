"""
scripts/seed_neon.py
────────────────────
Pobla la base de datos Neon con las 17 zonas climáticas de Tucumán
y estaciones meteorológicas de ejemplo.

Uso:
  cd cielotuc-backend/cielotuc
  set DATABASE_URL_SYNC=postgresql://user:pass@ep-xxx.neon.tech/cielotuc?sslmode=require
  python scripts/seed_neon.py
"""

import os
import sys

# Ensure we can import from app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.models.weather import Base, Zone, WeatherStation


# ── 17 zonas climáticas de Tucumán ─────────────────────────────
ZONES = [
    # (name, department, lat, lng, alt_m, population, is_mountain, impermeable_pct)
    ("Capital", "Capital", -26.8248, -65.2226, 450, 548865, False, 0.70),
    ("San Miguel de Tucumán", "Capital", -26.8083, -65.2176, 450, 548865, False, 0.65),
    ("Concepción", "Concepción", -27.3417, -65.5958, 320, 62000, False, 0.40),
    ("Banda del Río Sali", "Capital", -26.7833, -65.1667, 430, 15000, False, 0.55),
    ("Lules", "Capital", -26.9167, -65.3333, 460, 20000, False, 0.45),
    ("Villa Nougués", "Capital", -26.8333, -65.1333, 600, 8000, False, 0.35),
    ("Yerba Buena", "Capital", -26.8167, -65.1833, 470, 30000, False, 0.50),
    ("Tafí Viejo", "Tafí Viejo", -26.7333, -65.2667, 550, 35000, False, 0.40),
    ("Bella Vista", "Burruyacú", -26.9500, -65.3000, 480, 12000, False, 0.30),
    ("Burruyacú", "Burruyacú", -26.5000, -64.7500, 500, 35000, False, 0.25),
    ("Monteros", "Río Chico", -27.1667, -65.5000, 310, 25000, False, 0.35),
    ("Santiago del Estero", "Río Chico", -27.2000, -65.3500, 300, 8000, False, 0.25),
    ("Simoca", "Simoca", -27.2667, -65.3667, 290, 10000, False, 0.20),
    ("Famaillá", "Famaillá", -27.0500, -65.4000, 340, 15000, False, 0.30),
    ("Aguilares", "Rio Chico", -27.0333, -65.6167, 300, 12000, False, 0.25),
    ("San Pedro de Colalao", "Trancas", -26.2333, -65.5000, 1100, 5000, True, 0.15),
    ("Tafí del Valle", "Tafí del Valle", -26.8500, -65.7167, 2000, 6000, True, 0.10),
]

# ── Estaciones meteorológicas de ejemplo ─────────────────────────
# Una por zona principal, fuente mixta (SMN, NASA POWER, propias)
STATIONS = [
    # (zone_index, name, code, source, lat, lng, alt, battery)
    (0, "SMN Capital", "SMN_CAP", "smn", -26.8248, -65.2226, 450, 100),
    (1, "NASA Yerba Buena", "NASA_YB", "nasa_power", -26.8167, -65.1833, 470, 100),
    (2, "SMN Concepción", "SMN_CON", "smn", -27.3417, -65.5958, 320, 100),
    (3, "IoT Banda Sali", "IOT_BRS", "own_iot", -26.7833, -65.1667, 430, 85),
    (4, "IoT Lules", "IOT_LUL", "own_iot", -26.9167, -65.3333, 460, 90),
    (6, "IoT Yerba Buena", "IOT_YB", "own_iot", -26.8167, -65.1833, 470, 75),
    (7, "SMN Tafí Viejo", "SMN_TAF", "smn", -26.7333, -65.2667, 550, 100),
    (10, "NASA Monteros", "NASA_MON", "nasa_power", -27.1667, -65.5000, 310, 100),
    (13, "IoT Famaillá", "IOT_FAM", "own_iot", -27.0500, -65.4000, 340, 60),
    (15, "SMN Colalao", "SMN_COL", "smn", -26.2333, -65.5000, 1100, 100),
    (16, "NASA Tafí del Valle", "NASA_TDV", "nasa_power", -26.8500, -65.7167, 2000, 100),
]


def seed():
    db_url = os.environ.get("DATABASE_URL_SYNC")
    if not db_url:
        print("ERROR: Set DATABASE_URL_SYNC env var first.")
        print("  export DATABASE_URL_SYNC=postgresql://user:pass@host/db?sslmode=require")
        sys.exit(1)

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        existing = session.query(Zone).count()
        if existing >= 17:
            print(f"DB already has {existing} zones — skipping seed.")
            return

        print(f"Seeding {len(ZONES)} zones...")
        zone_objs = []
        for i, (name, dept, lat, lng, alt, pop, mtn, imperm) in enumerate(ZONES):
            z = Zone(
                name=name,
                department=dept,
                zone_type="neighbourhood" if i > 0 else "department",
                latitude=lat,
                longitude=lng,
                altitude_m=alt,
                population=pop,
                is_mountain=mtn,
                impermeable_pct=imperm,
            )
            session.add(z)
            zone_objs.append(z)

        session.flush()  # get IDs

        print(f"Seeding {len(STATIONS)} weather stations...")
        for z_idx, name, code, source, lat, lng, alt, batt in STATIONS:
            ws = WeatherStation(
                zone_id=zone_objs[z_idx].id,
                name=name,
                station_code=code,
                source=source,
                latitude=lat,
                longitude=lng,
                altitude_m=alt,
                is_active=True,
                battery_pct=batt,
            )
            session.add(ws)

        session.commit()
        print(f"Done! {len(ZONES)} zones + {len(STATIONS)} stations seeded.")


if __name__ == "__main__":
    seed()
