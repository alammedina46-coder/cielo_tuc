"""
scripts/init_db.py
───────────────────
Run once to:
  1. Create all tables from ORM models
  2. Seed the 17 Tucumán departments as Zone rows
  3. Seed sample weather stations (SMN + EEAOC)
  4. Insert a placeholder ModelVersion row so the API doesn't error

Usage:
  python scripts/init_db.py
"""

import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.core.config import settings
from app.db.session import Base
from app.models.weather import Zone, WeatherStation, ModelVersion

# Import all models so Base.metadata is populated
import app.models.weather  # noqa: F401


# ── Tucumán departments seed data ─────────────────────────────
DEPARTMENTS = [
    # (name, lat, lng, altitude_m, is_mountain, population, impermeable_pct)
    ("Capital",         -26.824, -65.204, 450,  False, 549000, 0.80),
    ("Yerba Buena",     -26.816, -65.316, 520,  False, 130000, 0.60),
    ("Tafí Viejo",      -26.728, -65.268, 580,  False,  95000, 0.55),
    ("Cruz Alta",       -26.600, -64.900, 310,  False, 280000, 0.30),
    ("Burruyacú",       -26.500, -64.750, 380,  False,  45000, 0.10),
    ("Trancas",         -26.230, -65.370, 780,  True,   22000, 0.08),
    ("Tafí del Valle",  -26.870, -65.720, 2040, True,   18000, 0.05),
    ("Lules",           -27.030, -65.350, 420,  False,  65000, 0.25),
    ("Famaillá",        -27.050, -65.400, 380,  False,  40000, 0.20),
    ("Monteros",        -27.170, -65.500, 360,  False,  75000, 0.25),
    ("Simoca",          -27.270, -65.360, 340,  False,  35000, 0.12),
    ("Leales",          -27.100, -65.050, 290,  False,  42000, 0.10),
    ("Graneros",        -27.570, -65.520, 620,  False,  20000, 0.08),
    ("La Cocha",        -27.780, -65.580, 680,  False,  18000, 0.07),
    ("J.B. Alberdi",    -27.580, -65.620, 560,  False,  25000, 0.15),
    ("Chicligasta",     -27.370, -65.550, 440,  False,  55000, 0.20),
    ("Río Chico",       -27.470, -65.800, 820,  True,   15000, 0.06),
]

# ── Weather stations seed data ─────────────────────────────────
# (zone_name, code, name, source, lat, lng, altitude_m)
STATIONS = [
    ("Capital",     "SMN-TUC-AER", "Aeropuerto Tucumán (SMN)",   "smn",    -26.840, -65.105, 447),
    ("Tafí del Valle","SMN-TUC-TAF","Tafí del Valle (SMN)",      "smn",    -26.852, -65.716, 2014),
    ("Famaillá",    "SMN-TUC-FAM", "Famaillá (SMN)",             "smn",    -27.055, -65.397, 363),
    ("Capital",     "EEAOC-CAP",   "EEAOC Capital",              "eeaoc",  -26.810, -65.210, 455),
    ("Lules",       "EEAOC-LUL",   "EEAOC Lules",                "eeaoc",  -27.020, -65.340, 420),
    ("Cruz Alta",   "EEAOC-CRU",   "EEAOC Cruz Alta",            "eeaoc",  -26.590, -64.880, 315),
    ("Monteros",    "EEAOC-MON",   "EEAOC Monteros",             "eeaoc",  -27.165, -65.495, 362),
    ("Simoca",      "EEAOC-SIM",   "EEAOC Simoca",               "eeaoc",  -27.260, -65.355, 338),
    ("Capital",     "IOT-CAP-001", "IoT Centro Tucumán",         "own_iot",-26.824, -65.204, 450),
    ("Yerba Buena", "IOT-YB-001",  "IoT Yerba Buena",            "own_iot",-26.816, -65.316, 520),
    ("Tafí Viejo",  "IOT-TV-001",  "IoT Tafí Viejo",             "own_iot",-26.728, -65.268, 580),
    ("Graneros",    "IOT-GRA-001", "IoT Graneros",               "own_iot",-27.570, -65.520, 620),
]


async def init():
    engine = create_async_engine(settings.database_url, echo=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created")

    async with SessionLocal() as db:
        # Seed zones
        for (name, lat, lng, alt, mountain, pop, imp) in DEPARTMENTS:
            zone = Zone(
                name=name,
                department=name,
                zone_type="department",
                latitude=lat,
                longitude=lng,
                altitude_m=alt,
                is_mountain=mountain,
                population=pop,
                impermeable_pct=imp,
            )
            db.add(zone)
        await db.flush()
        print(f"✅ Seeded {len(DEPARTMENTS)} departments")

        # Build name → id map
        from sqlalchemy import select
        result = await db.execute(select(Zone))
        zones_by_name = {z.name: z.id for z in result.scalars().all()}

        # Seed stations
        for (zone_name, code, sname, source, lat, lng, alt) in STATIONS:
            zone_id = zones_by_name.get(zone_name)
            if zone_id is None:
                print(f"⚠️  Zone '{zone_name}' not found — skipping station {code}")
                continue
            station = WeatherStation(
                zone_id=zone_id,
                name=sname,
                station_code=code,
                source=source,
                latitude=lat,
                longitude=lng,
                altitude_m=alt,
                is_active=True,
            )
            db.add(station)
        await db.flush()
        print(f"✅ Seeded {len(STATIONS)} weather stations")

        # Seed a placeholder model version
        mv = ModelVersion(
            version="0.0-bootstrap",
            trained_at=datetime.utcnow(),
            architecture="CNN-LSTM",
            training_samples=0,
            accuracy_overall=0.0,
            is_active=True,
            notes=(
                "Bootstrap placeholder. Replace by running "
                "scripts/train_initial_model.py"
            ),
        )
        db.add(mv)
        await db.commit()
        print("✅ Seeded placeholder ModelVersion")

    await engine.dispose()
    print("\n🚀 Database initialization complete!")
    print("   Next step: python scripts/train_initial_model.py")


if __name__ == "__main__":
    asyncio.run(init())
