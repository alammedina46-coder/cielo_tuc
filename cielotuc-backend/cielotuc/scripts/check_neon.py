"""Quick integration check against Neon DB."""
import os
from sqlalchemy import create_engine, text

URL = os.environ.get("DATABASE_URL_SYNC", "")

e = create_engine(URL)
with e.connect() as conn:
    tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")).fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    for t in tables:
        cnt = conn.execute(text(f'SELECT count(*) FROM "{t[0]}"')).scalar()
        print(f"  {t[0]}: {cnt} rows")

    # Check admin user
    users = conn.execute(text("SELECT email, role FROM users")).fetchall()
    print(f"\nUsers: {[(u[0], u[1]) for u in users]}")

    # Check zones
    zones = conn.execute(text("SELECT id, name, department FROM zones LIMIT 5")).fetchall()
    print(f"\nZones (first 5): {[(z[0], z[1], z[2]) for z in zones]}")

    # Check stations
    stations = conn.execute(text("SELECT id, name, source, zone_id FROM weather_stations LIMIT 5")).fetchall()
    print(f"\nStations (first 5): {[(s[0], s[1], s[2], s[3]) for s in stations]}")

    # Check model versions
    models = conn.execute(text("SELECT version, is_active FROM model_versions")).fetchall()
    print(f"\nModel versions: {[(m[0], m[1]) for m in models]}")
