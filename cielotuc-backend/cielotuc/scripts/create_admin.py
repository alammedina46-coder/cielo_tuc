"""
scripts/create_admin.py
──────────────────────
Crea un usuario admin en la base de datos Neon.

Uso:
  cd cielotuc-backend/cielotuc
  set DATABASE_URL_SYNC=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
  python scripts/create_admin.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.models.user import Base, User
from app.core.security import hash_password


ADMIN_EMAIL = "admin@cielotuc.com"
ADMIN_PASSWORD = "CieloTuc2026!"
ADMIN_NAME = "Administrador CIELO·TUC"
ADMIN_ROLE = "admin"


def create_admin():
    db_url = os.environ.get("DATABASE_URL_SYNC")
    if not db_url:
        print("ERROR: Set DATABASE_URL_SYNC env var first.")
        sys.exit(1)

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        existing = session.query(User).filter(User.email == ADMIN_EMAIL).first()
        if existing:
            print(f"Admin user already exists: {ADMIN_EMAIL}")
            return

        admin = User(
            email=ADMIN_EMAIL,
            name=ADMIN_NAME,
            hashed_password=hash_password(ADMIN_PASSWORD),
            role=ADMIN_ROLE,
            is_active=True,
        )
        session.add(admin)
        session.commit()
        print(f"Admin user created:")
        print(f"  Email:    {ADMIN_EMAIL}")
        print(f"  Password: {ADMIN_PASSWORD}")
        print(f"  Role:     {ADMIN_ROLE}")


if __name__ == "__main__":
    create_admin()
