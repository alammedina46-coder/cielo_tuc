"""
app/db/session.py
─────────────────
Async SQLAlchemy engine + session factory.
Use `get_db` as a FastAPI dependency to get a scoped session.
"""
from collections.abc import AsyncGenerator
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


def _clean_async_url(url: str) -> str:
    """Strip psycopg2-only params (sslmode, channel_binding) that asyncpg rejects."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    # Remove params that asyncpg doesn't understand
    for bad_key in ("sslmode", "channel_binding"):
        params.pop(bad_key, None)
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


# ── Engine ─────────────────────────────────────────────────────
db_url = _clean_async_url(settings.database_url)
engine = create_async_engine(
    db_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # auto-reconnect on stale connections
    connect_args={"ssl": "require" if "neon" in db_url else False},
)

# ── Session factory ────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ── Base class for all ORM models ──────────────────────────────
class Base(DeclarativeBase):
    pass


# ── FastAPI dependency ─────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session; always close on exit."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
