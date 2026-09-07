"""
app/main.py
────────────
CIELO·TUC — FastAPI application factory.

Startup sequence:
  1. Connect to DB
  2. Load active ML model into memory
  3. Register all routers
  4. Configure CORS, middleware, error handlers
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.forecast import router as forecast_router
from app.api.v1.endpoints.notifications import notifications_router
from app.api.v1.endpoints.other import (
    zones_router,
    sensors_router,
    alerts_router,
    model_router,
)


# ── Lifespan (replaces deprecated @app.on_event) ──────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks, yield, then run shutdown tasks."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Pre-load the ML model so first request isn't slow
    try:
        from app.db.session import AsyncSessionLocal
        from app.services.prediction_service import PredictionService
        service = PredictionService()
        async with AsyncSessionLocal() as db:
            await service.load_model(db)
        logger.info("ML model loaded and ready")
    except Exception as e:
        logger.warning(f"Could not load ML model on startup (DB may be offline): {e}")

    # Start background scheduler (fetch data, predictions, etc.)
    if settings.scheduler_enabled:
        try:
            from app.services.tasks import start_scheduler, stop_scheduler
            start_scheduler()
        except Exception as e:
            logger.warning(f"Could not start scheduler: {e}")
        _stop_scheduler = stop_scheduler
    else:
        _stop_scheduler = None

    yield   # ← app is running

    if _stop_scheduler:
        _stop_scheduler()
    logger.info("Shutting down CIELO·TUC")


# ── App factory ────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Backend API for CIELO·TUC — AI-powered hyper-local "
            "weather prediction for Tucumán, Argentina."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(auth_router, prefix=prefix)
    app.include_router(forecast_router, prefix=prefix)
    app.include_router(zones_router, prefix=prefix)
    app.include_router(sensors_router, prefix=prefix)
    app.include_router(alerts_router, prefix=prefix)
    app.include_router(model_router, prefix=prefix)
    app.include_router(notifications_router, prefix=prefix)

    # ── Health check ──────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health():
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    # ── Debug: DB check (remove after fixing) ────────────────
    @app.get("/debug/db", tags=["Debug"])
    async def debug_db():
        from sqlalchemy import text
        from app.db.session import AsyncSessionLocal
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(text("SELECT COUNT(*) FROM zones"))
                count = result.scalar()
                return {"zones_count": count, "db": "ok"}
        except Exception as e:
            return {"db": "error", "detail": str(e)}

    # ── Debug: CORS check ────────────────────────────────────
    @app.get("/debug/cors", tags=["Debug"])
    async def debug_cors():
        return {"cors_origins": settings.cors_origins}

    return app


app = create_app()


# ── Dev runner ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
