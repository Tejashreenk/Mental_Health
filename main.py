"""
Mental Health AI Recommendation Engine
Entry point — FastAPI application factory.

Run with:
  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from api.routes import router
from config import get_settings
from database.connection import init_db, close_db
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    log.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]")
    await init_db()
    log.info("Database initialised.")
    yield
    await close_db()
    log.info("Shutdown complete.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-powered recommendation engine for a mental health chatbot. "
            "Provides personalised resources using hybrid NLP + ML recommendations."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # ── Security middleware ───────────────────────────────────────────────────
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.DEBUG else ["yourdomain.com", "*.yourdomain.com"],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
