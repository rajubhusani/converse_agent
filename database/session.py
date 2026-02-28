"""
Async database session management — PostgreSQL, MySQL, SQLite.

Driver mapping:
  postgresql://  → postgresql+asyncpg://     (requires asyncpg)
  mysql://       → mysql+aiomysql://         (requires aiomysql)
  sqlite://      → sqlite+aiosqlite://       (requires aiosqlite)

Usage:
    await init_db()                    # Call once at startup
    async with get_session() as db:    # Use in request handlers
        result = await db.execute(...)
    await close_db()                   # Call at shutdown
"""
from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker,
)

from config.settings import get_settings
from database.models import Base

logger = structlog.get_logger()

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _to_async_url(db_url: str) -> str:
    """Convert a sync database URL to its async driver equivalent."""
    replacements = [
        ("postgresql://", "postgresql+asyncpg://"),
        ("postgres://", "postgresql+asyncpg://"),
        ("mysql://", "mysql+aiomysql://"),
        ("mysql+pymysql://", "mysql+aiomysql://"),
        ("sqlite://", "sqlite+aiosqlite://"),
    ]
    for sync_prefix, async_prefix in replacements:
        if db_url.startswith(sync_prefix):
            return db_url.replace(sync_prefix, async_prefix, 1)
    # Already has async driver or unknown — return as-is
    return db_url


def _engine_kwargs(db_url: str) -> dict:
    """Return database-specific engine configuration."""
    settings = get_settings()
    base = {"echo": settings.debug}

    if "sqlite" in db_url:
        # SQLite: no connection pooling needed
        return {**base, "connect_args": {"check_same_thread": False}}

    # PostgreSQL / MySQL: connection pool tuning
    return {
        **base,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_timeout": 30,
        "pool_recycle": 1800,
        "pool_pre_ping": True,
    }


def get_engine() -> AsyncEngine:
    """Return the global engine, creating it if needed."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = _to_async_url(settings.database.url)
        kwargs = _engine_kwargs(db_url)
        _engine = create_async_engine(db_url, **kwargs)
        logger.info("database_engine_created",
                     dialect=_engine.dialect.name,
                     url=str(_engine.url).split("@")[-1] if "@" in str(_engine.url) else str(_engine.url))
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async session scope."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables. Call once at application startup."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized",
                dialect=engine.dialect.name,
                tables=list(Base.metadata.tables.keys()))


async def close_db() -> None:
    """Dispose engine connections. Call at application shutdown."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("database_closed")
