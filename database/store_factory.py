"""
Store Factory — Create the right context store backend from configuration.

Configuration in settings.yaml:
    database:
      # Database URL — determines the SQL backend
      #   PostgreSQL:  postgresql://user:pass@host:5432/db
      #   MySQL:       mysql://user:pass@host:3306/db
      #   SQLite:      sqlite:///./converse_agent.db
      url: "sqlite:///./converse_agent.db"

      # Context store backend — where runtime state lives
      #   "sql"      — Same database as above (default for production)
      #   "memory"   — In-memory dicts (development, testing)
      #   "file"     — JSON files on disk (small deployments, demos)
      store_backend: "memory"

      # For file backend: directory path
      store_file_dir: "./data"

      # Redis URL (for queue backend, separate from store)
      redis_url: "redis://localhost:6379"

Usage:
    from database.store_factory import create_store, get_store
    store = create_store(config)     # Create from config dict
    store = get_store()              # Get singleton instance
"""
from __future__ import annotations

import structlog
from typing import Optional

from database.store_base import BaseContextStore

logger = structlog.get_logger()

_instance: Optional[BaseContextStore] = None


def create_store(config: dict = None) -> BaseContextStore:
    """
    Factory: create the appropriate context store backend.

    Args:
        config: dict with keys:
            store_backend: "sql" | "memory" | "file"  (default: "memory")
            store_file_dir: str (for file backend, default: "./data")
            url: str (database URL for sql backend)
    """
    global _instance
    if _instance is not None:
        return _instance

    config = config or {}
    backend = config.get("store_backend", "memory")

    if backend == "sql":
        from database.store import SqlContextStore
        _instance = SqlContextStore()
        logger.info("store_created", backend="sql")

    elif backend == "file":
        from database.store_file import FileContextStore
        data_dir = config.get("store_file_dir", "./data")
        _instance = FileContextStore(data_dir=data_dir)
        logger.info("store_created", backend="file", data_dir=data_dir)

    else:  # "memory" or default
        from database.store_memory import InMemoryContextStore
        _instance = InMemoryContextStore()
        logger.info("store_created", backend="memory")

    return _instance


def get_store() -> BaseContextStore:
    """Return the singleton store instance, creating a memory store if none exists."""
    global _instance
    if _instance is None:
        _instance = create_store()
    return _instance


def reset_store() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
