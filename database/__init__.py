"""
Database layer — Multi-backend persistence.

Backends:
  - SQL (PostgreSQL / MySQL / SQLite via SQLAlchemy async)
  - In-memory (dict-based, for development/testing)
  - File (JSON files on disk, for small deployments)

Quick start:
  from database import create_store, get_store
  store = create_store({"store_backend": "memory"})
  contact = await store.get_contact("c1")
"""
from database.models import (
    Base, ContactRow, ConversationRow, MessageRow,
    FollowUpRow, StateBindingRow, VoiceCallRow,
)
from database.session import get_engine, get_session, init_db, close_db
from database.store_base import BaseContextStore
from database.store import SqlContextStore, PostgresContextStore
from database.store_memory import InMemoryContextStore
from database.store_file import FileContextStore
from database.store_factory import create_store, get_store, reset_store
from database.adapter import StoreAdapter, create_configured_store, create_plain_store

__all__ = [
    # ORM models
    "Base", "ContactRow", "ConversationRow", "MessageRow",
    "FollowUpRow", "StateBindingRow", "VoiceCallRow",
    # Session management
    "get_engine", "get_session", "init_db", "close_db",
    # Store interface
    "BaseContextStore",
    # Store backends
    "SqlContextStore", "PostgresContextStore",
    "InMemoryContextStore", "FileContextStore",
    # Factory
    "create_store", "get_store", "reset_store",
    # Adapter (bridges ContextStore ↔ BaseContextStore)
    "StoreAdapter", "create_configured_store", "create_plain_store",
]
