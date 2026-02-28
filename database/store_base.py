"""
Abstract Context Store — Interface for all storage backends.

Implementations:
  - SqlContextStore      (PostgreSQL / MySQL / SQLite via SQLAlchemy)
  - InMemoryContextStore (dict-based, single-process, no persistence)
  - FileContextStore     (JSON files on disk, single-process, durable)

The ContextTracker in context/tracker.py already has an in-memory ContextStore.
These implementations follow the same interface as PostgresContextStore
(database/store.py in the original) so they're drop-in compatible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from models.schemas import Contact


class BaseContextStore(ABC):
    """Interface that all context store backends must implement."""

    # ── Contacts ──────────────────────────────────────────────

    @abstractmethod
    async def get_contact(self, contact_id: str) -> Optional[Contact]:
        ...

    @abstractmethod
    async def get_contact_by_external_id(self, external_id: str) -> Optional[Contact]:
        ...

    @abstractmethod
    async def find_contact_by_address(self, channel: str, address: str) -> Optional[Contact]:
        ...

    @abstractmethod
    async def upsert_contact(self, contact: Contact) -> Contact:
        ...

    # ── Conversations ─────────────────────────────────────────

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    async def find_active_conversation(self, contact_id: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    async def create_conversation(self, contact_id: str, channel: str = "", **kwargs) -> dict[str, Any]:
        ...

    @abstractmethod
    async def update_conversation(self, conversation_id: str, **kwargs) -> None:
        ...

    # ── Messages ──────────────────────────────────────────────

    @abstractmethod
    async def add_message(self, conversation_id: str, direction: str, channel: str,
                          content: str, metadata: dict = None, channel_message_id: str = "") -> dict[str, Any]:
        ...

    @abstractmethod
    async def get_conversation_messages(self, conversation_id: str, limit: int = 50) -> list[dict[str, Any]]:
        ...

    # ── Follow-ups ────────────────────────────────────────────

    @abstractmethod
    async def get_followup(self, followup_id: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    async def get_followup_by_external_id(self, external_id: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    async def create_followup(self, data: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    async def update_followup(self, followup_id: str, **kwargs) -> None:
        ...

    @abstractmethod
    async def get_pending_followups(self, limit: int = 100) -> list[dict[str, Any]]:
        ...

    # ── State Bindings ────────────────────────────────────────

    @abstractmethod
    async def get_state_binding(self, conversation_id: str, process_type: str) -> Optional[dict[str, Any]]:
        ...

    @abstractmethod
    async def upsert_state_binding(self, conversation_id: str, process_type: str, **kwargs) -> None:
        ...

    # ── Voice Calls ───────────────────────────────────────────

    @abstractmethod
    async def save_voice_call(self, call_data: dict[str, Any]) -> None:
        ...
