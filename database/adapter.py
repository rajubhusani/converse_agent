"""
StoreAdapter — Bridges ContextTracker's ContextStore interface with
configurable BaseContextStore backends (SQL, Memory, File).

The ContextTracker in context/tracker.py uses Pydantic model objects
(Contact, Conversation, FollowUp). The database/ backends return dicts.
This adapter translates between the two so the rest of the app doesn't
change while the storage backend is fully configurable.

Usage:
    from database.adapter import create_configured_store

    # Returns a ContextStore-compatible object backed by whatever
    # store_backend is configured in settings.yaml
    store = create_configured_store()
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import get_settings
from context.tracker import ContextStore
from database.store_base import BaseContextStore
from database.store_factory import create_store
from models.schemas import (
    Contact, ContactChannel, ChannelType,
    Conversation, ConversationStatus,
    FollowUp, FollowUpStatus, Message, MessageDirection,
    StateBinding,
)

logger = structlog.get_logger()


class StoreAdapter(ContextStore):
    """
    Wraps a BaseContextStore backend and exposes the ContextStore interface.

    - Contact operations: delegate directly (both use Contact objects)
    - Conversation/FollowUp/StateBinding: convert between dicts and Pydantic models
    - Preserves full ContextStore API so ContextTracker works unchanged
    """

    def __init__(self, backend: BaseContextStore):
        super().__init__()
        self._backend = backend
        self._backend_name = type(backend).__name__
        logger.info("store_adapter_initialized", backend=self._backend_name)

    @property
    def backend(self) -> BaseContextStore:
        """Access the underlying backend directly if needed."""
        return self._backend

    # ── Contacts (pass-through — both sides use Contact) ───────

    async def upsert_contact(self, contact: Contact) -> Contact:
        return await self._backend.upsert_contact(contact)

    async def get_contact(self, contact_id: str) -> Optional[Contact]:
        return await self._backend.get_contact(contact_id)

    async def find_contact_by_address(self, channel, address: str) -> Optional[Contact]:
        ch_str = channel.value if hasattr(channel, "value") else str(channel)
        return await self._backend.find_contact_by_address(ch_str, address)

    async def list_contacts(self, filters: dict[str, Any] = None) -> list[Contact]:
        # Backend doesn't have list_contacts — fall back to parent's in-memory
        return await super().list_contacts(filters)

    # ── Conversations (dict ↔ Pydantic model) ─────────────────

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        # Also store in parent's in-memory index for ContextTracker lookups
        await super().create_conversation(conversation)

        result = await self._backend.create_conversation(
            contact_id=conversation.contact_id,
            channel=conversation.active_channel.value if conversation.active_channel else "",
            process_type=conversation.business_context.get("process_type", ""),
            entity_type=conversation.business_context.get("entity_type", ""),
            entity_id=conversation.business_context.get("entity_id", ""),
            business_context=conversation.business_context,
            followup_id=conversation.business_context.get("followup_id", ""),
        )
        # Update the conversation ID from the backend if it generated one
        if result and "id" in result:
            conversation.id = result["id"]
        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        # Try in-memory first (fast path for active conversations)
        conv = await super().get_conversation(conversation_id)
        if conv:
            return conv
        # Fall back to backend
        data = await self._backend.get_conversation(conversation_id)
        return self._dict_to_conversation(data) if data else None

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        await super().update_conversation(conversation)
        await self._backend.update_conversation(
            conversation.id,
            status=conversation.status.value if hasattr(conversation.status, "value") else conversation.status,
            summary=conversation.summary or "",
            outcome=conversation.outcome or "",
        )
        return conversation

    async def get_active_conversation(self, contact_id: str, business_context_key: str = None) -> Optional[Conversation]:
        # Use in-memory index (fast, maintained by parent)
        return await super().get_active_conversation(contact_id, business_context_key)

    # ── Messages ──────────────────────────────────────────────

    async def add_message(
        self, conversation_id: str, message: Message,
    ) -> Message:
        """Store a message via backend. Not in the original ContextStore but useful."""
        await self._backend.add_message(
            conversation_id=conversation_id,
            direction=message.direction.value if hasattr(message.direction, "value") else message.direction,
            channel=message.channel.value if hasattr(message.channel, "value") else message.channel,
            content=message.content,
            metadata=message.metadata_ if hasattr(message, "metadata_") else {},
        )
        return message

    # ── Follow-ups (Pydantic ↔ dict) ──────────────────────────

    async def create_followup(self, followup: FollowUp) -> FollowUp:
        # Store in parent's in-memory
        await super().create_followup(followup)
        # Persist to backend
        data = {
            "contact_id": followup.contact_id,
            "external_id": getattr(followup, "external_id", None),
            "reason": followup.reason,
            "priority": followup.priority.value if hasattr(followup.priority, "value") else str(followup.priority),
            "process_type": getattr(followup, "process_type", "") or "",
            "entity_type": getattr(followup, "entity_type", "") or "",
            "entity_id": getattr(followup, "entity_id", "") or "",
            "business_context": followup.business_context or {},
            "channel_priority": [
                c.value if hasattr(c, "value") else str(c)
                for c in (followup.channel_priority or [])
            ],
        }
        result = await self._backend.create_followup(data)
        if result and "id" in result:
            followup.id = result["id"]
        return followup

    async def get_followup(self, followup_id: str) -> Optional[FollowUp]:
        fup = await super().get_followup(followup_id)
        if fup:
            return fup
        data = await self._backend.get_followup(followup_id)
        return self._dict_to_followup(data) if data else None

    async def update_followup(self, followup: FollowUp) -> FollowUp:
        await super().update_followup(followup)
        await self._backend.update_followup(
            followup.id,
            status=followup.status.value if hasattr(followup.status, "value") else followup.status,
            attempt_count=followup.attempt_count,
            outcome=followup.outcome or "",
        )
        return followup

    async def get_pending_followups(self) -> list[FollowUp]:
        # Use in-memory for speed
        return await super().get_pending_followups()

    # ── State Bindings ────────────────────────────────────────

    async def create_state_binding(self, binding: Any) -> Any:
        await super().create_state_binding(binding)
        await self._backend.upsert_state_binding(
            conversation_id=binding.conversation_id,
            process_type=binding.process_type,
            entity_type=binding.entity_type or "",
            entity_id=binding.entity_id or "",
            current_state=binding.current_state,
            state_data=binding.state_data or {},
        )
        return binding

    async def update_state_binding(self, binding: Any) -> Any:
        await super().update_state_binding(binding)
        await self._backend.upsert_state_binding(
            conversation_id=binding.conversation_id,
            process_type=binding.process_type,
            current_state=binding.current_state,
            previous_state=binding.previous_state or "",
            state_data=binding.state_data or {},
        )
        return binding

    # ── Voice Calls ───────────────────────────────────────────

    async def save_voice_call(self, call_data: dict[str, Any]) -> None:
        await self._backend.save_voice_call(call_data)

    # ── Dict → Pydantic converters ────────────────────────────

    @staticmethod
    def _dict_to_conversation(data: dict) -> Conversation:
        status = data.get("status", "active")
        try:
            status_enum = ConversationStatus(status)
        except ValueError:
            status_enum = ConversationStatus.ACTIVE

        channel = data.get("active_channel", "")
        try:
            channel_enum = ChannelType(channel) if channel else None
        except ValueError:
            channel_enum = None

        return Conversation(
            id=data["id"],
            contact_id=data["contact_id"],
            status=status_enum,
            active_channel=channel_enum,
            business_context=data.get("business_context", {}),
            summary=data.get("summary", ""),
            outcome=data.get("outcome", ""),
        )

    @staticmethod
    def _dict_to_followup(data: dict) -> FollowUp:
        status = data.get("status", "pending")
        try:
            status_enum = FollowUpStatus(status)
        except ValueError:
            status_enum = FollowUpStatus.SCHEDULED

        priority = data.get("priority", "medium")

        return FollowUp(
            id=data["id"],
            contact_id=data["contact_id"],
            reason=data.get("reason", ""),
            priority=priority,
            status=status_enum,
            process_type=data.get("process_type"),
            entity_type=data.get("entity_type"),
            entity_id=data.get("entity_id"),
            business_context=data.get("business_context", {}),
            attempt_count=data.get("attempt_count", 0),
            max_attempts=data.get("max_attempts", 3),
            outcome=data.get("outcome", ""),
        )

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        base_stats = {
            "backend": self._backend_name,
            "contacts_in_memory": len(self._contacts),
            "conversations_in_memory": len(self._conversations),
            "followups_in_memory": len(self._followups),
        }
        if hasattr(self._backend, "stats"):
            base_stats["backend_stats"] = self._backend.stats()
        return base_stats


# ══════════════════════════════════════════════════════════════
#  Factory function — reads config and returns the right store
# ══════════════════════════════════════════════════════════════

def create_configured_store() -> ContextStore:
    """
    Create a ContextStore backed by the configured storage backend.

    Reads database.store_backend from settings:
      - "memory"  → InMemoryContextStore (no deps, data lost on restart)
      - "file"    → FileContextStore (JSON files, survives restarts)
      - "sql"     → SqlContextStore (PostgreSQL, MySQL, or SQLite)

    Returns a StoreAdapter that wraps the backend in the ContextStore interface.
    """
    settings = get_settings()
    config = {
        "store_backend": settings.database.store_backend,
        "store_file_dir": settings.database.store_file_dir,
        "url": settings.database.url,
    }

    backend = create_store(config)
    adapter = StoreAdapter(backend)

    logger.info("configured_store_created",
                store_backend=settings.database.store_backend,
                db_url=settings.database.url.split("@")[-1] if "@" in settings.database.url else settings.database.url)

    return adapter


def create_plain_store() -> ContextStore:
    """
    Create a plain in-memory ContextStore (no backend delegation).
    For when you want the original zero-dependency behavior.
    """
    return ContextStore()
