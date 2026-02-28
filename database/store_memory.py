"""
InMemoryContextStore — Dict-backed store for development and testing.

Features:
  - Zero dependencies (no database, no Redis)
  - Full interface compatibility with SqlContextStore
  - Thread-safe via asyncio (single event loop)
  - All data lost on process restart

Best for: local development, unit tests, quick prototyping.
"""
from __future__ import annotations

import uuid
import structlog
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from database.store_base import BaseContextStore
from models.schemas import Contact, ContactChannel, ChannelType

logger = structlog.get_logger()

_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class InMemoryContextStore(BaseContextStore):
    """
    Full-featured in-memory store with the same interface as SqlContextStore.
    Returns dicts for conversations/followups/messages (not ORM rows).
    """

    def __init__(self):
        self._contacts: dict[str, dict] = {}           # id → contact dict
        self._conversations: dict[str, dict] = {}      # id → conversation dict
        self._messages: dict[str, list[dict]] = defaultdict(list)  # conv_id → [msg dicts]
        self._followups: dict[str, dict] = {}           # id → followup dict
        self._state_bindings: dict[str, dict] = {}      # "conv_id:process_type" → binding dict
        self._voice_calls: dict[str, dict] = {}         # id → call dict

        # Indexes
        self._address_index: dict[str, str] = {}        # "channel:address" → contact_id
        self._external_id_index: dict[str, str] = {}    # external_id → contact_id
        self._followup_ext_index: dict[str, str] = {}   # external_id → followup_id
        logger.info("inmemory_store_initialized")

    # ── Contacts ──────────────────────────────────────────

    async def get_contact(self, contact_id: str) -> Optional[Contact]:
        data = self._contacts.get(contact_id)
        return self._dict_to_contact(data) if data else None

    async def get_contact_by_external_id(self, external_id: str) -> Optional[Contact]:
        cid = self._external_id_index.get(external_id)
        if not cid:
            return None
        return await self.get_contact(cid)

    async def find_contact_by_address(self, channel: str, address: str) -> Optional[Contact]:
        cid = self._address_index.get(f"{channel}:{address}")
        if not cid:
            return None
        return await self.get_contact(cid)

    async def upsert_contact(self, contact: Contact) -> Contact:
        channel_dicts = [ch.model_dump(mode="json") for ch in contact.channels]
        data = {
            "id": contact.id, "name": contact.name, "role": contact.role,
            "organization": contact.organization,
            "channels": channel_dicts,
            "metadata": contact.metadata_ if hasattr(contact, "metadata_") else {},
            "external_id": getattr(contact, "external_id", None),
        }
        self._contacts[contact.id] = data
        # Update indexes
        for ch in channel_dicts:
            self._address_index[f"{ch['channel']}:{ch['address']}"] = contact.id
        if data.get("external_id"):
            self._external_id_index[data["external_id"]] = contact.id
        return contact

    # ── Conversations ─────────────────────────────────────

    async def get_conversation(self, conversation_id: str) -> Optional[dict[str, Any]]:
        return self._conversations.get(conversation_id)

    async def find_active_conversation(self, contact_id: str) -> Optional[dict[str, Any]]:
        # Find most recently updated active conversation for this contact
        active = [
            c for c in self._conversations.values()
            if c["contact_id"] == contact_id and c["status"] == "active"
        ]
        if not active:
            return None
        active.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return active[0]

    async def create_conversation(
        self, contact_id: str, channel: str = "",
        process_type: str = "", entity_type: str = "",
        entity_id: str = "", business_context: dict = None,
        followup_id: str = "",
    ) -> dict[str, Any]:
        now = _utcnow().isoformat()
        conv = {
            "id": _new_id(), "contact_id": contact_id,
            "status": "active", "active_channel": channel,
            "process_type": process_type, "entity_type": entity_type,
            "entity_id": entity_id, "business_context": business_context or {},
            "summary": "", "outcome": "",
            "followup_id": followup_id or None,
            "created_at": now, "updated_at": now,
        }
        self._conversations[conv["id"]] = conv
        return conv

    async def update_conversation(self, conversation_id: str, **kwargs) -> None:
        conv = self._conversations.get(conversation_id)
        if conv:
            conv.update(kwargs)
            conv["updated_at"] = _utcnow().isoformat()

    # ── Messages ──────────────────────────────────────────

    async def add_message(
        self, conversation_id: str, direction: str, channel: str,
        content: str, metadata: dict = None, channel_message_id: str = "",
    ) -> dict[str, Any]:
        now = _utcnow()
        msg = {
            "id": _new_id(), "conversation_id": conversation_id,
            "direction": direction, "channel": channel,
            "content": content, "timestamp": now.isoformat(),
            "metadata": metadata or {},
            "channel_message_id": channel_message_id,
        }
        self._messages[conversation_id].append(msg)
        # Touch conversation
        conv = self._conversations.get(conversation_id)
        if conv:
            conv["updated_at"] = now.isoformat()
        return msg

    async def get_conversation_messages(
        self, conversation_id: str, limit: int = 50,
    ) -> list[dict[str, Any]]:
        msgs = self._messages.get(conversation_id, [])
        # Return last N messages in chronological order
        return msgs[-limit:]

    # ── Follow-ups ────────────────────────────────────────

    async def get_followup(self, followup_id: str) -> Optional[dict[str, Any]]:
        return self._followups.get(followup_id)

    async def get_followup_by_external_id(self, external_id: str) -> Optional[dict[str, Any]]:
        fid = self._followup_ext_index.get(external_id)
        return self._followups.get(fid) if fid else None

    async def create_followup(self, data: dict[str, Any]) -> dict[str, Any]:
        now = _utcnow().isoformat()
        fup = {
            "id": _new_id(),
            "contact_id": data["contact_id"],
            "external_id": data.get("external_id"),
            "reason": data.get("reason", ""),
            "priority": data.get("priority", "medium"),
            "status": "pending",
            "process_type": data.get("process_type", ""),
            "entity_type": data.get("entity_type", ""),
            "entity_id": data.get("entity_id", ""),
            "business_context": data.get("business_context", {}),
            "channel_priority": data.get("channel_priority", ["voice", "whatsapp", "email"]),
            "current_channel": "",
            "attempt_count": 0,
            "max_attempts": data.get("max_attempts", 3),
            "outcome": "", "outcome_data": {},
            "created_at": now,
        }
        self._followups[fup["id"]] = fup
        if fup.get("external_id"):
            self._followup_ext_index[fup["external_id"]] = fup["id"]
        return fup

    async def update_followup(self, followup_id: str, **kwargs) -> None:
        fup = self._followups.get(followup_id)
        if fup:
            fup.update(kwargs)

    async def get_pending_followups(self, limit: int = 100) -> list[dict[str, Any]]:
        pending = [
            f for f in self._followups.values()
            if f["status"] in ("pending", "dispatched")
            and f["attempt_count"] < f["max_attempts"]
        ]
        # Sort by priority then created_at
        pending.sort(key=lambda f: (
            _PRIORITY_RANK.get(f["priority"], 99),
            f.get("created_at", ""),
        ))
        return pending[:limit]

    # ── State Bindings ────────────────────────────────────

    async def get_state_binding(
        self, conversation_id: str, process_type: str,
    ) -> Optional[dict[str, Any]]:
        key = f"{conversation_id}:{process_type}"
        return self._state_bindings.get(key)

    async def upsert_state_binding(
        self, conversation_id: str, process_type: str, **kwargs,
    ) -> None:
        key = f"{conversation_id}:{process_type}"
        existing = self._state_bindings.get(key)
        if existing:
            existing.update(kwargs)
            existing["updated_at"] = _utcnow().isoformat()
        else:
            binding = {
                "id": _new_id(),
                "conversation_id": conversation_id,
                "process_type": process_type,
                **kwargs,
                "created_at": _utcnow().isoformat(),
                "updated_at": _utcnow().isoformat(),
            }
            self._state_bindings[key] = binding

    # ── Voice Calls ───────────────────────────────────────

    async def save_voice_call(self, call_data: dict[str, Any]) -> None:
        self._voice_calls[call_data["id"]] = call_data

    # ── Helpers ───────────────────────────────────────────

    @staticmethod
    def _dict_to_contact(data: dict) -> Contact:
        channels = [
            ContactChannel(
                channel=ChannelType(ch["channel"]),
                address=ch["address"],
                preferred=ch.get("preferred", False),
                verified=ch.get("verified", True),
            )
            for ch in (data.get("channels") or [])
        ]
        return Contact(
            id=data["id"], name=data["name"],
            role=data.get("role", ""),
            organization=data.get("organization", ""),
            channels=channels,
        )

    # ── Stats (for debugging) ─────────────────────────────

    def stats(self) -> dict[str, int]:
        return {
            "contacts": len(self._contacts),
            "conversations": len(self._conversations),
            "messages": sum(len(v) for v in self._messages.values()),
            "followups": len(self._followups),
            "state_bindings": len(self._state_bindings),
            "voice_calls": len(self._voice_calls),
        }
