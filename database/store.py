"""
SqlContextStore — Portable SQL queries for PostgreSQL, MySQL, SQLite.

Replaces PG-specific functions:
  - func.array_position() → case() expression for priority sorting
  - JSONB .contains()     → Python-side channel address filtering
"""
from __future__ import annotations

import json
import structlog
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    ContactRow, ConversationRow, MessageRow, FollowUpRow,
    StateBindingRow, VoiceCallRow,
)
from database.session import get_session
from database.store_base import BaseContextStore
from models.schemas import (
    Contact, ContactChannel, ChannelType,
)

logger = structlog.get_logger()

# Priority ordering — used in CASE expression instead of PG array_position
_PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class SqlContextStore(BaseContextStore):
    """
    Persistent context store backed by any SQLAlchemy-supported database.
    Works with PostgreSQL, MySQL 8+, and SQLite.
    """

    # ── Contact operations ─────────────────────────────────

    async def get_contact(self, contact_id: str) -> Optional[Contact]:
        async with get_session() as db:
            row = await db.get(ContactRow, contact_id)
            return self._row_to_contact(row) if row else None

    async def get_contact_by_external_id(self, external_id: str) -> Optional[Contact]:
        async with get_session() as db:
            stmt = select(ContactRow).where(ContactRow.external_id == external_id)
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            return self._row_to_contact(row) if row else None

    async def find_contact_by_address(self, channel: str, address: str) -> Optional[Contact]:
        """
        Find contact by channel address. Uses Python-side filtering for
        cross-database compatibility (JSON column search is not portable).
        For large contact tables, consider adding a separate address lookup table.
        """
        async with get_session() as db:
            # Load contacts in batches and filter in Python
            # This is portable across all databases
            stmt = select(ContactRow)
            result = await db.execute(stmt)
            for row in result.scalars():
                channels = row.channels or []
                # Handle both dict and string (SQLite stores JSON as text)
                if isinstance(channels, str):
                    channels = json.loads(channels)
                for ch in channels:
                    if ch.get("channel") == channel and ch.get("address") == address:
                        return self._row_to_contact(row)
        return None

    async def upsert_contact(self, contact: Contact) -> Contact:
        async with get_session() as db:
            existing = await db.get(ContactRow, contact.id)
            channel_dicts = [ch.model_dump(mode="json") for ch in contact.channels]
            if existing:
                existing.name = contact.name
                existing.role = contact.role
                existing.organization = contact.organization
                existing.channels = channel_dicts
                existing.metadata_ = contact.metadata_ if hasattr(contact, "metadata_") else {}
            else:
                row = ContactRow(
                    id=contact.id,
                    name=contact.name,
                    role=contact.role,
                    organization=contact.organization,
                    channels=channel_dicts,
                    metadata_=contact.metadata_ if hasattr(contact, "metadata_") else {},
                )
                db.add(row)
            return contact

    # ── Conversation operations ────────────────────────────

    async def get_conversation(self, conversation_id: str) -> Optional[dict[str, Any]]:
        async with get_session() as db:
            row = await db.get(ConversationRow, conversation_id)
            return self._conversation_to_dict(row) if row else None

    async def find_active_conversation(self, contact_id: str) -> Optional[dict[str, Any]]:
        async with get_session() as db:
            stmt = (
                select(ConversationRow)
                .where(and_(
                    ConversationRow.contact_id == contact_id,
                    ConversationRow.status == "active",
                ))
                .order_by(ConversationRow.updated_at.desc())
                .limit(1)
            )
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            return self._conversation_to_dict(row) if row else None

    async def create_conversation(
        self, contact_id: str, channel: str = "",
        process_type: str = "", entity_type: str = "",
        entity_id: str = "", business_context: dict = None,
        followup_id: str = "",
    ) -> dict[str, Any]:
        async with get_session() as db:
            row = ConversationRow(
                contact_id=contact_id,
                active_channel=channel,
                process_type=process_type,
                entity_type=entity_type,
                entity_id=entity_id,
                business_context=business_context or {},
                followup_id=followup_id or None,
            )
            db.add(row)
            await db.flush()
            return self._conversation_to_dict(row)

    async def update_conversation(self, conversation_id: str, **kwargs) -> None:
        async with get_session() as db:
            stmt = (
                update(ConversationRow)
                .where(ConversationRow.id == conversation_id)
                .values(**kwargs, updated_at=datetime.now(timezone.utc))
            )
            await db.execute(stmt)

    # ── Message operations ─────────────────────────────────

    async def add_message(
        self, conversation_id: str, direction: str, channel: str,
        content: str, metadata: dict = None, channel_message_id: str = "",
    ) -> dict[str, Any]:
        async with get_session() as db:
            row = MessageRow(
                conversation_id=conversation_id,
                direction=direction,
                channel=channel,
                content=content,
                metadata_=metadata or {},
                channel_message_id=channel_message_id,
            )
            db.add(row)
            await db.flush()

            await db.execute(
                update(ConversationRow)
                .where(ConversationRow.id == conversation_id)
                .values(updated_at=datetime.now(timezone.utc))
            )

            return {
                "id": row.id, "conversation_id": conversation_id,
                "direction": direction, "channel": channel,
                "content": content, "timestamp": row.timestamp.isoformat(),
                "metadata": row.metadata_,
            }

    async def get_conversation_messages(
        self, conversation_id: str, limit: int = 50,
    ) -> list[dict[str, Any]]:
        async with get_session() as db:
            stmt = (
                select(MessageRow)
                .where(MessageRow.conversation_id == conversation_id)
                .order_by(MessageRow.timestamp.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            rows = result.scalars().all()
            return [
                {
                    "id": r.id, "direction": r.direction, "channel": r.channel,
                    "content": r.content, "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata_,
                }
                for r in reversed(rows)
            ]

    # ── Follow-up operations ───────────────────────────────

    async def get_followup(self, followup_id: str) -> Optional[dict[str, Any]]:
        async with get_session() as db:
            row = await db.get(FollowUpRow, followup_id)
            return self._followup_to_dict(row) if row else None

    async def get_followup_by_external_id(self, external_id: str) -> Optional[dict[str, Any]]:
        async with get_session() as db:
            stmt = select(FollowUpRow).where(FollowUpRow.external_id == external_id)
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            return self._followup_to_dict(row) if row else None

    async def create_followup(self, data: dict[str, Any]) -> dict[str, Any]:
        async with get_session() as db:
            row = FollowUpRow(
                contact_id=data["contact_id"],
                external_id=data.get("external_id"),
                reason=data.get("reason", ""),
                priority=data.get("priority", "medium"),
                process_type=data.get("process_type", ""),
                entity_type=data.get("entity_type", ""),
                entity_id=data.get("entity_id", ""),
                business_context=data.get("business_context", {}),
                channel_priority=data.get("channel_priority", ["voice", "whatsapp", "email"]),
            )
            db.add(row)
            await db.flush()
            return self._followup_to_dict(row)

    async def update_followup(self, followup_id: str, **kwargs) -> None:
        async with get_session() as db:
            stmt = (
                update(FollowUpRow)
                .where(FollowUpRow.id == followup_id)
                .values(**kwargs, updated_at=datetime.now(timezone.utc))
            )
            await db.execute(stmt)

    async def get_pending_followups(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get pending follow-ups ordered by priority.
        Uses CASE expression instead of PG-specific array_position().
        """
        async with get_session() as db:
            priority_order = case(
                _PRIORITY_ORDER,
                value=FollowUpRow.priority,
                else_=99,
            )
            stmt = (
                select(FollowUpRow)
                .where(and_(
                    FollowUpRow.status.in_(["pending", "dispatched"]),
                    FollowUpRow.attempt_count < FollowUpRow.max_attempts,
                ))
                .order_by(priority_order, FollowUpRow.created_at)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return [self._followup_to_dict(r) for r in result.scalars().all()]

    # ── State binding operations ───────────────────────────

    async def get_state_binding(
        self, conversation_id: str, process_type: str,
    ) -> Optional[dict[str, Any]]:
        async with get_session() as db:
            stmt = select(StateBindingRow).where(and_(
                StateBindingRow.conversation_id == conversation_id,
                StateBindingRow.process_type == process_type,
            ))
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()
            if not row:
                return None
            return {
                "id": row.id, "conversation_id": row.conversation_id,
                "process_type": row.process_type, "entity_type": row.entity_type,
                "entity_id": row.entity_id, "current_state": row.current_state,
                "previous_state": row.previous_state, "state_data": row.state_data,
                "transition_history": row.transition_history,
            }

    async def upsert_state_binding(
        self, conversation_id: str, process_type: str, **kwargs,
    ) -> None:
        async with get_session() as db:
            stmt = select(StateBindingRow).where(and_(
                StateBindingRow.conversation_id == conversation_id,
                StateBindingRow.process_type == process_type,
            ))
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()

            if row:
                for k, v in kwargs.items():
                    setattr(row, k, v)
            else:
                row = StateBindingRow(
                    conversation_id=conversation_id,
                    process_type=process_type,
                    **kwargs,
                )
                db.add(row)

    # ── Voice call records ─────────────────────────────────

    async def save_voice_call(self, call_data: dict[str, Any]) -> None:
        async with get_session() as db:
            existing = await db.get(VoiceCallRow, call_data["id"])
            if existing:
                for k, v in call_data.items():
                    if k != "id":
                        setattr(existing, k, v)
            else:
                row = VoiceCallRow(**call_data)
                db.add(row)

    # ── Converters ─────────────────────────────────────────

    @staticmethod
    def _row_to_contact(row: ContactRow) -> Contact:
        raw_channels = row.channels or []
        if isinstance(raw_channels, str):
            raw_channels = json.loads(raw_channels)
        channels = [
            ContactChannel(
                channel=ChannelType(ch["channel"]),
                address=ch["address"],
                preferred=ch.get("preferred", False),
                verified=ch.get("verified", True),
            )
            for ch in raw_channels
        ]
        return Contact(
            id=row.id, name=row.name, role=row.role,
            organization=row.organization, channels=channels,
        )

    @staticmethod
    def _conversation_to_dict(row: ConversationRow) -> dict[str, Any]:
        return {
            "id": row.id, "contact_id": row.contact_id,
            "status": row.status, "active_channel": row.active_channel,
            "process_type": row.process_type, "entity_type": row.entity_type,
            "entity_id": row.entity_id, "business_context": row.business_context,
            "summary": row.summary, "outcome": row.outcome,
            "followup_id": row.followup_id,
            "created_at": row.created_at.isoformat() if row.created_at else "",
            "updated_at": row.updated_at.isoformat() if row.updated_at else "",
        }

    @staticmethod
    def _followup_to_dict(row: FollowUpRow) -> dict[str, Any]:
        return {
            "id": row.id, "contact_id": row.contact_id,
            "external_id": row.external_id, "reason": row.reason,
            "priority": row.priority, "status": row.status,
            "process_type": row.process_type, "entity_type": row.entity_type,
            "entity_id": row.entity_id, "business_context": row.business_context,
            "channel_priority": row.channel_priority,
            "current_channel": row.current_channel,
            "attempt_count": row.attempt_count,
            "max_attempts": row.max_attempts,
            "outcome": row.outcome, "outcome_data": row.outcome_data,
            "created_at": row.created_at.isoformat() if row.created_at else "",
        }


# Backwards compatibility alias
PostgresContextStore = SqlContextStore
