"""
SQLAlchemy ORM models — Cross-database compatible.

Supports: PostgreSQL, MySQL 8+, SQLite.

Key design decisions:
  - JSON type instead of PostgreSQL-specific JSONB — on PG the dialect maps
    JSON to jsonb automatically; on MySQL it uses native JSON; on SQLite
    it serializes to TEXT.
  - No PostgreSQL partial indexes or GIN indexes.
  - String primary keys (uuid hex) — no database-specific sequences.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    String, Integer, Float, DateTime, Text, ForeignKey,
    Index, JSON,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# ──────────────────────────────────────────────────────────────
#  Contacts
# ──────────────────────────────────────────────────────────────

class ContactRow(Base):
    __tablename__ = "contacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(128), default="")
    organization: Mapped[str] = mapped_column(String(256), default="")
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata")
    language: Mapped[str] = mapped_column(String(16), default="en")

    channels: Mapped[Any] = mapped_column(JSON, default=list)
    metadata_: Mapped[Any] = mapped_column("metadata", JSON, default=dict)
    tags: Mapped[Any] = mapped_column(JSON, default=list)

    external_id: Mapped[Optional[str]] = mapped_column(String(256), index=True, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    conversations: Mapped[list["ConversationRow"]] = relationship(back_populates="contact", lazy="selectin")

    __table_args__ = (
        Index("ix_contacts_org", "organization"),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "role": self.role,
            "organization": self.organization, "timezone": self.timezone,
            "language": self.language, "channels": self.channels,
            "metadata": self.metadata_, "tags": self.tags,
            "external_id": self.external_id,
        }


# ──────────────────────────────────────────────────────────────
#  Conversations
# ──────────────────────────────────────────────────────────────

class ConversationRow(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    contact_id: Mapped[str] = mapped_column(String(64), ForeignKey("contacts.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active")
    active_channel: Mapped[str] = mapped_column(String(32), default="")
    followup_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("followups.id"), nullable=True)

    process_type: Mapped[str] = mapped_column(String(64), default="")
    entity_type: Mapped[str] = mapped_column(String(64), default="")
    entity_id: Mapped[str] = mapped_column(String(128), default="")
    business_context: Mapped[Any] = mapped_column(JSON, default=dict)

    summary: Mapped[str] = mapped_column(Text, default="")
    outcome: Mapped[str] = mapped_column(String(128), default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    contact: Mapped["ContactRow"] = relationship(back_populates="conversations", lazy="selectin")
    messages: Mapped[list["MessageRow"]] = relationship(back_populates="conversation", lazy="selectin", order_by="MessageRow.timestamp")
    state_bindings: Mapped[list["StateBindingRow"]] = relationship(back_populates="conversation", lazy="selectin")

    __table_args__ = (
        Index("ix_conversations_contact", "contact_id"),
        Index("ix_conversations_status", "status"),
        Index("ix_conversations_followup", "followup_id"),
    )


# ──────────────────────────────────────────────────────────────
#  Messages
# ──────────────────────────────────────────────────────────────

class MessageRow(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversations.id"), nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    metadata_: Mapped[Any] = mapped_column("metadata", JSON, default=dict)

    channel_message_id: Mapped[str] = mapped_column(String(256), default="")
    delivery_status: Mapped[str] = mapped_column(String(32), default="sent")

    conversation: Mapped["ConversationRow"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_messages_conversation_ts", "conversation_id", "timestamp"),
        Index("ix_messages_channel_msg_id", "channel_message_id"),
    )


# ──────────────────────────────────────────────────────────────
#  Follow-ups
# ──────────────────────────────────────────────────────────────

class FollowUpRow(Base):
    __tablename__ = "followups"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    contact_id: Mapped[str] = mapped_column(String(64), ForeignKey("contacts.id"), nullable=False)
    external_id: Mapped[Optional[str]] = mapped_column(String(256), index=True, nullable=True)

    reason: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    status: Mapped[str] = mapped_column(String(32), default="pending")

    process_type: Mapped[str] = mapped_column(String(64), default="")
    entity_type: Mapped[str] = mapped_column(String(64), default="")
    entity_id: Mapped[str] = mapped_column(String(128), default="")
    business_context: Mapped[Any] = mapped_column(JSON, default=dict)

    channel_priority: Mapped[Any] = mapped_column(JSON, default=lambda: ["voice", "whatsapp", "email"])
    current_channel: Mapped[str] = mapped_column(String(32), default="")

    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    outcome: Mapped[str] = mapped_column(Text, default="")
    outcome_data: Mapped[Any] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_followups_status_next", "status", "next_attempt_at"),
        Index("ix_followups_contact", "contact_id"),
        Index("ix_followups_external", "external_id"),
    )


# ──────────────────────────────────────────────────────────────
#  State Bindings
# ──────────────────────────────────────────────────────────────

class StateBindingRow(Base):
    __tablename__ = "state_bindings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_id)
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversations.id"), nullable=False)
    process_type: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), default="")
    entity_id: Mapped[str] = mapped_column(String(128), default="")
    current_state: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_state: Mapped[str] = mapped_column(String(64), default="")

    state_data: Mapped[Any] = mapped_column(JSON, default=dict)
    transition_history: Mapped[Any] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    conversation: Mapped["ConversationRow"] = relationship(back_populates="state_bindings")

    __table_args__ = (
        Index("ix_state_bindings_conv", "conversation_id"),
        Index("ix_state_bindings_entity", "entity_type", "entity_id"),
    )


# ──────────────────────────────────────────────────────────────
#  Voice Call Records
# ──────────────────────────────────────────────────────────────

class VoiceCallRow(Base):
    __tablename__ = "voice_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversations.id"), nullable=False)
    contact_id: Mapped[str] = mapped_column(String(64), ForeignKey("contacts.id"), nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)

    sip_call_id: Mapped[str] = mapped_column(String(256), default="")
    from_number: Mapped[str] = mapped_column(String(32), default="")
    to_number: Mapped[str] = mapped_column(String(32), default="")
    telephony_provider: Mapped[str] = mapped_column(String(32), default="twilio")

    status: Mapped[str] = mapped_column(String(32), default="initiating")
    disposition: Mapped[str] = mapped_column(String(32), default="")
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    transcript: Mapped[Any] = mapped_column(JSON, default=list)

    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    barge_in_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    latency_breakdown: Mapped[Any] = mapped_column(JSON, default=dict)

    recording_url: Mapped[str] = mapped_column(String(512), default="")

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    connected_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_voice_calls_conversation", "conversation_id"),
        Index("ix_voice_calls_contact", "contact_id"),
        Index("ix_voice_calls_started", "started_at"),
    )
