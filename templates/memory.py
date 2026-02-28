"""
Flow Memory — Cross-execution knowledge accumulation.

Each dialogue flow execution can read and write to a shared memory
space scoped to a contact, conversation, or entity. This allows:

  - A payment reminder flow to know that 3 previous reminders were sent
  - A feedback collection flow to remember the contact prefers WhatsApp
  - An escalation flow to recall all previous issues for this entity
  - The planner to learn from past interaction patterns

Memory entries are typed (fact, preference, event, summary) and
carry confidence scores so the planner can weight them appropriately.

Memory is NOT a raw conversation log. It's structured, distilled
knowledge — the kind of thing a human agent would keep in their
mental model of a customer.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class MemoryType(str, Enum):
    """What kind of knowledge this memory represents."""
    FACT = "fact"                 # Concrete data: "contact paid ₹5000 on Jan 15"
    PREFERENCE = "preference"    # Behavioral: "contact prefers WhatsApp", "responds mornings"
    EVENT = "event"              # Something that happened: "escalation triggered", "complaint filed"
    SUMMARY = "summary"          # Distilled conversation summary
    INTENT = "intent"            # Detected intent: "contact intends to pay by Friday"
    ENTITY = "entity"            # Extracted entity: "reference number INV-2024-1234"
    OBSERVATION = "observation"  # Agent observation: "contact tone shifted from hostile to cooperative"


class MemoryEntry(BaseModel):
    """A single piece of accumulated knowledge."""
    id: str = Field(default_factory=lambda: f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}")
    type: MemoryType
    key: str                                           # Lookup key: "payment_status", "preferred_channel"
    value: Any                                         # The knowledge itself
    confidence: float = 1.0                            # 0.0–1.0, how certain we are
    source_flow_id: str = ""                           # Which flow produced this
    source_step_id: str = ""                           # Which step in the flow
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None              # Auto-expire old memories
    tags: list[str] = []                               # For filtering

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) >= self.expires_at


class MemoryScope(BaseModel):
    """A scoped collection of memories."""
    scope_type: str                                    # "contact", "conversation", "entity"
    scope_id: str                                      # The id of the scoped object
    entries: dict[str, MemoryEntry] = {}               # key → latest entry
    history: list[MemoryEntry] = []                    # All entries, chronological

    def remember(self, entry: MemoryEntry):
        """Add or update a memory entry."""
        # If same key exists with lower confidence, replace
        existing = self.entries.get(entry.key)
        if existing and existing.confidence > entry.confidence:
            # Keep higher-confidence version but log the new one in history
            self.history.append(entry)
            return

        self.entries[entry.key] = entry
        self.history.append(entry)

    def recall(self, key: str) -> Optional[MemoryEntry]:
        """Recall a specific memory by key."""
        entry = self.entries.get(key)
        if entry and entry.is_expired:
            del self.entries[key]
            return None
        return entry

    def recall_value(self, key: str, default: Any = None) -> Any:
        """Recall just the value of a memory."""
        entry = self.recall(key)
        return entry.value if entry else default

    def recall_by_type(self, mem_type: MemoryType) -> list[MemoryEntry]:
        """Get all active memories of a given type."""
        return [
            e for e in self.entries.values()
            if e.type == mem_type and not e.is_expired
        ]

    def recall_by_tag(self, tag: str) -> list[MemoryEntry]:
        """Get all active memories with a given tag."""
        return [
            e for e in self.entries.values()
            if tag in e.tags and not e.is_expired
        ]

    def recall_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get the N most recent non-expired entries."""
        active = [e for e in self.history if not e.is_expired]
        return active[-n:]

    def to_context_dict(self) -> dict[str, Any]:
        """Export active memories as a flat dict for template interpolation."""
        result = {}
        for key, entry in self.entries.items():
            if not entry.is_expired:
                result[f"memory.{key}"] = entry.value
        return result

    def describe_for_llm(self, max_entries: int = 15) -> str:
        """Build a description of accumulated knowledge for the planner."""
        active = [e for e in self.entries.values() if not e.is_expired]
        if not active:
            return "No prior knowledge about this contact/conversation."

        # Sort by recency
        active.sort(key=lambda e: e.created_at, reverse=True)
        active = active[:max_entries]

        lines = ["Known information:"]
        for entry in active:
            conf = f" (confidence: {entry.confidence:.0%})" if entry.confidence < 1.0 else ""
            lines.append(f"  • [{entry.type.value}] {entry.key}: {entry.value}{conf}")
        return "\n".join(lines)

    def clean_expired(self):
        """Remove expired entries."""
        expired_keys = [k for k, e in self.entries.items() if e.is_expired]
        for k in expired_keys:
            del self.entries[k]


class FlowMemoryStore:
    """
    In-memory store for flow memories, scoped by contact/conversation/entity.

    In production, back this with Redis or a database.
    """

    def __init__(self):
        self._scopes: dict[str, MemoryScope] = {}

    def _scope_key(self, scope_type: str, scope_id: str) -> str:
        return f"{scope_type}:{scope_id}"

    def get_scope(self, scope_type: str, scope_id: str) -> MemoryScope:
        """Get or create a memory scope."""
        key = self._scope_key(scope_type, scope_id)
        if key not in self._scopes:
            self._scopes[key] = MemoryScope(
                scope_type=scope_type,
                scope_id=scope_id,
            )
        return self._scopes[key]

    def remember(
        self,
        scope_type: str,
        scope_id: str,
        key: str,
        value: Any,
        mem_type: MemoryType = MemoryType.FACT,
        confidence: float = 1.0,
        source_flow_id: str = "",
        source_step_id: str = "",
        tags: list[str] = None,
        expires_at: datetime = None,
    ) -> MemoryEntry:
        """Store a memory in the given scope."""
        scope = self.get_scope(scope_type, scope_id)
        entry = MemoryEntry(
            type=mem_type,
            key=key,
            value=value,
            confidence=confidence,
            source_flow_id=source_flow_id,
            source_step_id=source_step_id,
            tags=tags or [],
            expires_at=expires_at,
        )
        scope.remember(entry)
        logger.debug("memory_stored",
                      scope=f"{scope_type}:{scope_id}",
                      key=key, type=mem_type.value)
        return entry

    def recall(
        self,
        scope_type: str,
        scope_id: str,
        key: str,
    ) -> Optional[MemoryEntry]:
        """Recall a specific memory."""
        scope = self.get_scope(scope_type, scope_id)
        return scope.recall(key)

    def recall_value(
        self,
        scope_type: str,
        scope_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Recall just the value."""
        entry = self.recall(scope_type, scope_id, key)
        return entry.value if entry else default

    def build_context_overlay(
        self,
        contact_id: str = "",
        conversation_id: str = "",
        entity_type: str = "",
        entity_id: str = "",
    ) -> dict[str, Any]:
        """
        Build a merged memory overlay for template interpolation.
        Merges entity → contact → conversation scopes (narrowest wins).
        """
        overlay = {}

        # Broadest scope first
        if entity_type and entity_id:
            scope = self.get_scope("entity", f"{entity_type}:{entity_id}")
            overlay.update(scope.to_context_dict())

        if contact_id:
            scope = self.get_scope("contact", contact_id)
            overlay.update(scope.to_context_dict())

        if conversation_id:
            scope = self.get_scope("conversation", conversation_id)
            overlay.update(scope.to_context_dict())

        return overlay

    def describe_for_llm(
        self,
        contact_id: str = "",
        conversation_id: str = "",
        entity_type: str = "",
        entity_id: str = "",
        max_entries: int = 20,
    ) -> str:
        """Build merged knowledge description for the planner."""
        sections = []

        if entity_type and entity_id:
            scope = self.get_scope("entity", f"{entity_type}:{entity_id}")
            desc = scope.describe_for_llm(max_entries // 3)
            if "No prior" not in desc:
                sections.append(f"About {entity_type} {entity_id}:\n{desc}")

        if contact_id:
            scope = self.get_scope("contact", contact_id)
            desc = scope.describe_for_llm(max_entries // 3)
            if "No prior" not in desc:
                sections.append(f"About this contact:\n{desc}")

        if conversation_id:
            scope = self.get_scope("conversation", conversation_id)
            desc = scope.describe_for_llm(max_entries // 3)
            if "No prior" not in desc:
                sections.append(f"This conversation:\n{desc}")

        return "\n\n".join(sections) if sections else "No accumulated knowledge yet."

    @property
    def total_entries(self) -> int:
        return sum(len(s.entries) for s in self._scopes.values())

    def get_stats(self) -> dict[str, Any]:
        return {
            "scopes": len(self._scopes),
            "total_entries": self.total_entries,
            "by_scope_type": {
                stype: sum(1 for k in self._scopes if k.startswith(f"{stype}:"))
                for stype in ["contact", "conversation", "entity"]
            },
        }
