"""
Chat Channel Adapter — WebSocket-based real-time messaging.

Provides:
- Connection lifecycle with registration and superseding
- Presence tracking (online/offline/away)
- Typing indicators
- Offline message queue with drain-on-reconnect
- Heartbeat handling
- System event broadcasting
- Client event routing (message, typing, ack, heartbeat)
"""
from __future__ import annotations

import json
import time
import uuid
import asyncio
import structlog
from typing import Any, Callable, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import deque

from models.schemas import ChannelType, Contact
from channels.base import ChannelAdapter

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  CONNECTION & QUEUE MODELS
# ══════════════════════════════════════════════════════════════

class ConnectionState:
    """Tracks a single WebSocket connection."""

    def __init__(self, user_id: str, ws: Any):
        self.user_id = user_id
        self.ws = ws
        self.connected_at = datetime.now(timezone.utc)
        self.last_heartbeat = time.monotonic()
        self.is_typing: bool = False
        self.message_count: int = 0


@dataclass
class QueuedMessage:
    """Message queued for delivery when user reconnects."""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ══════════════════════════════════════════════════════════════
#  CHAT ADAPTER
# ══════════════════════════════════════════════════════════════

class ChatAdapter(ChannelAdapter):
    """
    Real-time chat over WebSockets with offline queue.

    Features:
    - Per-user connection tracking
    - Automatic offline queuing + drain on reconnect
    - Typing and presence indicators
    - Heartbeat-based stale connection detection
    - System event broadcasting
    - Connection superseding (new connection replaces old)
    """

    channel_type = ChannelType.CHAT

    def __init__(self):
        super().__init__()
        self._connections: dict[str, ConnectionState] = {}
        self._offline_queues: dict[str, deque[QueuedMessage]] = {}
        self._max_queue_size: int = 100
        self._on_message_callback: Optional[Callable] = None

    async def initialize(self, config: dict[str, Any]) -> None:
        self._config = config
        self._max_queue_size = config.get("max_queue_size", 100)
        self._initialized = True

    def set_message_callback(self, callback: Callable):
        self._on_message_callback = callback

    # ── Connection management ─────────────────────────────────

    async def register_connection(self, user_id: str, ws: Any) -> None:
        """
        Register a WebSocket connection for a user.
        Supersedes any existing connection and drains queued messages.
        """
        # Supersede old connection
        existing = self._connections.get(user_id)
        if existing:
            try:
                await existing.ws.close()
            except Exception:
                pass
            logger.info("connection_superseded", user_id=user_id)

        self._connections[user_id] = ConnectionState(user_id, ws)
        logger.info("connection_registered", user_id=user_id)

        # Drain offline queue
        await self._drain_queue(user_id, ws)

    async def remove_connection(self, user_id: str) -> None:
        self._connections.pop(user_id, None)
        logger.info("connection_removed", user_id=user_id)

    def is_connected(self, user_id: str) -> bool:
        return user_id in self._connections

    def get_presence(self, user_id: str) -> str:
        if user_id not in self._connections:
            return "offline"
        conn = self._connections[user_id]
        # Stale after 90s without heartbeat
        if time.monotonic() - conn.last_heartbeat > 90:
            return "away"
        return "online"

    # ── Send ──────────────────────────────────────────────────

    async def _do_send(
        self, contact: Contact, content: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        user_id = self.get_address(contact)
        if not user_id:
            return {"status": "failed", "error": "No chat address"}

        conn = self._connections.get(user_id)
        if not conn:
            # Queue for when they reconnect
            self._enqueue(user_id, content, metadata)
            return {"status": "queued", "channel_message_id": str(uuid.uuid4())}

        msg_id = str(uuid.uuid4())
        payload = {
            "type": "message",
            "message_id": msg_id,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in metadata.items() if k not in ("message_id",)},
        }

        try:
            await conn.ws.send_text(json.dumps(payload))
            conn.message_count += 1
            return {"status": "delivered", "channel_message_id": msg_id}
        except Exception as e:
            # Connection broken — remove and queue
            self._connections.pop(user_id, None)
            self._enqueue(user_id, content, metadata)
            return {"status": "queued", "channel_message_id": msg_id, "error": str(e)}

    async def _do_send_template(
        self, contact: Contact, template_name: str, template_data: dict[str, Any]
    ) -> dict[str, Any]:
        content = self._render_template(template_name, template_data)
        return await self._do_send(contact, content, {"template": template_name})

    # ── Client event handling ─────────────────────────────────

    async def handle_client_event(
        self, user_id: str, event: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """
        Handle events from a connected client.
        Returns parsed message dict for 'message' events, None otherwise.
        """
        conn = self._connections.get(user_id)
        event_type = event.get("type", "")

        if event_type == "heartbeat":
            if conn:
                conn.last_heartbeat = time.monotonic()
            return None

        elif event_type == "typing":
            if conn:
                conn.is_typing = event.get("is_typing", False)
            return None

        elif event_type == "ack":
            # Client acknowledges receipt of a message
            return None

        elif event_type == "message":
            content = event.get("content", "")
            if not content:
                return None
            return {
                "sender_address": user_id,
                "content": content,
                "metadata": {
                    "channel": "chat",
                    "channel_message_id": event.get("message_id", str(uuid.uuid4())),
                },
            }

        return None

    # ── Typing and system events ──────────────────────────────

    async def send_typing_indicator(self, user_id: str, is_typing: bool = True) -> None:
        conn = self._connections.get(user_id)
        if not conn:
            return
        payload = {
            "type": "typing",
            "is_typing": is_typing,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await conn.ws.send_text(json.dumps(payload))
        except Exception:
            pass

    async def send_system_event(
        self, user_id: str, event_name: str, data: dict[str, Any] = None
    ) -> None:
        conn = self._connections.get(user_id)
        if not conn:
            return
        payload = {
            "type": "system",
            "event": event_name,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            await conn.ws.send_text(json.dumps(payload))
        except Exception:
            pass

    # ── Offline queue ─────────────────────────────────────────

    def _enqueue(self, user_id: str, content: str, metadata: dict[str, Any]) -> None:
        if user_id not in self._offline_queues:
            self._offline_queues[user_id] = deque(maxlen=self._max_queue_size)
        self._offline_queues[user_id].append(QueuedMessage(content=content, metadata=metadata))

    async def _drain_queue(self, user_id: str, ws: Any) -> None:
        queue = self._offline_queues.pop(user_id, None)
        if not queue:
            return
        for msg in queue:
            payload = {
                "type": "message",
                "message_id": msg.message_id,
                "content": msg.content,
                "queued_at": msg.queued_at.isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                break

    # ── Inbound parsing ───────────────────────────────────────

    async def _parse_inbound(self, raw_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        content = raw_payload.get("content", "")
        sender = raw_payload.get("sender", raw_payload.get("user_id", ""))
        if not sender or not content:
            return None
        return {
            "sender_address": sender,
            "content": content,
            "metadata": {
                "channel": "chat",
                "channel_message_id": raw_payload.get("message_id", str(uuid.uuid4())),
            },
        }

    # ── Utilities ─────────────────────────────────────────────

    def _render_template(self, name: str, data: dict) -> str:
        templates = {
            "payment_reminder": (
                f"Hi {data.get('contact_name', 'there')}! Quick reminder about "
                f"your payment of {data.get('currency', 'INR')} {data.get('amount', '')} "
                f"for invoice {data.get('invoice_number', '')}."
            ),
        }
        return templates.get(name, f"Hi! Following up regarding {name}.")

    async def health_check(self) -> dict[str, Any]:
        base = await super().health_check()
        return {
            **base,
            "connected_users": len(self._connections),
            "queued_users": len(self._offline_queues),
            "total_queued_messages": sum(len(q) for q in self._offline_queues.values()),
        }
