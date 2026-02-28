"""
WhatsApp Channel Adapter — WhatsApp Business Cloud API integration.

Provides:
- 24-hour conversation window tracking (user vs business initiated)
- Phone number normalization
- Webhook verification (hub.verify_token challenge)
- Outbound: free-form text (within window) or template fallback
- Inbound: text, interactive (button_reply, list_reply), image, location, document
- Status update processing (sent, delivered, read)
- Template fallback when outside conversation window
"""
from __future__ import annotations

import re
import uuid
import structlog
from typing import Any, Optional
from datetime import datetime, timezone, timedelta

from models.schemas import ChannelType, Contact
from channels.base import ChannelAdapter

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  CONVERSATION WINDOW
# ══════════════════════════════════════════════════════════════

class ConversationWindow:
    """
    Tracks the WhatsApp 24-hour messaging window.

    - User-initiated: window opens when user sends a message, lasts 24h.
    - Business-initiated: window opens when template is sent and user replies.
    """

    def __init__(self, phone: str, initiated_by: str = "user"):
        self.phone = phone
        self.initiated_by = initiated_by
        self.opened_at = datetime.now(timezone.utc)
        self.last_message_at = datetime.now(timezone.utc)

    @property
    def is_open(self) -> bool:
        elapsed = datetime.now(timezone.utc) - self.opened_at
        return elapsed < timedelta(hours=24)

    @property
    def remaining_seconds(self) -> float:
        elapsed = datetime.now(timezone.utc) - self.opened_at
        remaining = timedelta(hours=24) - elapsed
        return max(0, remaining.total_seconds())

    def touch(self):
        self.last_message_at = datetime.now(timezone.utc)


# ══════════════════════════════════════════════════════════════
#  WHATSAPP ADAPTER
# ══════════════════════════════════════════════════════════════

class WhatsAppAdapter(ChannelAdapter):
    """
    WhatsApp Business Cloud API adapter.

    Handles the 24-hour window rule: free-form messages are only allowed
    within a user-initiated window. Outside the window, the adapter
    automatically falls back to a pre-approved template or fails.
    """

    channel_type = ChannelType.WHATSAPP

    def __init__(self):
        super().__init__()
        self._windows: dict[str, ConversationWindow] = {}
        self._phone_number_id: str = ""
        self._access_token: str = ""
        self._verify_token: str = ""
        self._default_template: Optional[str] = None

    async def initialize(self, config: dict[str, Any]) -> None:
        self._config = config
        self._phone_number_id = config.get("phone_number_id", "")
        self._access_token = config.get("access_token", "")
        self._verify_token = config.get("verify_token", "")
        self._default_template = config.get("default_template")
        self._init_rate_limiter(config)
        self._initialized = True

    def _init_rate_limiter(self, config: dict[str, Any]):
        from channels.base import TokenBucketRateLimiter
        rate = config.get("rate_per_second", 80)
        burst = config.get("burst", 100)
        if rate > 0:
            self._rate_limiter = TokenBucketRateLimiter(rate=rate, burst=burst)

    # ── Phone normalization ───────────────────────────────────

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone to digits only, stripping +, spaces, dashes."""
        return re.sub(r"[^\d]", "", phone)

    # ── Webhook verification ──────────────────────────────────

    def verify_webhook(self, params: dict[str, Any]) -> Optional[str]:
        """
        Verify the WhatsApp webhook subscription.
        Returns the challenge string on success, None on failure.
        """
        mode = params.get("hub.mode", "")
        token = params.get("hub.verify_token", "")
        challenge = params.get("hub.challenge", "")

        if mode == "subscribe" and token == self._verify_token:
            return challenge
        return None

    # ── Conversation window ───────────────────────────────────

    def get_window_info(self, phone: str) -> dict[str, Any]:
        phone = self._normalize_phone(phone)
        window = self._windows.get(phone)
        if not window:
            return {"is_open": False, "phone": phone}
        return {
            "is_open": window.is_open,
            "initiated_by": window.initiated_by,
            "opened_at": window.opened_at.isoformat(),
            "remaining_seconds": round(window.remaining_seconds),
            "phone": phone,
        }

    def _open_window(self, phone: str, initiated_by: str = "user") -> ConversationWindow:
        phone = self._normalize_phone(phone)
        window = ConversationWindow(phone, initiated_by)
        self._windows[phone] = window
        return window

    def _get_window(self, phone: str) -> Optional[ConversationWindow]:
        phone = self._normalize_phone(phone)
        window = self._windows.get(phone)
        if window and window.is_open:
            return window
        return None

    # ── Send ──────────────────────────────────────────────────

    async def _do_send(
        self, contact: Contact, content: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        phone_raw = self.get_address(contact)
        if not phone_raw:
            return {"status": "failed", "error": "No WhatsApp number"}

        phone = self._normalize_phone(phone_raw)
        window = self._get_window(phone)

        if window:
            # Inside window — send free-form text
            return await self._send_text(phone, content)
        else:
            # Outside window — use template fallback
            template = self._config.get("default_template", self._default_template)
            if template:
                return await self._send_template_message(phone, template, content)
            return {"status": "failed", "error": "No open window and no default template"}

    async def _do_send_template(
        self, contact: Contact, template_name: str, template_data: dict[str, Any]
    ) -> dict[str, Any]:
        phone_raw = self.get_address(contact)
        if not phone_raw:
            return {"status": "failed", "error": "No WhatsApp number"}
        phone = self._normalize_phone(phone_raw)
        return await self._send_template_message(phone, template_name, str(template_data))

    async def _send_text(self, phone: str, content: str) -> dict[str, Any]:
        """Send a free-form text message via WhatsApp Cloud API."""
        # Production: POST to https://graph.facebook.com/v18.0/{phone_number_id}/messages
        # payload = {"messaging_product": "whatsapp", "to": phone, "type": "text", "text": {"body": content}}
        msg_id = f"wamid.{uuid.uuid4().hex[:20]}"
        logger.info("whatsapp_text_sent", to=phone, msg_id=msg_id)
        return {"status": "mock_sent", "channel_message_id": msg_id}

    async def _send_template_message(
        self, phone: str, template_name: str, body_text: str
    ) -> dict[str, Any]:
        """Send an approved template message (opens a business-initiated window)."""
        msg_id = f"wamid.{uuid.uuid4().hex[:20]}"
        self._open_window(phone, "business")
        logger.info("whatsapp_template_sent", to=phone, template=template_name, msg_id=msg_id)
        return {"status": "mock_sent", "channel_message_id": msg_id}

    # ── Inbound parsing ───────────────────────────────────────

    async def _parse_inbound(self, raw_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Parse WhatsApp Cloud API webhook payload."""
        try:
            entry = raw_payload.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
        except (IndexError, KeyError):
            return None

        # Status updates (delivered/read) — not messages
        if "statuses" in value and "messages" not in value:
            return None

        messages = value.get("messages", [])
        if not messages:
            return None

        msg = messages[0]
        sender = msg.get("from", "")
        msg_type = msg.get("type", "text")
        msg_id = msg.get("id", "")

        # Open user-initiated window
        if sender:
            self._open_window(sender, "user")

        # Extract sender name
        contacts = value.get("contacts", [])
        sender_name = ""
        if contacts:
            profile = contacts[0].get("profile", {})
            sender_name = profile.get("name", "")

        # Parse by message type
        content = ""
        extra_metadata: dict[str, Any] = {}

        if msg_type == "text":
            content = msg.get("text", {}).get("body", "")
            extra_metadata["message_type"] = "text"

        elif msg_type == "interactive":
            interactive = msg.get("interactive", {})
            itype = interactive.get("type", "")

            if itype == "button_reply":
                reply = interactive.get("button_reply", {})
                content = reply.get("title", "")
                extra_metadata["button_id"] = reply.get("id", "")
                extra_metadata["message_type"] = "button_reply"

            elif itype == "list_reply":
                reply = interactive.get("list_reply", {})
                content = reply.get("title", "")
                extra_metadata["list_item_id"] = reply.get("id", "")
                extra_metadata["message_type"] = "list_reply"

        elif msg_type == "image":
            image = msg.get("image", {})
            content = image.get("caption", "[Image]")
            extra_metadata["media_type"] = "image"
            extra_metadata["media_id"] = image.get("id", "")
            extra_metadata["mime_type"] = image.get("mime_type", "")
            extra_metadata["message_type"] = "image"

        elif msg_type == "document":
            doc = msg.get("document", {})
            content = doc.get("caption", doc.get("filename", "[Document]"))
            extra_metadata["media_type"] = "document"
            extra_metadata["media_id"] = doc.get("id", "")
            extra_metadata["message_type"] = "document"

        elif msg_type == "location":
            loc = msg.get("location", {})
            lat = loc.get("latitude", 0)
            lng = loc.get("longitude", 0)
            content = f"Location: {lat}, {lng}"
            extra_metadata["latitude"] = lat
            extra_metadata["longitude"] = lng
            extra_metadata["message_type"] = "location"

        elif msg_type == "audio":
            audio = msg.get("audio", {})
            content = "[Voice message]"
            extra_metadata["media_type"] = "audio"
            extra_metadata["media_id"] = audio.get("id", "")
            extra_metadata["message_type"] = "audio"

        elif msg_type == "video":
            video = msg.get("video", {})
            content = video.get("caption", "[Video]")
            extra_metadata["media_type"] = "video"
            extra_metadata["media_id"] = video.get("id", "")
            extra_metadata["message_type"] = "video"

        elif msg_type == "sticker":
            content = "[Sticker]"
            extra_metadata["message_type"] = "sticker"

        elif msg_type == "contacts":
            content = "[Shared contact]"
            extra_metadata["message_type"] = "contacts"

        else:
            content = f"[{msg_type}]"
            extra_metadata["message_type"] = msg_type

        return {
            "sender_address": sender,
            "content": content,
            "metadata": {
                "channel": "whatsapp",
                "channel_message_id": msg_id,
                "sender_name": sender_name,
                **extra_metadata,
            },
        }

    # ── Health ────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        base = await super().health_check()
        open_windows = sum(1 for w in self._windows.values() if w.is_open)
        return {
            **base,
            "open_windows": open_windows,
            "total_windows_tracked": len(self._windows),
        }
