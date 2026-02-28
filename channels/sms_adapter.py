"""
SMS Channel Adapter — Twilio-style SMS messaging.

Provides:
- GSM-7 vs Unicode detection for accurate segment counting
- Segment counting for billing awareness
- Automatic message truncation to max segment limit
- STOP/START opt-out/opt-in compliance
- Inbound parsing with media attachment support
- Status webhook handling (sent, delivered, failed)
"""
from __future__ import annotations

import re
import uuid
import structlog
from typing import Any, Optional

from models.schemas import ChannelType, Contact
from channels.base import ChannelAdapter

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  GSM-7 CHARACTER SET & SEGMENT COUNTING
# ══════════════════════════════════════════════════════════════

# GSM-7 basic character set (includes space, digits, common punctuation, Latin letters)
_GSM7_CHARS = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)

# Extended GSM-7 (takes 2 bytes each): ^{}[~]|\€
_GSM7_EXTENDED = set("^{}[]~|\\€")


def _is_gsm7(text: str) -> bool:
    """Check if all characters in text are in the GSM-7 charset."""
    return all(c in _GSM7_CHARS or c in _GSM7_EXTENDED for c in text)


def _segment_count(text: str) -> int:
    """
    Calculate SMS segment count based on encoding.

    GSM-7: 160 chars single / 153 chars per segment (7 chars for UDH header)
    Unicode: 70 chars single / 67 chars per segment
    """
    if not text:
        return 0

    if _is_gsm7(text):
        # Count extended chars as 2
        char_count = sum(2 if c in _GSM7_EXTENDED else 1 for c in text)
        if char_count <= 160:
            return 1
        return (char_count + 152) // 153  # ceil division
    else:
        if len(text) <= 70:
            return 1
        return (len(text) + 66) // 67


# ══════════════════════════════════════════════════════════════
#  SMS ADAPTER
# ══════════════════════════════════════════════════════════════

class SMSAdapter(ChannelAdapter):
    """
    SMS adapter with segment awareness and opt-out compliance.

    Automatically truncates messages to stay within the configured
    max_segments limit. Maintains an opt-out list for STOP/START
    compliance (TCPA, CTIA).
    """

    channel_type = ChannelType.SMS

    def __init__(self):
        super().__init__()
        self._from_number: str = ""
        self._max_segments: int = 3
        self._opt_out_list: set[str] = set()

    async def initialize(self, config: dict[str, Any]) -> None:
        self._config = config
        self._from_number = config.get("from_number", "")
        self._max_segments = config.get("max_segments", 3)
        self._initialized = True

    # ── Send ──────────────────────────────────────────────────

    async def _do_send(
        self, contact: Contact, content: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        phone = self.get_address(contact)
        if not phone:
            return {"status": "failed", "error": "No SMS number"}

        # Normalize for opt-out check
        normalized = re.sub(r"[^\d]", "", phone)
        if normalized in self._opt_out_list:
            return {"status": "failed", "error": "opted_out"}

        # Truncate if exceeds max segments
        content = self._truncate_to_segments(content, self._max_segments)
        segments = _segment_count(content)

        # Production: Twilio client.messages.create()
        msg_sid = f"SM{uuid.uuid4().hex[:32]}"

        logger.info("sms_sent", to=phone, segments=segments, msg_sid=msg_sid)
        return {
            "status": "mock_sent",
            "channel_message_id": msg_sid,
            "segments": segments,
            "to": phone,
        }

    async def _do_send_template(
        self, contact: Contact, template_name: str, template_data: dict[str, Any]
    ) -> dict[str, Any]:
        content = self._render_template(template_name, template_data)
        return await self._do_send(contact, content, {"template": template_name})

    # ── Inbound ───────────────────────────────────────────────

    async def _parse_inbound(self, raw_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Parse Twilio-style inbound SMS webhook."""
        sender = raw_payload.get("From", "")
        body = raw_payload.get("Body", "").strip()
        message_sid = raw_payload.get("MessageSid", str(uuid.uuid4()))

        if not sender:
            return None

        normalized = re.sub(r"[^\d]", "", sender)

        # STOP/START compliance
        body_upper = body.upper().strip()
        if body_upper in ("STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT"):
            self._opt_out_list.add(normalized)
            logger.info("sms_opt_out", phone=sender)
            return None  # Don't forward opt-out messages

        if body_upper in ("START", "YES", "UNSTOP", "SUBSCRIBE"):
            self._opt_out_list.discard(normalized)
            logger.info("sms_opt_in", phone=sender)
            return None  # Don't forward opt-in messages

        # Media attachments
        media_urls = []
        num_media = int(raw_payload.get("NumMedia", 0))
        for i in range(num_media):
            url = raw_payload.get(f"MediaUrl{i}", "")
            if url:
                media_urls.append(url)

        metadata: dict[str, Any] = {
            "channel": "sms",
            "channel_message_id": message_sid,
            "from_number": sender,
            "segments": int(raw_payload.get("NumSegments", 1)),
        }
        if media_urls:
            metadata["media_urls"] = media_urls
            metadata["has_media"] = True

        return {
            "sender_address": sender,
            "content": body,
            "metadata": metadata,
        }

    # ── Status webhook ────────────────────────────────────────

    async def handle_status_webhook(self, data: dict[str, Any]) -> dict[str, Any]:
        """Handle Twilio status callback."""
        msg_sid = data.get("MessageSid", "")
        status = data.get("MessageStatus", "").lower()
        error_code = data.get("ErrorCode", "")
        return {
            "message_sid": msg_sid,
            "status": status,
            "error_code": error_code,
        }

    # ── Truncation ────────────────────────────────────────────

    def _truncate_to_segments(self, content: str, max_segments: int) -> str:
        """Truncate message to fit within max_segments."""
        if _segment_count(content) <= max_segments:
            return content

        if _is_gsm7(content):
            max_chars = 153 * max_segments - 3  # space for "..."
        else:
            max_chars = 67 * max_segments - 3

        return content[:max_chars] + "..."

    # ── Templates ─────────────────────────────────────────────

    def _render_template(self, name: str, data: dict) -> str:
        cn = data.get("contact_name", "there")
        templates = {
            "payment_reminder": (
                f"Hi {cn}, reminder: {data.get('currency', 'INR')} "
                f"{data.get('amount', '')} due for {data.get('invoice_number', '')}. "
                f"Please pay ASAP. Reply STOP to opt out."
            ),
        }
        return templates.get(name, f"Hi {cn}, following up. Reply STOP to opt out.")
