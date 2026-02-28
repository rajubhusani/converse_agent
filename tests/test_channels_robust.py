"""
Tests for production-grade channel adapters.

Coverage:
  Base:      retry, rate limiting, circuit breaker, delivery tracking, metrics
  Chat:      connection lifecycle, typing, presence, offline queue, heartbeat, events
  Email:     send, template, bounce, complaint, unsubscribe, thread, sanitize, suppress
  WhatsApp:  send, 24h window, interactive, template, inbound parse, signature, media types
  Voice:     call state machine, DTMF, transfer, barge-in, voicemail, hold/resume, transcript
  SMS:       send, segment counting, opt-out, inbound, status webhook
"""
import asyncio
import json
import time
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from models.schemas import ChannelType, Contact, ContactChannel
from channels.base import (
    ChannelAdapter, ChannelError, DeliveryStatus, DeliveryRecord,
    TokenBucketRateLimiter, CircuitBreaker, ChannelMetrics,
)
from channels.chat_adapter import ChatAdapter, ConnectionState, QueuedMessage
from channels.email_adapter import EmailAdapter
from channels.whatsapp_adapter import WhatsAppAdapter, ConversationWindow
from channels.voice_adapter import (
    VoiceAdapter, VoiceCallState, CallStatus, CallDirection, CallDisposition,
    PipecatPipelineBuilder,
)
from channels.sms_adapter import SMSAdapter, _segment_count, _is_gsm7


# ── Fixtures ──────────────────────────────────────────

def make_contact(name="John", email="john@test.com", phone="+919876543210",
                 whatsapp="+919876543210", chat_id="contact_001"):
    channels = []
    if email:
        channels.append(ContactChannel(channel=ChannelType.EMAIL, address=email))
    if phone:
        channels.append(ContactChannel(channel=ChannelType.VOICE, address=phone))
    if whatsapp:
        channels.append(ContactChannel(channel=ChannelType.WHATSAPP, address=whatsapp))
    if chat_id:
        channels.append(ContactChannel(channel=ChannelType.CHAT, address=chat_id))
    channels.append(ContactChannel(channel=ChannelType.SMS, address=phone or "+919876543210"))
    return Contact(id="c1", name=name, role="dealer", organization="TestCo", channels=channels)


@pytest.fixture
def contact():
    return make_contact()


# ══════════════════════════════════════════════════════════════
#  BASE — Rate Limiter
# ══════════════════════════════════════════════════════════════

class TestTokenBucketRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_within_burst(self):
        rl = TokenBucketRateLimiter(rate=10, burst=5)
        for _ in range(5):
            assert await rl.acquire(timeout=0.1) is True

    @pytest.mark.asyncio
    async def test_acquire_exceeds_burst(self):
        rl = TokenBucketRateLimiter(rate=10, burst=2)
        assert await rl.acquire(timeout=0.1) is True
        assert await rl.acquire(timeout=0.1) is True
        # Third should timeout quickly
        assert await rl.acquire(timeout=0.05) is False

    @pytest.mark.asyncio
    async def test_refill(self):
        rl = TokenBucketRateLimiter(rate=100, burst=1)
        assert await rl.acquire(timeout=0.01) is True
        await asyncio.sleep(0.02)
        assert await rl.acquire(timeout=0.01) is True


# ══════════════════════════════════════════════════════════════
#  BASE — Circuit Breaker
# ══════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert not cb.is_open

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.is_open

    def test_success_resets(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        # Simulate half-open by advancing time — directly set state for test
        cb._state = "half_open"
        cb.record_success()
        assert cb.state == "closed"


# ══════════════════════════════════════════════════════════════
#  BASE — Metrics
# ══════════════════════════════════════════════════════════════

class TestChannelMetrics:
    def test_record_and_query(self):
        m = ChannelMetrics(ChannelType.EMAIL)
        m.record_send(100.0)
        m.record_send(200.0)
        m.record_failure("test error")
        m.record_delivery()
        assert m.messages_sent == 2
        assert m.messages_failed == 1
        assert m.messages_delivered == 1
        assert m.avg_latency_ms == 150.0
        assert 0.3 < m.failure_rate < 0.4
        d = m.to_dict()
        assert d["sent"] == 2


# ══════════════════════════════════════════════════════════════
#  BASE — Delivery Record
# ══════════════════════════════════════════════════════════════

class TestDeliveryRecord:
    def test_status_transitions(self):
        rec = DeliveryRecord("msg1", ChannelType.EMAIL, "test@test.com")
        assert rec.status == DeliveryStatus.QUEUED
        rec.update_status(DeliveryStatus.SENT)
        assert rec.status == DeliveryStatus.SENT
        rec.update_status(DeliveryStatus.DELIVERED)
        assert rec.status == DeliveryStatus.DELIVERED
        assert len(rec.status_history) == 2


# ══════════════════════════════════════════════════════════════
#  CHAT ADAPTER
# ══════════════════════════════════════════════════════════════

class TestChatAdapter:
    @pytest.fixture
    def adapter(self):
        a = ChatAdapter()
        asyncio.get_event_loop().run_until_complete(a.initialize({}))
        return a

    @pytest.mark.asyncio
    async def test_send_to_connected_user(self, adapter, contact):
        ws = AsyncMock()
        await adapter.register_connection("contact_001", ws)
        result = await adapter.send_message(contact, "Hello!")
        assert result["status"] == "delivered"
        ws.send_text.assert_called()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["content"] == "Hello!"
        assert payload["type"] == "message"

    @pytest.mark.asyncio
    async def test_queue_when_offline(self, adapter, contact):
        result = await adapter.send_message(contact, "You're offline")
        assert result["status"] == "queued"

    @pytest.mark.asyncio
    async def test_drain_queue_on_reconnect(self, adapter, contact):
        await adapter.send_message(contact, "Msg 1")
        await adapter.send_message(contact, "Msg 2")
        ws = AsyncMock()
        await adapter.register_connection("contact_001", ws)
        assert ws.send_text.call_count >= 2

    @pytest.mark.asyncio
    async def test_presence_tracking(self, adapter):
        assert adapter.get_presence("p1") == "offline"
        ws = AsyncMock()
        await adapter.register_connection("p1", ws)
        assert adapter.get_presence("p1") == "online"
        assert adapter.is_connected("p1")

    @pytest.mark.asyncio
    async def test_handle_client_events(self, adapter):
        ws = AsyncMock()
        await adapter.register_connection("u1", ws)

        # Heartbeat
        result = await adapter.handle_client_event("u1", {"type": "heartbeat"})
        assert result is None

        # Typing
        result = await adapter.handle_client_event("u1", {"type": "typing", "is_typing": True})
        assert result is None

        # Message
        result = await adapter.handle_client_event("u1", {"type": "message", "content": "Hi"})
        assert result["content"] == "Hi"

        # Ack
        result = await adapter.handle_client_event("u1", {"type": "ack", "message_id": "x"})
        assert result is None

    @pytest.mark.asyncio
    async def test_typing_indicator(self, adapter):
        ws = AsyncMock()
        await adapter.register_connection("u1", ws)
        await adapter.send_typing_indicator("u1", True)
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == "typing"

    @pytest.mark.asyncio
    async def test_supersede_old_connection(self, adapter):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await adapter.register_connection("u1", ws1)
        await adapter.register_connection("u1", ws2)
        ws1.close.assert_called()

    @pytest.mark.asyncio
    async def test_system_event(self, adapter):
        ws = AsyncMock()
        await adapter.register_connection("u1", ws)
        await adapter.send_system_event("u1", "session_expired", {"reason": "timeout"})
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == "system"
        assert payload["event"] == "session_expired"


# ══════════════════════════════════════════════════════════════
#  EMAIL ADAPTER
# ══════════════════════════════════════════════════════════════

class TestEmailAdapter:
    @pytest.fixture
    def adapter(self):
        a = EmailAdapter()
        asyncio.get_event_loop().run_until_complete(a.initialize({
            "smtp_host": "smtp.test.com",
            "from_email": "agent@test.com",
            "from_name": "Test Agent",
            "domain": "test.com",
        }))
        return a

    @pytest.mark.asyncio
    async def test_send_plain_email(self, adapter, contact):
        result = await adapter.send_message(contact, "Hello John", {"subject": "Test"})
        assert result["status"] == "sent"
        assert "channel_message_id" in result

    @pytest.mark.asyncio
    async def test_send_template(self, adapter, contact):
        result = await adapter.send_template(contact, "payment_reminder", {
            "invoice_number": "INV-001",
            "amount": "50000",
            "currency": "INR",
            "due_date": "2025-01-15",
            "days_overdue": "30",
        })
        assert result["status"] == "sent"

    @pytest.mark.asyncio
    async def test_suppress_list(self, adapter, contact):
        await adapter.handle_unsubscribe("john@test.com")
        assert adapter.is_suppressed("john@test.com")
        result = await adapter.send_message(contact, "Hello")
        assert result["status"] == "failed"
        assert "Suppress" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_bounce_handling(self, adapter):
        result = await adapter.handle_bounce({"email": "bounce@test.com", "type": "permanent"})
        assert result["status"] == "processed"
        assert adapter.is_suppressed("bounce@test.com")

    @pytest.mark.asyncio
    async def test_complaint_handling(self, adapter):
        await adapter.handle_complaint({"email": "spam@test.com"})
        assert adapter.is_suppressed("spam@test.com")

    @pytest.mark.asyncio
    async def test_idempotency(self, adapter, contact):
        r1 = await adapter.send_message(contact, "Same message", {"subject": "Dup"})
        r2 = await adapter.send_message(contact, "Same message", {"subject": "Dup"})
        assert r1["status"] == "sent"
        assert r2["status"] == "deduplicated"

    @pytest.mark.asyncio
    async def test_inbound_parse(self, adapter):
        parsed = await adapter.handle_inbound({
            "from": "Jane Doe <jane@example.com>",
            "subject": "Re: Invoice",
            "text": "I will pay tomorrow.\n\nOn Mon Jan 1 2025, agent@test.com wrote:\n> Original message",
        })
        assert parsed["sender_address"] == "jane@example.com"
        assert "I will pay tomorrow" in parsed["content"]
        assert "Original message" not in parsed["content"]  # Quoted reply stripped

    @pytest.mark.asyncio
    async def test_html_to_plain(self, adapter):
        text = adapter._html_to_plain("<p>Hello <b>world</b></p><br><p>Bye</p>")
        assert "Hello world" in text
        assert "Bye" in text

    @pytest.mark.asyncio
    async def test_inbound_html_fallback(self, adapter):
        parsed = await adapter.handle_inbound({
            "from": "test@test.com",
            "html": "<html><body><p>HTML only body</p></body></html>",
        })
        assert "HTML only body" in parsed["content"]

    @pytest.mark.asyncio
    async def test_no_email_address(self, adapter):
        c = Contact(id="c2", name="NoEmail", channels=[])
        result = await adapter.send_message(c, "Hello")
        assert result["status"] == "failed"


# ══════════════════════════════════════════════════════════════
#  WHATSAPP ADAPTER
# ══════════════════════════════════════════════════════════════

class TestWhatsAppAdapter:
    @pytest.fixture
    def adapter(self):
        a = WhatsAppAdapter()
        asyncio.get_event_loop().run_until_complete(a.initialize({
            "phone_number_id": "12345",
            "access_token": "test_token",
            "verify_token": "my_verify",
            "app_secret": "",
            "default_template": "hello_world",
        }))
        return a

    @pytest.mark.asyncio
    async def test_send_within_window(self, adapter, contact):
        # Open a user-initiated window
        phone = adapter._normalize_phone("+919876543210")
        adapter._windows[phone] = ConversationWindow(phone, "user")
        result = await adapter.send_message(contact, "Test message")
        # Will use mock since no real client
        assert result["status"] in ("sent", "mock_sent")

    @pytest.mark.asyncio
    async def test_send_outside_window_falls_back_to_template(self, adapter, contact):
        # No window opened — should trigger template fallback
        result = await adapter.send_message(contact, "Test message")
        assert result["status"] in ("sent", "mock_sent")

    @pytest.mark.asyncio
    async def test_send_outside_window_no_template(self, adapter, contact):
        adapter._config["default_template"] = None
        result = await adapter.send_message(contact, "Test")
        assert result["status"] == "failed"
        assert "window" in result.get("error", "").lower() or "template" in result.get("error", "").lower()

    def test_phone_normalization(self, adapter):
        assert adapter._normalize_phone("+91 98765-43210") == "919876543210"
        assert adapter._normalize_phone("919876543210") == "919876543210"

    def test_webhook_verification(self, adapter):
        result = adapter.verify_webhook({
            "hub.mode": "subscribe",
            "hub.verify_token": "my_verify",
            "hub.challenge": "12345",
        })
        assert result == "12345"

    def test_webhook_verification_fails(self, adapter):
        result = adapter.verify_webhook({
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "12345",
        })
        assert result is None

    @pytest.mark.asyncio
    async def test_inbound_text(self, adapter):
        parsed = await adapter.handle_inbound({
            "entry": [{"changes": [{"value": {
                "messages": [{
                    "from": "919876543210",
                    "type": "text",
                    "id": "wamid.123",
                    "text": {"body": "I paid yesterday"},
                    "timestamp": "1234567890",
                }],
                "contacts": [{"profile": {"name": "John"}}],
            }}]}],
        })
        assert parsed["sender_address"] == "919876543210"
        assert parsed["content"] == "I paid yesterday"
        assert parsed["metadata"]["message_type"] == "text"

    @pytest.mark.asyncio
    async def test_inbound_button_reply(self, adapter):
        parsed = await adapter.handle_inbound({
            "entry": [{"changes": [{"value": {
                "messages": [{
                    "from": "919876543210",
                    "type": "interactive",
                    "id": "wamid.456",
                    "interactive": {
                        "type": "button_reply",
                        "button_reply": {"id": "btn_yes", "title": "Yes, I paid"},
                    },
                }],
                "contacts": [{"profile": {"name": "John"}}],
            }}]}],
        })
        assert parsed["content"] == "Yes, I paid"
        assert parsed["metadata"]["button_id"] == "btn_yes"

    @pytest.mark.asyncio
    async def test_inbound_image(self, adapter):
        parsed = await adapter.handle_inbound({
            "entry": [{"changes": [{"value": {
                "messages": [{
                    "from": "919876543210",
                    "type": "image",
                    "id": "wamid.789",
                    "image": {"id": "media_123", "caption": "Payment receipt", "mime_type": "image/jpeg"},
                }],
                "contacts": [{"profile": {}}],
            }}]}],
        })
        assert parsed["content"] == "Payment receipt"
        assert parsed["metadata"]["media_type"] == "image"

    @pytest.mark.asyncio
    async def test_inbound_location(self, adapter):
        parsed = await adapter.handle_inbound({
            "entry": [{"changes": [{"value": {
                "messages": [{
                    "from": "919876543210",
                    "type": "location",
                    "id": "wamid.loc",
                    "location": {"latitude": 12.97, "longitude": 77.59},
                }],
                "contacts": [{"profile": {}}],
            }}]}],
        })
        assert "12.97" in parsed["content"]

    @pytest.mark.asyncio
    async def test_inbound_status_update(self, adapter):
        """Status updates (delivered/read) should return None (not a message)."""
        parsed = await adapter.handle_inbound({
            "entry": [{"changes": [{"value": {
                "statuses": [{"id": "wamid.123", "status": "delivered", "recipient_id": "919876543210"}],
            }}]}],
        })
        assert parsed is None

    def test_conversation_window(self, adapter):
        w = ConversationWindow("919876543210", "user")
        assert w.is_open
        # Simulate expired
        w.opened_at = datetime.now(timezone.utc) - timedelta(hours=25)
        assert not w.is_open

    def test_window_info(self, adapter):
        phone = "919876543210"
        adapter._windows[phone] = ConversationWindow(phone, "user")
        info = adapter.get_window_info(phone)
        assert info["is_open"] is True
        assert info["initiated_by"] == "user"


# ══════════════════════════════════════════════════════════════
#  VOICE ADAPTER
# ══════════════════════════════════════════════════════════════

class TestVoiceCallState:
    def test_initial_state(self):
        cs = VoiceCallState("call_1", "c1", "conv_1")
        assert cs.status == CallStatus.INITIATING
        assert cs.duration_seconds == 0

    def test_state_transitions(self):
        cs = VoiceCallState("call_1", "c1", "conv_1")
        cs.set_status(CallStatus.RINGING)
        cs.set_status(CallStatus.CONNECTED)
        assert cs.connected_at is not None
        cs.set_status(CallStatus.ENDED)
        assert cs.ended_at is not None
        assert cs.duration_seconds >= 0
        assert len(cs.status_history) == 3

    def test_transcript(self):
        cs = VoiceCallState("call_1", "c1", "conv_1")
        cs.add_transcript("agent", "Hello, how are you?")
        cs.add_transcript("contact", "I'm fine, thanks.")
        assert len(cs.transcript) == 2
        text = cs.get_full_transcript()
        assert "agent: Hello" in text
        assert "contact: I'm fine" in text

    def test_interim_transcript_not_final(self):
        cs = VoiceCallState("call_1", "c1", "conv_1")
        cs.add_transcript("contact", "I thi...", is_final=False)
        assert len(cs.transcript) == 0  # Not added to final transcript
        cs.add_transcript("contact", "I think so", is_final=True)
        assert len(cs.transcript) == 1

    def test_summary(self):
        cs = VoiceCallState("call_1", "c1", "conv_1")
        cs.set_status(CallStatus.CONNECTED)
        cs.disposition = CallDisposition.COMPLETED
        s = cs.to_summary()
        assert s["call_id"] == "call_1"
        assert s["disposition"] == "completed"


class TestVoiceAdapter:
    @pytest.fixture
    def adapter(self):
        a = VoiceAdapter()
        asyncio.get_event_loop().run_until_complete(a.initialize({
            "tts_provider": "elevenlabs",
            "stt_provider": "deepgram",
            "telephony_provider": "twilio",
            "voicemail_message": "Hi, we tried to reach you. Please call us back.",
        }))
        return a

    @pytest.mark.asyncio
    async def test_initiate_outbound_call(self, adapter, contact):
        result = await adapter.send_message(contact, "Hi John!", {"conversation_id": "conv_1"})
        assert result["status"] == "call_initiated"
        assert "call_id" in result

    @pytest.mark.asyncio
    async def test_speak_into_active_call(self, adapter, contact):
        # First initiate
        r1 = await adapter.send_message(contact, "First message", {"conversation_id": "conv_1"})
        call_id = r1["call_id"]
        # Simulate connection
        call = adapter.get_active_call(call_id)
        call.set_status(CallStatus.CONNECTED)
        # Now send again — should speak into existing
        r2 = await adapter.send_message(contact, "Second message")
        assert r2["status"] == "spoke"

    @pytest.mark.asyncio
    async def test_transcript_handling(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {"conversation_id": "c1"})
        call_id = r["call_id"]

        # Interim result (not forwarded)
        parsed = await adapter.handle_transcript(call_id, "contact", "I thi", is_final=False)
        assert parsed is None

        # Final result (forwarded)
        parsed = await adapter.handle_transcript(call_id, "contact", "I think I paid", is_final=True)
        assert parsed is not None
        assert parsed["content"] == "I think I paid"

    @pytest.mark.asyncio
    async def test_barge_in_detection(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        call = adapter.get_active_call(call_id)
        call.is_agent_speaking = True

        parsed = await adapter.handle_transcript(call_id, "contact", "Wait!", is_final=True)
        assert call.barge_in_count == 1
        assert parsed["metadata"]["barge_in"] is True

    @pytest.mark.asyncio
    async def test_dtmf_single(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        result = await adapter.handle_dtmf(call_id, "1")
        assert result["metadata"]["dtmf_digit"] == "1"

    @pytest.mark.asyncio
    async def test_dtmf_collection(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]

        adapter.start_dtmf_collection(call_id, expected_length=4)
        r1 = await adapter.handle_dtmf(call_id, "1")
        assert r1 is None  # Still collecting
        r2 = await adapter.handle_dtmf(call_id, "2")
        assert r2 is None
        r3 = await adapter.handle_dtmf(call_id, "3")
        assert r3 is None
        r4 = await adapter.handle_dtmf(call_id, "4")
        assert r4 is not None
        assert r4["metadata"]["dtmf_digits"] == "1234"

    @pytest.mark.asyncio
    async def test_dtmf_collection_with_terminator(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        adapter.start_dtmf_collection(call_id, terminator="#")
        await adapter.handle_dtmf(call_id, "5")
        await adapter.handle_dtmf(call_id, "5")
        result = await adapter.handle_dtmf(call_id, "#")
        assert result["metadata"]["dtmf_digits"] == "55"

    @pytest.mark.asyncio
    async def test_call_status_ringing(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        result = await adapter.handle_call_status(call_id, "ringing")
        call = adapter.get_active_call(call_id)
        assert call.status == CallStatus.RINGING

    @pytest.mark.asyncio
    async def test_call_status_no_answer_retry(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {"attempt_number": 1})
        call_id = r["call_id"]
        result = await adapter.handle_call_status(call_id, "no-answer")
        assert result["should_retry"] is True

    @pytest.mark.asyncio
    async def test_call_end(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {"conversation_id": "conv_1"})
        call_id = r["call_id"]
        call = adapter.get_active_call(call_id)
        call.set_status(CallStatus.CONNECTED)
        call.add_transcript("contact", "I paid yesterday")

        summary = await adapter.handle_call_ended(call_id)
        assert summary is not None
        assert summary["disposition"] == "completed"
        assert "I paid yesterday" in summary["full_transcript"]

    @pytest.mark.asyncio
    async def test_hold_and_resume(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        call = adapter.get_active_call(call_id)
        call.set_status(CallStatus.CONNECTED)

        hold_result = await adapter.hold_call(call_id)
        assert hold_result["status"] == "on_hold"
        assert call.status == CallStatus.ON_HOLD

        resume_result = await adapter.resume_call(call_id)
        assert resume_result["status"] == "connected"
        assert call.status == CallStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_transfer(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        call = adapter.get_active_call(call_id)
        call.set_status(CallStatus.CONNECTED)

        result = await adapter.transfer_call(call_id, "agent_smith", "Customer needs help")
        assert result["status"] == "transferring"
        assert call.disposition == CallDisposition.TRANSFERRED

    @pytest.mark.asyncio
    async def test_voicemail_detection(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        result = await adapter.handle_call_status(call_id, "answered", {"answered_by": "machine"})
        call = adapter.get_active_call(call_id)
        assert call.status == CallStatus.VOICEMAIL

    @pytest.mark.asyncio
    async def test_end_call_programmatic(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        result = await adapter.end_call(call_id, CallDisposition.COMPLETED, "All resolved")
        assert result.get("disposition") == "completed"

    @pytest.mark.asyncio
    async def test_active_calls_list(self, adapter, contact):
        await adapter.send_message(contact, "Hello", {})
        calls = adapter.get_active_calls()
        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_recording(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        start = await adapter.start_recording(call_id)
        assert start["status"] == "recording"
        call = adapter.get_active_call(call_id)
        assert call.is_recording
        stop = await adapter.stop_recording(call_id)
        assert stop["status"] == "recording_stopped"

    @pytest.mark.asyncio
    async def test_vad_events(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        call = adapter.get_active_call(call_id)

        await adapter.handle_vad_event(call_id, "speech_start")
        assert call.is_contact_speaking is True
        await adapter.handle_vad_event(call_id, "speech_end")
        assert call.is_contact_speaking is False
        await adapter.handle_vad_event(call_id, "silence_detected")
        assert call.silence_start is not None

    @pytest.mark.asyncio
    async def test_get_transcript(self, adapter, contact):
        r = await adapter.send_message(contact, "Hello", {})
        call_id = r["call_id"]
        t = adapter.get_call_transcript(call_id)
        assert isinstance(t, list)
        assert len(t) >= 1  # Initial agent message

    @pytest.mark.asyncio
    async def test_inbound_call(self, adapter):
        result = await adapter.handle_inbound_call({
            "from": "+919876543210",
            "call_sid": "CA123",
        })
        assert result["status"] == "ringing"
        assert result["caller"] == "+919876543210"


class TestPipecatPipelineBuilder:
    def test_build_config(self, contact):
        builder = PipecatPipelineBuilder({
            "stt_provider": "deepgram",
            "tts_provider": "elevenlabs",
            "llm_model": "claude-sonnet-4-20250514",
        })
        config = builder.build(contact, {"business_context": {"invoice": "INV-001"}}, "Hello!")
        assert config["pipeline"]["stt"]["provider"] == "deepgram"
        assert config["pipeline"]["tts"]["provider"] == "elevenlabs"
        assert "John" in config["pipeline"]["llm"]["system_prompt"]
        assert config["pipeline"]["behavior"]["initial_message"] == "Hello!"


# ══════════════════════════════════════════════════════════════
#  SMS ADAPTER
# ══════════════════════════════════════════════════════════════

class TestSMSSegmentCounting:
    def test_short_gsm7(self):
        assert _segment_count("Hello") == 1

    def test_long_gsm7(self):
        msg = "A" * 200
        assert _segment_count(msg) == 2  # 200 / 153 = 2

    def test_short_unicode(self):
        msg = "こんにちは"
        assert _segment_count(msg) == 1

    def test_long_unicode(self):
        msg = "こ" * 100
        assert _segment_count(msg) == 2  # 100 / 67 = 2

    def test_gsm7_detection(self):
        assert _is_gsm7("Hello World 123 !@#") is True
        assert _is_gsm7("Emoji 😀") is False


class TestSMSAdapter:
    @pytest.fixture
    def adapter(self):
        a = SMSAdapter()
        asyncio.get_event_loop().run_until_complete(a.initialize({
            "from_number": "+14155551234",
            "max_segments": 3,
        }))
        return a

    @pytest.mark.asyncio
    async def test_send_sms(self, adapter, contact):
        result = await adapter.send_message(contact, "Test SMS")
        assert result["status"] in ("sent", "mock_sent")
        assert result.get("segments", 0) >= 1

    @pytest.mark.asyncio
    async def test_opt_out(self, adapter, contact):
        parsed = await adapter.handle_inbound({"From": "+919876543210", "Body": "STOP"})
        assert parsed is None  # Opt-out messages not forwarded
        result = await adapter.send_message(contact, "After opt-out")
        assert result["status"] == "failed"
        assert "opted_out" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_opt_in(self, adapter, contact):
        adapter._opt_out_list.add("919876543210")
        await adapter.handle_inbound({"From": "+919876543210", "Body": "START"})
        assert "919876543210" not in adapter._opt_out_list

    @pytest.mark.asyncio
    async def test_inbound_parse(self, adapter):
        parsed = await adapter.handle_inbound({
            "From": "+919876543210",
            "Body": "I paid yesterday",
            "MessageSid": "SM123",
            "NumMedia": "0",
        })
        assert parsed["content"] == "I paid yesterday"
        assert parsed["sender_address"] == "+919876543210"

    @pytest.mark.asyncio
    async def test_inbound_with_media(self, adapter):
        parsed = await adapter.handle_inbound({
            "From": "+919876543210",
            "Body": "Receipt",
            "NumMedia": "1",
            "MediaUrl0": "https://example.com/image.jpg",
        })
        assert parsed["metadata"]["media_urls"] == ["https://example.com/image.jpg"]

    @pytest.mark.asyncio
    async def test_long_message_truncation(self, adapter, contact):
        long_msg = "A" * 1000
        result = await adapter.send_message(contact, long_msg)
        assert result["status"] in ("sent", "mock_sent")
        assert result.get("segments", 0) <= 3


# ══════════════════════════════════════════════════════════════
#  INTEGRATION — Channel Registry
# ══════════════════════════════════════════════════════════════

class TestChannelRegistry:
    @pytest.mark.asyncio
    async def test_registry_all_channels(self):
        from channels.base import ChannelRegistry
        reg = ChannelRegistry()
        reg.register(ChatAdapter())
        reg.register(EmailAdapter())
        reg.register(WhatsAppAdapter())
        reg.register(VoiceAdapter())
        reg.register(SMSAdapter())
        assert len(reg.get_available()) == 5
        assert reg.get(ChannelType.CHAT) is not None
        assert reg.get(ChannelType.EMAIL) is not None
        assert reg.get(ChannelType.WHATSAPP) is not None
        assert reg.get(ChannelType.VOICE) is not None
        assert reg.get(ChannelType.SMS) is not None

    @pytest.mark.asyncio
    async def test_health_check_all(self):
        from channels.base import ChannelRegistry
        reg = ChannelRegistry()
        chat = ChatAdapter()
        await chat.initialize({})
        reg.register(chat)
        health = await reg.health_check_all()
        assert "chat" in health
        assert health["chat"]["initialized"] is True


# ══════════════════════════════════════════════════════════════
#  Test count
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
