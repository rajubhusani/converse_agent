"""
Tests for TelephonyFactory, provider-specific webhook normalization,
CallEndReason, and provider-switching integration.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from channels.telephony.factory import TelephonyFactory, TelephonyClient
from channels.telephony.twilio_client import TwilioClient
from channels.telephony.exotel import ExotelClient
from channels.telephony.plivo_client import PlivoClient
from channels.voice_adapter import (
    VoiceAdapter, CallStatus, CallDisposition, CallEndReason, VoiceCallState,
)
from voice.providers import TelephonyProvider, TelephonyConfig, QualityPreset


# ══════════════════════════════════════════════════════════════
#  TelephonyFactory Tests
# ══════════════════════════════════════════════════════════════

class TestTelephonyFactory:
    """Test factory creates correct client types."""

    def test_create_twilio(self):
        config = TelephonyConfig(
            provider=TelephonyProvider.TWILIO,
            account_sid="AC_test_sid",
            auth_token="test_token",
            phone_number="+14155551234",
        )
        client = TelephonyFactory.create(config)
        assert isinstance(client, TwilioClient)
        assert client.account_sid == "AC_test_sid"
        assert client.from_number == "+14155551234"

    def test_create_exotel(self):
        config = TelephonyConfig(
            provider=TelephonyProvider.EXOTEL,
            account_sid="exotel_sid",
            auth_token="exotel_token",
            phone_number="+919876543210",
        )
        client = TelephonyFactory.create(config)
        assert isinstance(client, ExotelClient)

    def test_create_plivo(self):
        config = TelephonyConfig(
            provider=TelephonyProvider.PLIVO,
            account_sid="plivo_id",
            auth_token="plivo_token",
            phone_number="+919876543210",
        )
        client = TelephonyFactory.create(config)
        assert isinstance(client, PlivoClient)

    def test_create_unsupported_raises(self):
        config = TelephonyConfig(provider=TelephonyProvider.DAILY)
        with pytest.raises(ValueError, match="Unsupported"):
            TelephonyFactory.create(config)

    def test_detect_twilio_webhook(self):
        payload = {"CallSid": "CA_abc", "CallStatus": "completed"}
        assert TelephonyFactory.detect_provider_from_webhook(payload) == TelephonyProvider.TWILIO

    def test_detect_exotel_webhook(self):
        payload = {"CallSid": "exo_abc", "Status": "completed"}
        assert TelephonyFactory.detect_provider_from_webhook(payload) == TelephonyProvider.EXOTEL

    def test_detect_plivo_webhook(self):
        payload = {"CallUUID": "plivo_abc", "Event": "hangup"}
        assert TelephonyFactory.detect_provider_from_webhook(payload) == TelephonyProvider.PLIVO

    def test_detect_unknown_returns_none(self):
        payload = {"random": "data"}
        assert TelephonyFactory.detect_provider_from_webhook(payload) is None

    def test_get_webhook_parser_twilio(self):
        parser = TelephonyFactory.get_webhook_parser(TelephonyProvider.TWILIO)
        assert parser == TwilioClient.parse_status_webhook

    def test_get_webhook_parser_exotel(self):
        parser = TelephonyFactory.get_webhook_parser(TelephonyProvider.EXOTEL)
        assert parser == ExotelClient.parse_status_webhook

    def test_get_webhook_parser_plivo(self):
        parser = TelephonyFactory.get_webhook_parser(TelephonyProvider.PLIVO)
        assert parser == PlivoClient.parse_status_webhook


# ══════════════════════════════════════════════════════════════
#  Twilio Webhook Parsing
# ══════════════════════════════════════════════════════════════

class TestTwilioWebhookParsing:

    def test_completed_call(self):
        payload = {
            "CallSid": "CA_abc123",
            "CallStatus": "completed",
            "Direction": "outbound-api",
            "From": "+14155551234",
            "To": "+919876543210",
            "CallDuration": "60",
        }
        result = TwilioClient.parse_status_webhook(payload)
        assert result["call_id"] == "CA_abc123"
        assert result["status"] == "completed"
        assert result["direction"] == "outbound"
        assert result["duration"] == 60
        assert result["from"] == "+14155551234"
        assert result["to"] == "+919876543210"

    def test_ringing_status(self):
        payload = {"CallSid": "CA_123", "CallStatus": "ringing", "Direction": "outbound-api"}
        result = TwilioClient.parse_status_webhook(payload)
        assert result["status"] == "ringing"

    def test_in_progress(self):
        payload = {"CallSid": "CA_123", "CallStatus": "in-progress"}
        result = TwilioClient.parse_status_webhook(payload)
        assert result["status"] == "in-progress"

    def test_no_answer(self):
        payload = {"CallSid": "CA_123", "CallStatus": "no-answer"}
        result = TwilioClient.parse_status_webhook(payload)
        assert result["status"] == "no-answer"

    def test_busy(self):
        payload = {"CallSid": "CA_123", "CallStatus": "busy"}
        result = TwilioClient.parse_status_webhook(payload)
        assert result["status"] == "busy"

    def test_failed(self):
        payload = {"CallSid": "CA_123", "CallStatus": "failed"}
        result = TwilioClient.parse_status_webhook(payload)
        assert result["status"] == "failed"

    def test_answered_by_machine(self):
        payload = {"CallSid": "CA_123", "CallStatus": "completed", "AnsweredBy": "machine_start"}
        result = TwilioClient.parse_status_webhook(payload)
        assert result["answered_by"] == "machine_start"

    def test_recording_url(self):
        payload = {
            "CallSid": "CA_123",
            "CallStatus": "completed",
            "RecordingUrl": "https://api.twilio.com/recordings/RE_abc",
        }
        result = TwilioClient.parse_status_webhook(payload)
        assert "twilio.com" in result["recording_url"]


# ══════════════════════════════════════════════════════════════
#  Exotel Webhook Parsing
# ══════════════════════════════════════════════════════════════

class TestExotelWebhookParsing:

    def test_completed_call(self):
        payload = {
            "CallSid": "exo_abc123",
            "Status": "completed",
            "Direction": "outbound",
            "From": "+919876543210",
            "To": "+919123456789",
            "Duration": "45",
        }
        result = ExotelClient.parse_status_webhook(payload)
        assert result["call_id"] == "exo_abc123"
        assert result["status"] == "completed"
        assert result["duration"] == 45

    def test_ringing(self):
        payload = {"CallSid": "exo_123", "Status": "ringing"}
        result = ExotelClient.parse_status_webhook(payload)
        assert result["status"] == "ringing"

    def test_in_progress(self):
        payload = {"CallSid": "exo_123", "Status": "in-progress"}
        result = ExotelClient.parse_status_webhook(payload)
        assert result["status"] == "in-progress"

    def test_lowercase_keys(self):
        payload = {"call_sid": "exo_123", "status": "completed"}
        result = ExotelClient.parse_status_webhook(payload)
        assert result["status"] == "completed"

    def test_no_answer(self):
        payload = {"CallSid": "exo_123", "Status": "no-answer"}
        result = ExotelClient.parse_status_webhook(payload)
        assert result["status"] == "no-answer"


# ══════════════════════════════════════════════════════════════
#  Plivo Webhook Parsing
# ══════════════════════════════════════════════════════════════

class TestPlivoWebhookParsing:

    def test_hangup_event(self):
        payload = {
            "CallUUID": "plivo_abc123",
            "Event": "hangup",
            "Direction": "outbound",
            "From": "+919876543210",
            "To": "+919123456789",
            "Duration": "30",
        }
        result = PlivoClient.parse_status_webhook(payload)
        assert result["call_id"] == "plivo_abc123"
        assert result["status"] == "completed"
        assert result["duration"] == 30

    def test_answer_event(self):
        payload = {"CallUUID": "plivo_123", "Event": "answer"}
        result = PlivoClient.parse_status_webhook(payload)
        assert result["status"] == "in-progress"

    def test_ring_event(self):
        payload = {"CallUUID": "plivo_123", "Event": "ring"}
        result = PlivoClient.parse_status_webhook(payload)
        assert result["status"] == "ringing"


# ══════════════════════════════════════════════════════════════
#  CallEndReason Tests
# ══════════════════════════════════════════════════════════════

class TestCallEndReason:

    def test_all_reasons_map_to_dispositions(self):
        """Every CallEndReason must map to a valid CallDisposition."""
        for reason in CallEndReason:
            disposition = reason.to_disposition()
            assert isinstance(disposition, CallDisposition)

    def test_completed_maps_correctly(self):
        assert CallEndReason.COMPLETED.to_disposition() == CallDisposition.COMPLETED

    def test_agent_hangup_maps_to_completed(self):
        assert CallEndReason.AGENT_HANGUP.to_disposition() == CallDisposition.COMPLETED

    def test_silence_timeout_maps_correctly(self):
        assert CallEndReason.SILENCE_TIMEOUT.to_disposition() == CallDisposition.SILENCE_TIMEOUT

    def test_error_maps_to_failed(self):
        assert CallEndReason.ERROR.to_disposition() == CallDisposition.FAILED

    def test_voicemail_maps_correctly(self):
        assert CallEndReason.VOICEMAIL.to_disposition() == CallDisposition.VOICEMAIL

    def test_string_construction(self):
        reason = CallEndReason("agent_hangup")
        assert reason == CallEndReason.AGENT_HANGUP


# ══════════════════════════════════════════════════════════════
#  Exotel Number Normalization
# ══════════════════════════════════════════════════════════════

class TestExotelNumberNormalization:

    def test_10_digit_indian(self):
        assert ExotelClient._normalize_indian_number("9876543210") == "+919876543210"

    def test_already_e164(self):
        assert ExotelClient._normalize_indian_number("+919876543210") == "+919876543210"

    def test_91_without_plus(self):
        assert ExotelClient._normalize_indian_number("919876543210") == "+919876543210"

    def test_with_spaces(self):
        assert ExotelClient._normalize_indian_number("98765 43210") == "+919876543210"

    def test_with_dashes(self):
        assert ExotelClient._normalize_indian_number("98765-43210") == "+919876543210"

    def test_non_indian_passthrough(self):
        assert ExotelClient._normalize_indian_number("+14155551234") == "+14155551234"

    def test_starts_with_6(self):
        assert ExotelClient._normalize_indian_number("6123456789") == "+916123456789"


# ══════════════════════════════════════════════════════════════
#  VoiceAdapter Integration Tests
# ══════════════════════════════════════════════════════════════

class TestVoiceAdapterProviderIntegration:

    @pytest.fixture
    def adapter(self):
        return VoiceAdapter()

    @pytest.mark.asyncio
    async def test_initialize_with_twilio_default(self, adapter):
        config = {
            "telephony_provider": "twilio",
            "telephony_account_sid": "AC_test",
            "telephony_auth_token": "test_token",
            "telephony_phone_number": "+14155551234",
            "stt_provider": "deepgram",
            "tts_provider": "cartesia",
        }
        await adapter.initialize(config)
        assert adapter._telephony_provider == "twilio"
        assert adapter._telephony_client is not None
        assert isinstance(adapter._telephony_client, TwilioClient)

    @pytest.mark.asyncio
    async def test_initialize_with_exotel(self, adapter):
        config = {
            "telephony_provider": "exotel",
            "telephony_account_sid": "exotel_sid",
            "telephony_auth_token": "exotel_token",
            "telephony_phone_number": "+919876543210",
        }
        await adapter.initialize(config)
        assert adapter._telephony_provider == "exotel"
        assert isinstance(adapter._telephony_client, ExotelClient)

    @pytest.mark.asyncio
    async def test_initialize_without_credentials_stubs(self, adapter):
        config = {
            "telephony_provider": "twilio",
            # No credentials — should enter stub mode
        }
        await adapter.initialize(config)
        assert adapter._initialized is True
        # No credentials → stub mode (client is None, calls won't reach PSTN)
        assert adapter._telephony_client is None
        assert adapter._telephony_provider == "twilio"

    @pytest.mark.asyncio
    async def test_shutdown_closes_telephony_client(self, adapter):
        config = {
            "telephony_provider": "twilio",
            "telephony_account_sid": "test",
            "telephony_auth_token": "test",
            "telephony_phone_number": "+1234567890",
        }
        await adapter.initialize(config)
        await adapter.shutdown()
        # Should not raise

    @pytest.mark.asyncio
    async def test_handle_call_status_from_twilio_webhook(self, adapter):
        await adapter.initialize({
            "telephony_provider": "twilio",
            "telephony_account_sid": "test",
            "telephony_auth_token": "test",
            "telephony_phone_number": "+1234567890",
        })

        # Create an active call state manually
        from models.schemas import Contact
        contact = Contact(id="c1", name="Test Dealer", phone="+919876543210")
        result = await adapter._initiate_outbound_call(
            contact, "Hello!", {"conversation_id": "conv1"}
        )
        call_id = result["call_id"]

        # Simulate Twilio webhook
        status_result = await adapter.handle_call_status(
            call_id, "in-progress", {}
        )
        call_state = adapter.get_active_call(call_id)
        assert call_state is not None
        assert call_state.status == CallStatus.CONNECTED


# ══════════════════════════════════════════════════════════════
#  Quality Preset Tests
# ══════════════════════════════════════════════════════════════

class TestCostOptimizedPreset:

    def test_cost_optimized_exists(self):
        from voice.providers import QUALITY_PRESETS
        assert QualityPreset.COST_OPTIMIZED in QUALITY_PRESETS

    def test_cost_optimized_uses_deepgram(self):
        from voice.providers import QUALITY_PRESETS, STTProvider
        preset = QUALITY_PRESETS[QualityPreset.COST_OPTIMIZED]
        assert preset["stt"]["provider"] == STTProvider.DEEPGRAM

    def test_cost_optimized_uses_cartesia(self):
        from voice.providers import QUALITY_PRESETS, TTSProvider
        preset = QUALITY_PRESETS[QualityPreset.COST_OPTIMIZED]
        assert preset["tts"]["provider"] == TTSProvider.CARTESIA

    def test_cost_optimized_uses_haiku(self):
        from voice.providers import QUALITY_PRESETS
        preset = QUALITY_PRESETS[QualityPreset.COST_OPTIMIZED]
        assert "haiku" in preset["llm"]["model"]

    def test_cost_optimized_target_latency(self):
        from voice.providers import QUALITY_PRESETS
        preset = QUALITY_PRESETS[QualityPreset.COST_OPTIMIZED]
        assert preset["pipeline"]["target_latency_ms"] == 400

    def test_load_cost_optimized_preset(self):
        from voice.providers import VoiceProviderRegistry
        registry = VoiceProviderRegistry()
        registry.load_preset(QualityPreset.COST_OPTIMIZED)
        stt = registry.get_stt()
        tts = registry.get_tts()
        assert stt is not None
        assert tts is not None
        assert stt.provider.value == "deepgram"
        assert tts.provider.value == "cartesia"
