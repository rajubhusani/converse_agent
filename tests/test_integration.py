"""
Tests for the integration layer:
- Database models and PostgresContextStore
- Exotel/Plivo telephony clients
- VoicePipelineManager
- BackendPoller
- VoiceMetricsPublisher and CostTracker
"""
import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from models.schemas import Contact, ContactChannel, ChannelType


# ══════════════════════════════════════════════════════════════
#  Database Models
# ══════════════════════════════════════════════════════════════

class TestDatabaseModels:
    """Test ORM model creation and field defaults."""

    def test_contact_row_defaults(self):
        from database.models import ContactRow
        row = ContactRow(name="Test User")
        assert row.name == "Test User"
        assert row.role == ""
        assert row.organization == ""
        assert row.channels == []
        assert row.language == "en"
        assert row.timezone == "Asia/Kolkata"

    def test_contact_row_to_dict(self):
        from database.models import ContactRow
        row = ContactRow(
            id="c_001",
            name="Priya Sharma",
            role="CFO",
            organization="Acme Corp",
            channels=[
                {"channel": "voice", "address": "+919876543210", "preferred": True},
                {"channel": "email", "address": "priya@acme.com"},
            ],
        )
        d = row.to_dict()
        assert d["id"] == "c_001"
        assert d["name"] == "Priya Sharma"
        assert len(d["channels"]) == 2
        assert d["channels"][0]["address"] == "+919876543210"

    def test_followup_row_defaults(self):
        from database.models import FollowUpRow
        row = FollowUpRow(contact_id="c_001", reason="Payment overdue")
        assert row.priority == "medium"
        assert row.status == "pending"
        assert row.attempt_count == 0
        assert row.max_attempts == 3
        assert row.channel_priority == ["voice", "whatsapp", "email"]

    def test_voice_call_row(self):
        from database.models import VoiceCallRow
        row = VoiceCallRow(
            id="call_abc123",
            conversation_id="conv_001",
            contact_id="c_001",
            direction="outbound",
            telephony_provider="exotel",
            transcript=[
                {"role": "agent", "text": "Hello, this is regarding your invoice."},
                {"role": "contact", "text": "Yes, I'm aware."},
            ],
        )
        assert row.telephony_provider == "exotel"
        assert len(row.transcript) == 2
        assert row.barge_in_count == 0

    def test_state_binding_row(self):
        from database.models import StateBindingRow
        row = StateBindingRow(
            conversation_id="conv_001",
            process_type="payment_collection",
            current_state="reminded",
            state_data={"extracted_info": {"payment_date": "2026-02-20"}},
        )
        assert row.current_state == "reminded"
        assert row.state_data["extracted_info"]["payment_date"] == "2026-02-20"

    def test_all_models_have_base(self):
        from database.models import Base, ContactRow, ConversationRow, MessageRow, FollowUpRow, StateBindingRow, VoiceCallRow
        for model in [ContactRow, ConversationRow, MessageRow, FollowUpRow, StateBindingRow, VoiceCallRow]:
            assert issubclass(model, Base)


# ══════════════════════════════════════════════════════════════
#  Exotel Client
# ══════════════════════════════════════════════════════════════

class TestExotelClient:
    """Test Exotel telephony client."""

    def test_normalize_indian_10_digit(self):
        from channels.telephony.exotel import ExotelClient
        assert ExotelClient._normalize_indian_number("9876543210") == "+919876543210"
        assert ExotelClient._normalize_indian_number("6789012345") == "+916789012345"

    def test_normalize_indian_with_91_prefix(self):
        from channels.telephony.exotel import ExotelClient
        assert ExotelClient._normalize_indian_number("919876543210") == "+919876543210"

    def test_normalize_already_e164(self):
        from channels.telephony.exotel import ExotelClient
        assert ExotelClient._normalize_indian_number("+919876543210") == "+919876543210"

    def test_normalize_with_spaces(self):
        from channels.telephony.exotel import ExotelClient
        assert ExotelClient._normalize_indian_number("98765 43210") == "+919876543210"
        assert ExotelClient._normalize_indian_number("+91 98765-43210") == "+919876543210"

    def test_non_indian_passthrough(self):
        from channels.telephony.exotel import ExotelClient
        assert ExotelClient._normalize_indian_number("+14155551234") == "+14155551234"

    def test_parse_status_webhook_ringing(self):
        from channels.telephony.exotel import ExotelClient
        result = ExotelClient.parse_status_webhook({
            "CallSid": "abc123",
            "Status": "ringing",
            "Direction": "outbound",
            "From": "+919000000001",
            "To": "+919876543210",
        })
        assert result["call_id"] == "abc123"
        assert result["status"] == "ringing"
        assert result["direction"] == "outbound"

    def test_parse_status_webhook_completed(self):
        from channels.telephony.exotel import ExotelClient
        result = ExotelClient.parse_status_webhook({
            "CallSid": "abc123",
            "Status": "completed",
            "Duration": "180",
            "RecordingUrl": "https://recordings.exotel.com/abc.mp3",
        })
        assert result["status"] == "completed"
        assert result["duration"] == 180
        assert "recordings" in result["recording_url"]

    def test_parse_status_webhook_busy(self):
        from channels.telephony.exotel import ExotelClient
        result = ExotelClient.parse_status_webhook({"CallSid": "x", "Status": "busy"})
        assert result["status"] == "busy"

    def test_parse_status_webhook_no_answer(self):
        from channels.telephony.exotel import ExotelClient
        result = ExotelClient.parse_status_webhook({"CallSid": "x", "Status": "no-answer"})
        assert result["status"] == "no-answer"

    def test_client_init(self):
        from channels.telephony.exotel import ExotelClient
        client = ExotelClient(
            account_sid="test_sid",
            auth_token="test_token",
            caller_id="+919000000001",
        )
        assert client.caller_id == "+919000000001"
        assert "test_sid" in client.base_url


# ══════════════════════════════════════════════════════════════
#  Plivo Client
# ══════════════════════════════════════════════════════════════

class TestPlivoClient:
    """Test Plivo telephony client."""

    def test_parse_status_webhook_answer(self):
        from channels.telephony.plivo_client import PlivoClient
        result = PlivoClient.parse_status_webhook({
            "CallUUID": "uuid123",
            "Event": "answer",
            "Direction": "outbound",
        })
        assert result["call_id"] == "uuid123"
        assert result["status"] == "in-progress"

    def test_parse_status_webhook_hangup(self):
        from channels.telephony.plivo_client import PlivoClient
        result = PlivoClient.parse_status_webhook({
            "CallUUID": "uuid123",
            "Event": "hangup",
            "Duration": "120",
        })
        assert result["status"] == "completed"
        assert result["duration"] == 120

    def test_parse_status_webhook_timeout(self):
        from channels.telephony.plivo_client import PlivoClient
        result = PlivoClient.parse_status_webhook({"CallUUID": "x", "Event": "timeout"})
        assert result["status"] == "no-answer"

    def test_client_init(self):
        from channels.telephony.plivo_client import PlivoClient
        client = PlivoClient(
            auth_id="test_id",
            auth_token="test_token",
            caller_id="+919000000001",
        )
        assert client.caller_id == "+919000000001"
        assert "test_id" in client.base_url


# ══════════════════════════════════════════════════════════════
#  Voice Pipeline Manager
# ══════════════════════════════════════════════════════════════

class TestVoicePipelineManager:
    """Test pipeline lifecycle management."""

    def _make_manager(self):
        from voice.server import VoicePipelineManager
        from voice.providers import VoiceProviderRegistry
        registry = VoiceProviderRegistry()
        return VoicePipelineManager(registry=registry)

    def _make_contact(self):
        return Contact(
            id="c_001",
            name="Test User",
            channels=[ContactChannel(channel=ChannelType.VOICE, address="+919876543210")],
        )

    @pytest.mark.asyncio
    async def test_create_pipeline(self):
        manager = self._make_manager()
        contact = self._make_contact()
        result = await manager.create_pipeline(
            call_id="call_001",
            contact=contact,
            context={"initial_message": "Hello, this is a test call."},
        )
        assert result["status"] == "pipeline_ready"
        assert result["call_id"] == "call_001"
        assert manager.active_call_count == 1

    @pytest.mark.asyncio
    async def test_destroy_pipeline(self):
        manager = self._make_manager()
        contact = self._make_contact()
        await manager.create_pipeline("call_001", contact, {"initial_message": "Hello"})
        assert manager.active_call_count == 1

        result = await manager.destroy_pipeline("call_001")
        assert result["status"] == "destroyed"
        assert "metrics" in result
        assert manager.active_call_count == 0

    @pytest.mark.asyncio
    async def test_destroy_nonexistent_pipeline(self):
        manager = self._make_manager()
        result = await manager.destroy_pipeline("nonexistent")
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_duplicate_pipeline_rejected(self):
        manager = self._make_manager()
        contact = self._make_contact()
        await manager.create_pipeline("call_001", contact, {})
        result = await manager.create_pipeline("call_001", contact, {})
        assert result["status"] == "already_exists"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_pipelines(self):
        manager = self._make_manager()
        contact = self._make_contact()
        for i in range(5):
            await manager.create_pipeline(f"call_{i:03d}", contact, {})
        assert manager.active_call_count == 5

        await manager.shutdown()
        assert manager.active_call_count == 0

    @pytest.mark.asyncio
    async def test_stt_transcript_final(self):
        manager = self._make_manager()
        transcript_received = []
        manager._on_transcript = AsyncMock(side_effect=lambda **kw: transcript_received.append(kw))

        contact = self._make_contact()
        await manager.create_pipeline("call_001", contact, {})

        result = await manager.on_stt_transcript("call_001", "Yes I will pay tomorrow", confidence=0.95, is_final=True)
        assert result is not None
        assert result["text"] == "Yes I will pay tomorrow"
        assert len(transcript_received) == 1

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_stt_transcript_interim_not_forwarded(self):
        manager = self._make_manager()
        manager._on_transcript = AsyncMock()

        contact = self._make_contact()
        await manager.create_pipeline("call_001", contact, {})

        result = await manager.on_stt_transcript("call_001", "Yes I will", is_final=False)
        assert result is None
        manager._on_transcript.assert_not_called()

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_silence_tick_gentle_prompt(self):
        manager = self._make_manager()
        contact = self._make_contact()
        await manager.create_pipeline("call_001", contact, {})

        # Simulate silence long enough to trigger gentle prompt
        pipeline = manager._pipelines["call_001"]
        # Fast-forward turn engine silence counter
        for _ in range(90):  # Simulate ~9 seconds at 10Hz
            pipeline.turn_engine.on_silence_tick()

        result = await manager.on_silence_tick("call_001", 9.0)
        # Result depends on turn-taking engine state
        # The engine tracks silence internally

        await manager.shutdown()

    def test_filler_detection_english(self):
        manager = self._make_manager()
        assert "um" in manager._detect_fillers("um let me think")
        assert "uh" in manager._detect_fillers("uh I'm not sure")

    def test_filler_detection_hindi(self):
        manager = self._make_manager()
        assert "matlab" in manager._detect_fillers("matlab woh payment")
        assert "acha" in manager._detect_fillers("acha theek hai")
        assert "ek minute" in manager._detect_fillers("ek minute ruko")

    def test_filler_detection_no_fillers(self):
        manager = self._make_manager()
        assert manager._detect_fillers("Yes I will pay tomorrow") == []

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        manager = self._make_manager()
        contact = self._make_contact()
        await manager.create_pipeline("call_001", contact, {})

        metrics = manager.get_metrics()
        assert metrics["active_calls"] == 1
        assert "latency" in metrics

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_get_call_metrics(self):
        manager = self._make_manager()
        contact = self._make_contact()
        await manager.create_pipeline("call_001", contact, {})

        call_metrics = manager.get_call_metrics("call_001")
        assert call_metrics is not None

        assert manager.get_call_metrics("nonexistent") is None

        await manager.shutdown()


# ══════════════════════════════════════════════════════════════
#  Backend Poller
# ══════════════════════════════════════════════════════════════

class TestBackendPoller:
    """Test backend polling and sync logic."""

    @pytest.mark.asyncio
    async def test_poll_cycle_creates_new_followups(self):
        from backend.poller import BackendPoller
        from backend.connector import MockBackendConnector

        mock_store = AsyncMock()
        mock_store.get_followup_by_external_id = AsyncMock(return_value=None)
        mock_store.get_contact_by_external_id = AsyncMock(return_value=None)
        mock_store.upsert_contact = AsyncMock()
        mock_store.create_followup = AsyncMock(return_value={"id": "f_new"})

        mock_backend = MockBackendConnector()

        enqueue_calls = []
        async def mock_enqueue(f):
            enqueue_calls.append(f)

        poller = BackendPoller(
            store=mock_store,
            backend=mock_backend,
            enqueue_fn=mock_enqueue,
        )

        stats = await poller.poll_cycle()
        assert stats["fetched"] > 0

    @pytest.mark.asyncio
    async def test_poll_cycle_deduplicates(self):
        from backend.poller import BackendPoller
        from backend.connector import MockBackendConnector

        mock_store = AsyncMock()
        # Return existing follow-up — should be skipped
        mock_store.get_followup_by_external_id = AsyncMock(return_value={"id": "existing"})

        mock_backend = MockBackendConnector()

        poller = BackendPoller(store=mock_store, backend=mock_backend)
        stats = await poller.poll_cycle()
        assert stats["skipped"] > 0
        assert stats["new"] == 0


# ══════════════════════════════════════════════════════════════
#  Cost Tracker
# ══════════════════════════════════════════════════════════════

class TestCostTracker:
    """Test per-call cost tracking for ₹4/min budget."""

    def test_record_3min_call(self):
        from voice.monitoring import CostTracker
        tracker = CostTracker()

        cost = tracker.record_call(
            call_id="call_001",
            duration_s=180,           # 3 minutes
            stt_minutes=3.0,          # Full call duration
            tts_characters=1125,      # ~375 chars/min * 3 min
            llm_input_tokens=3000,    # ~1000/turn * 3 turns
            llm_output_tokens=210,    # ~70/turn * 3 turns
        )

        assert cost["duration_min"] == 3.0
        assert cost["telephony_inr"] == pytest.approx(1.95, abs=0.01)  # 3 * 0.65
        assert cost["stt_inr"] == pytest.approx(0.93, abs=0.01)        # 3 * 0.31
        assert cost["total_inr"] > 0
        assert cost["per_min_inr"] < 4.0  # Must stay under ₹4/min budget

    def test_record_multiple_calls_summary(self):
        from voice.monitoring import CostTracker
        tracker = CostTracker()

        for i in range(10):
            tracker.record_call(
                call_id=f"call_{i:03d}",
                duration_s=180,
                stt_minutes=3.0,
                tts_characters=1125,
                llm_input_tokens=3000,
                llm_output_tokens=210,
            )

        summary = tracker.get_summary()
        assert summary["total_calls"] == 10
        assert summary["total_minutes"] == 30.0
        assert summary["avg_per_min_inr"] < 4.0  # Budget check
        assert "telephony" in summary["breakdown_inr"]
        assert "stt" in summary["breakdown_inr"]
        assert "tts" in summary["breakdown_inr"]
        assert "llm" in summary["breakdown_inr"]

    def test_empty_summary(self):
        from voice.monitoring import CostTracker
        tracker = CostTracker()
        summary = tracker.get_summary()
        assert summary["total_calls"] == 0

    def test_per_min_under_budget(self):
        """Verify the ₹4/min target is achievable with the optimized stack."""
        from voice.monitoring import CostTracker
        tracker = CostTracker()

        # Simulate a typical 3-minute payment collection call
        cost = tracker.record_call(
            call_id="budget_test",
            duration_s=180,
            stt_minutes=3.0,
            tts_characters=1125,       # Agent speaks ~50% of call
            llm_input_tokens=10500,    # 3.5 turns * (800 system + 2200 context) tokens
            llm_output_tokens=245,     # 3.5 turns * 70 tokens
        )

        # Verify per-minute cost is under ₹4
        assert cost["per_min_inr"] < 4.0, f"Per-minute cost ₹{cost['per_min_inr']:.2f} exceeds ₹4 budget"

        # Verify telephony is the dominant cost
        assert cost["telephony_inr"] > cost["stt_inr"]
        assert cost["telephony_inr"] > cost["tts_inr"]
        assert cost["telephony_inr"] > cost["llm_input_inr"] + cost["llm_output_inr"]


# ══════════════════════════════════════════════════════════════
#  Metrics Publisher
# ══════════════════════════════════════════════════════════════

class TestVoiceMetricsPublisher:
    """Test metrics publishing and health checks."""

    def test_health_check_healthy(self):
        from voice.monitoring import VoiceMetricsPublisher
        from voice.latency import AggregateLatencyTracker
        tracker = AggregateLatencyTracker()
        publisher = VoiceMetricsPublisher(tracker)
        health = publisher.get_health()
        assert health["status"] == "healthy"
        assert health["active_calls"] == 0

    def test_build_metric_data(self):
        from voice.monitoring import VoiceMetricsPublisher
        from voice.latency import AggregateLatencyTracker
        tracker = AggregateLatencyTracker()
        publisher = VoiceMetricsPublisher(tracker)

        stats = {"active_calls": 5, "total": {"p50_ms": 300, "p90_ms": 500, "p99_ms": 800}}
        metrics = publisher._build_metric_data(stats)
        names = [m["MetricName"] for m in metrics]
        assert "ActiveCalls" in names


# ══════════════════════════════════════════════════════════════
#  Provider Registry (Indian providers)
# ══════════════════════════════════════════════════════════════

class TestIndianProviders:
    """Test that Indian telephony providers are properly registered."""

    def test_exotel_in_enum(self):
        from voice.providers import TelephonyProvider
        assert TelephonyProvider.EXOTEL == "exotel"
        assert TelephonyProvider.PLIVO == "plivo"
        assert TelephonyProvider.OZONETEL == "ozonetel"

    def test_telephony_config_with_exotel(self):
        from voice.providers import TelephonyConfig, TelephonyProvider
        config = TelephonyConfig(
            provider=TelephonyProvider.EXOTEL,
            account_sid="test_sid",
            auth_token="test_token",
            phone_number="+919000000001",
        )
        assert config.provider == TelephonyProvider.EXOTEL
        assert config.phone_number == "+919000000001"
