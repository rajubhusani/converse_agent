"""
Tests for Voice Subsystem — providers, turn-taking, latency, pipeline.

Coverage:
- Provider registry: STT/TTS/Telephony config, presets, latency estimates
- Turn-taking engine: floor management, barge-in, backchannels, silence escalation
- Latency tracker: per-stage timing, budgets, violations, optimization hints
- Pipeline builder: full pipeline generation, prompt naturalness, adaptive quality
- Naturalness helpers: filler detection, sentence completion, question detection
"""
import time
import pytest
from unittest.mock import MagicMock

from models.schemas import Contact, ContactChannel, ChannelType

# ── Providers ─────────────────────────────────────────────────
from voice.providers import (
    VoiceProviderRegistry,
    STTProvider, TTSProvider, TelephonyProvider,
    STTConfig, TTSConfig, TelephonyConfig,
    QualityPreset, AudioCodec, LatencyProfile,
    QUALITY_PRESETS,
)

# ── Turn-taking ───────────────────────────────────────────────
from voice.turn_taking import (
    TurnTakingEngine,
    FloorHolder, ConversationPhase, SilenceLevel,
    BackchannelConfig, EndpointingConfig, SilenceConfig,
    InterruptionConfig, InterruptionStrategy,
    detect_fillers, is_likely_complete, is_question,
    FILLER_WORDS,
)

# ── Latency ───────────────────────────────────────────────────
from voice.latency import (
    PipelineStage, LatencyBudget,
    CallLatencyTracker, AggregateLatencyTracker, StageTracker,
)

# ── Pipeline ──────────────────────────────────────────────────
from voice.pipeline import (
    VoicePipelineBuilder, VoicePromptBuilder, PipelineConfig,
    create_turn_taking_engine,
)


# ══════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def contact():
    return Contact(
        id="c1",
        name="Priya Sharma",
        role="CFO",
        organization="Acme Corp",
        channels=[ContactChannel(channel=ChannelType.VOICE, address="+919876543210")],
    )


@pytest.fixture
def registry():
    reg = VoiceProviderRegistry()
    reg.register_stt("default", STTConfig(
        provider=STTProvider.DEEPGRAM,
        model="nova-2",
        endpointing_ms=250,
        latency=LatencyProfile(time_to_first_byte_ms=100, processing_per_second_ms=40),
    ), default=True)
    reg.register_tts("default", TTSConfig(
        provider=TTSProvider.ELEVENLABS,
        model="eleven_turbo_v2_5",
        voice_id="voice_abc",
        latency=LatencyProfile(time_to_first_byte_ms=150, processing_per_second_ms=30),
    ), default=True)
    reg.set_telephony(TelephonyConfig(
        provider=TelephonyProvider.TWILIO,
        phone_number="+14155551234",
    ))
    return reg


@pytest.fixture
def turn_engine():
    return TurnTakingEngine()


@pytest.fixture
def call_context():
    return {
        "business_context": {
            "invoice": "INV-301",
            "amount": 52400,
            "currency": "INR",
            "due_date": "2026-02-20",
        },
        "followup": {"reason": "Payment reminder for overdue invoice"},
        "conversation_history": [
            {"role": "agent", "content": "Sent email reminder 3 days ago"},
            {"role": "contact", "content": "Acknowledged but no payment yet"},
        ],
    }


# ══════════════════════════════════════════════════════════════
#  PROVIDER REGISTRY TESTS
# ══════════════════════════════════════════════════════════════

class TestSTTConfig:
    def test_deepgram_params(self):
        cfg = STTConfig(provider=STTProvider.DEEPGRAM, model="nova-2", language="en")
        params = cfg.to_provider_params()
        assert params["model"] == "nova-2"
        assert params["language"] == "en"
        assert params["interim_results"] is True
        assert params["vad_events"] is True
        assert params["smart_format"] is True
        assert params["filler_words"] is True
        assert "endpointing" in params

    def test_deepgram_with_keywords(self):
        cfg = STTConfig(
            provider=STTProvider.DEEPGRAM,
            keywords=["invoice", "payment", "Acme"],
            keyword_boost=2.0,
        )
        params = cfg.to_provider_params()
        assert len(params["keywords"]) == 3
        assert "invoice:2.0" in params["keywords"]

    def test_assemblyai_params(self):
        cfg = STTConfig(provider=STTProvider.ASSEMBLYAI, model="best")
        params = cfg.to_provider_params()
        assert params["speech_model"] == "best"
        assert params["punctuate"] is True

    def test_google_params(self):
        cfg = STTConfig(provider=STTProvider.GOOGLE, keywords=["payment"])
        params = cfg.to_provider_params()
        assert params["enable_automatic_punctuation"] is True
        assert len(params["speech_contexts"]) == 1

    def test_azure_params(self):
        cfg = STTConfig(provider=STTProvider.AZURE, endpointing_ms=200)
        params = cfg.to_provider_params()
        assert params["end_silence_timeout_ms"] == "200"

    def test_whisper_params(self):
        cfg = STTConfig(provider=STTProvider.WHISPER)
        params = cfg.to_provider_params()
        assert params["model"] == "whisper-1"
        assert params["temperature"] == 0.0

    def test_multichannel_deepgram(self):
        cfg = STTConfig(provider=STTProvider.DEEPGRAM, multichannel=True, diarize=True)
        params = cfg.to_provider_params()
        assert params["multichannel"] is True
        assert params["diarize"] is True


class TestTTSConfig:
    def test_elevenlabs_params(self):
        cfg = TTSConfig(
            provider=TTSProvider.ELEVENLABS,
            voice_id="voice_123",
            model="eleven_turbo_v2_5",
            stability=0.6,
            similarity_boost=0.8,
        )
        params = cfg.to_provider_params()
        assert params["voice_id"] == "voice_123"
        assert params["model_id"] == "eleven_turbo_v2_5"
        assert params["voice_settings"]["stability"] == 0.6
        assert params["optimize_streaming_latency"] == 4

    def test_cartesia_params(self):
        cfg = TTSConfig(provider=TTSProvider.CARTESIA, voice_id="sonic_voice")
        params = cfg.to_provider_params()
        assert params["model_id"] == "sonic-english"
        assert params["voice"]["mode"] == "id"
        assert params["output_format"]["encoding"] == "pcm_s16le"

    def test_playht_params(self):
        cfg = TTSConfig(provider=TTSProvider.PLAYHT, speed=1.2)
        params = cfg.to_provider_params()
        assert params["speed"] == 1.2
        assert params["output_format"] == "raw"

    def test_azure_tts_params(self):
        cfg = TTSConfig(provider=TTSProvider.AZURE, speed=1.1, pitch=2.0)
        params = cfg.to_provider_params()
        assert params["voice_name"] == "en-US-JennyNeural"
        assert "110%" in params["speaking_rate"]

    def test_google_tts_params(self):
        cfg = TTSConfig(provider=TTSProvider.GOOGLE)
        params = cfg.to_provider_params()
        assert params["voice"]["language_code"] == "en-US"
        assert params["audio_config"]["audio_encoding"] == "LINEAR16"

    def test_xtts_local_params(self):
        cfg = TTSConfig(provider=TTSProvider.XTTS, voice_id="/data/speaker.wav")
        params = cfg.to_provider_params()
        assert params["model"] == "xtts_v2"
        assert params["speaker_wav"] == "/data/speaker.wav"
        assert params["stream"] is True

    def test_elevenlabs_format_mapping(self):
        for codec, expected in [
            (AudioCodec.PCM_16K, "pcm_16000"),
            (AudioCodec.PCM_24K, "pcm_24000"),
            (AudioCodec.MULAW_8K, "ulaw_8000"),
            (AudioCodec.MP3, "mp3_44100_128"),
        ]:
            cfg = TTSConfig(provider=TTSProvider.ELEVENLABS, output_format=codec)
            assert cfg.to_provider_params()["output_format"] == expected


class TestProviderRegistry:
    def test_register_and_retrieve_stt(self, registry):
        stt = registry.get_stt()
        assert stt is not None
        assert stt.provider == STTProvider.DEEPGRAM
        assert stt.model == "nova-2"

    def test_register_and_retrieve_tts(self, registry):
        tts = registry.get_tts()
        assert tts.provider == TTSProvider.ELEVENLABS
        assert tts.voice_id == "voice_abc"

    def test_multiple_stt_profiles(self, registry):
        registry.register_stt("fast", STTConfig(
            provider=STTProvider.DEEPGRAM, model="nova-2",
            endpointing_ms=150,
            latency=LatencyProfile(80, 30),
        ))
        registry.register_stt("accurate", STTConfig(
            provider=STTProvider.ASSEMBLYAI, model="best",
            latency=LatencyProfile(200, 60),
        ))
        assert registry.get_stt("fast").endpointing_ms == 150
        assert registry.get_stt("accurate").provider == STTProvider.ASSEMBLYAI

    def test_multiple_tts_profiles(self, registry):
        registry.register_tts("cartesia_fast", TTSConfig(
            provider=TTSProvider.CARTESIA,
            latency=LatencyProfile(60, 20),
        ))
        assert registry.get_tts("cartesia_fast").provider == TTSProvider.CARTESIA

    def test_telephony_config(self, registry):
        tel = registry.get_telephony()
        assert tel.provider == TelephonyProvider.TWILIO
        assert tel.phone_number == "+14155551234"

    def test_estimated_pipeline_latency(self, registry):
        latency = registry.estimated_pipeline_latency_ms()
        # 100 (stt) + 200 (llm est) + 150 (tts) = 450
        assert latency == 450

    def test_list_providers(self, registry):
        stt_list = registry.list_stt_providers()
        assert len(stt_list) == 1
        assert stt_list[0]["provider"] == "deepgram"
        tts_list = registry.list_tts_providers()
        assert len(tts_list) == 1
        assert tts_list[0]["voice_id"] == "voice_abc"


class TestQualityPresets:
    def test_all_presets_exist(self):
        for preset in QualityPreset:
            assert preset in QUALITY_PRESETS

    def test_ultra_low_latency_preset(self):
        config = QUALITY_PRESETS[QualityPreset.ULTRA_LOW_LATENCY]
        assert config["stt"]["endpointing_ms"] == 150
        assert config["tts"]["provider"] == TTSProvider.CARTESIA
        assert config["pipeline"]["target_latency_ms"] == 300
        assert config["pipeline"]["llm_sentence_streaming"] is True
        assert config["pipeline"]["prefetch_greeting"] is True

    def test_low_latency_preset(self):
        config = QUALITY_PRESETS[QualityPreset.LOW_LATENCY]
        assert config["tts"]["provider"] == TTSProvider.ELEVENLABS
        assert config["tts"]["optimize_streaming_latency"] == 4

    def test_high_quality_preset(self):
        config = QUALITY_PRESETS[QualityPreset.HIGH_QUALITY]
        assert config["stt"]["diarize"] is True
        assert config["tts"]["optimize_streaming_latency"] == 0
        assert config["pipeline"]["llm_sentence_streaming"] is False

    def test_load_preset_into_registry(self):
        reg = VoiceProviderRegistry()
        reg.load_preset(QualityPreset.LOW_LATENCY, api_keys={"stt": "key1", "tts": "key2"})
        stt = reg.get_stt()
        tts = reg.get_tts()
        assert stt is not None
        assert tts is not None
        assert stt.api_key == "key1"
        assert tts.api_key == "key2"

    def test_latency_profile_round_trip(self):
        lp = LatencyProfile(time_to_first_byte_ms=80, processing_per_second_ms=30)
        # 3-second utterance: 80 + 30*3 = 170ms
        assert lp.estimated_round_trip_ms == 170


# ══════════════════════════════════════════════════════════════
#  NATURALNESS HELPERS TESTS
# ══════════════════════════════════════════════════════════════

class TestFillerDetection:
    def test_detect_common_fillers(self):
        assert "um" in detect_fillers("um I think so")
        assert "uh" in detect_fillers("uh let me check")
        assert "like" in detect_fillers("it was like really good")

    def test_no_fillers(self):
        assert detect_fillers("Yes I agree completely") == []

    def test_multiple_fillers(self):
        found = detect_fillers("um well you know I think so")
        assert len(found) >= 2

    def test_no_false_positives(self):
        """'ah' should not match 'ahead', 'er' should not match 'there'."""
        assert detect_fillers("Go ahead with the payment") == []
        assert detect_fillers("I was there yesterday") == []
        assert detect_fillers("The farmer was working") == []

    def test_empty_string(self):
        assert detect_fillers("") == []


class TestSentenceCompletion:
    def test_period_ending(self):
        assert is_likely_complete("I'll pay tomorrow.") is True

    def test_question_mark(self):
        assert is_likely_complete("Can you send the invoice?") is True

    def test_exclamation(self):
        assert is_likely_complete("That sounds great!") is True

    def test_incomplete_sentence(self):
        assert is_likely_complete("I was thinking about") is False

    def test_short_affirmative(self):
        assert is_likely_complete("yes") is True
        assert is_likely_complete("okay") is True
        assert is_likely_complete("thanks") is True
        assert is_likely_complete("nope") is True

    def test_empty(self):
        assert is_likely_complete("") is False


class TestQuestionDetection:
    def test_question_mark(self):
        assert is_question("How much do I owe?") is True

    def test_question_word_start(self):
        assert is_question("What is the due date") is True
        assert is_question("When should I pay") is True
        assert is_question("How can I transfer") is True

    def test_auxiliary_verb_question(self):
        assert is_question("Can you send the details") is True
        assert is_question("Do you accept card payments") is True
        assert is_question("Is there a late fee") is True

    def test_statement(self):
        assert is_question("I will pay tomorrow") is False
        assert is_question("The amount is correct") is False


# ══════════════════════════════════════════════════════════════
#  TURN-TAKING ENGINE TESTS
# ══════════════════════════════════════════════════════════════

class TestTurnTakingFloor:
    def test_initial_state(self, turn_engine):
        assert turn_engine.floor == FloorHolder.AGENT
        assert turn_engine.phase == ConversationPhase.GREETING

    def test_agent_speaks(self, turn_engine):
        turn_engine.on_agent_speech_start()
        assert turn_engine.floor == FloorHolder.AGENT

    def test_agent_finishes_yields_floor(self, turn_engine):
        turn_engine.on_agent_speech_start()
        turn_engine.on_agent_speech_end("How are you?")
        assert turn_engine.floor == FloorHolder.YIELDED

    def test_contact_takes_floor(self, turn_engine):
        turn_engine.on_agent_speech_end("Hello")
        result = turn_engine.on_contact_speech_start()
        assert turn_engine.floor == FloorHolder.CONTACT
        assert result["action"] == "listen"

    def test_contact_finishes_transitions(self, turn_engine):
        turn_engine.on_agent_speech_end("Hello")
        turn_engine.on_contact_speech_start()
        result = turn_engine.on_contact_speech_end("I'm good, thanks.")
        assert turn_engine.floor == FloorHolder.TRANSITIONING
        assert "endpointing_ms" in result

    def test_phase_change(self, turn_engine):
        turn_engine.set_phase(ConversationPhase.BODY)
        assert turn_engine.phase == ConversationPhase.BODY


class TestBargeIn:
    def test_barge_in_during_agent_speech(self, turn_engine):
        turn_engine.on_agent_speech_start()
        result = turn_engine.on_contact_speech_start()
        assert result["action"] == "barge_in"
        assert result["stop_tts"] is True
        assert turn_engine.floor == FloorHolder.CONTACT

    def test_barge_in_strategy_immediate_yield(self):
        engine = TurnTakingEngine(
            interruption=InterruptionConfig(
                default_strategy=InterruptionStrategy.IMMEDIATE_YIELD,
            )
        )
        engine.on_agent_speech_start()
        result = engine.on_contact_speech_start()
        assert result["strategy"] == "immediate_yield"
        assert result["stop_tts"] is True

    def test_barge_in_strategy_finish_sentence(self):
        engine = TurnTakingEngine(
            interruption=InterruptionConfig(
                default_strategy=InterruptionStrategy.FINISH_SENTENCE,
            )
        )
        engine.on_agent_speech_start()
        result = engine.on_contact_speech_start()
        assert result["strategy"] == "finish_sentence"
        assert result["stop_tts"] is False

    def test_barge_in_strategy_ignore_then_yield(self):
        engine = TurnTakingEngine(
            interruption=InterruptionConfig(
                default_strategy=InterruptionStrategy.IGNORE,
                max_ignored_interruptions=2,
            )
        )
        # First two interruptions ignored
        engine.on_agent_speech_start()
        r1 = engine.on_contact_speech_start()
        assert r1["action"] == "continue_speaking"

        engine.on_agent_speech_start()
        r2 = engine.on_contact_speech_start()
        assert r2["action"] == "continue_speaking"

        # Third: forced yield
        engine.on_agent_speech_start()
        r3 = engine.on_contact_speech_start()
        assert r3["action"] == "barge_in"
        assert r3["stop_tts"] is True


class TestAdaptiveEndpointing:
    def test_base_endpointing(self):
        cfg = EndpointingConfig(base_ms=300)
        result = cfg.compute({})
        assert result == 300

    def test_question_shortens_endpointing(self):
        cfg = EndpointingConfig(question_ms=200)
        result = cfg.compute({"agent_asked_question": True})
        assert result == 200

    def test_filler_extends_endpointing(self):
        cfg = EndpointingConfig(base_ms=300, after_filler_ms=800)
        result = cfg.compute({"filler_detected": True})
        assert result == 800

    def test_mid_sentence_extends(self):
        cfg = EndpointingConfig(base_ms=300, mid_sentence_ms=600)
        result = cfg.compute({"mid_sentence": True})
        assert result == 600

    def test_number_dictation(self):
        cfg = EndpointingConfig(number_dictation_ms=1200)
        result = cfg.compute({"dictating_numbers": True})
        assert result == 1200

    def test_clamps_to_min_max(self):
        cfg = EndpointingConfig(min_ms=150, max_ms=2000)
        assert cfg.compute({}) >= 150
        assert cfg.compute({"filler_detected": True, "dictating_numbers": True}) <= 2000

    def test_question_then_filler_takes_max(self):
        cfg = EndpointingConfig(question_ms=200, after_filler_ms=800)
        result = cfg.compute({"agent_asked_question": True, "filler_detected": True})
        assert result == 800  # filler takes precedence (max)

    def test_engine_adaptive_endpointing(self, turn_engine):
        turn_engine.on_agent_speech_end("What's your account number?")
        turn_engine.on_contact_speech_start()
        result = turn_engine.on_contact_speech_end("one two three")
        # Numbers detected → longer endpointing
        assert result["endpointing_ms"] >= 300

    def test_filler_event_extends(self, turn_engine):
        result = turn_engine.on_filler_detected("um")
        assert result["action"] == "extend_endpointing"
        assert result["new_endpointing_ms"] == 800


class TestBackchannel:
    def test_backchannel_after_enough_words(self):
        engine = TurnTakingEngine(
            backchannel=BackchannelConfig(enabled=True, trigger_after_words=5),
        )
        engine.on_agent_speech_end("Tell me about the issue.")
        engine.on_contact_speech_start()
        # Simulate enough words
        engine._contact_word_count = 20
        result = engine.on_contact_speech_end("so we had this problem with the invoice and it was really confusing")
        assert "backchannel" in result
        assert isinstance(result["backchannel"], str)

    def test_no_backchannel_when_disabled(self):
        engine = TurnTakingEngine(
            backchannel=BackchannelConfig(enabled=False),
        )
        engine.on_agent_speech_end("Go ahead.")
        engine.on_contact_speech_start()
        engine._contact_word_count = 50
        result = engine.on_contact_speech_end("a very long monologue about the situation at hand")
        assert "backchannel" not in result

    def test_backchannel_config_responses(self):
        cfg = BackchannelConfig()
        assert cfg.get_response("acknowledgment", 0) in cfg.acknowledgment
        assert cfg.get_response("empathy", 0) in cfg.empathy
        assert cfg.get_response("encouragement", 0) in cfg.encouragement

    def test_backchannel_respects_interval(self):
        engine = TurnTakingEngine(
            backchannel=BackchannelConfig(enabled=True, trigger_after_words=3, min_interval_ms=60000),
        )
        engine._last_backchannel = time.monotonic()  # just happened
        engine.on_agent_speech_end("Go on.")
        engine.on_contact_speech_start()
        engine._contact_word_count = 50
        result = engine.on_contact_speech_end("lots and lots of words from the contact")
        assert "backchannel" not in result


class TestSilenceEscalation:
    def test_silence_levels(self):
        cfg = SilenceConfig()
        assert cfg.get_level(0) == SilenceLevel.NONE
        assert cfg.get_level(6) == SilenceLevel.GENTLE_PROMPT
        assert cfg.get_level(11) == SilenceLevel.CHECK_IN
        assert cfg.get_level(21) == SilenceLevel.FINAL_CHECK
        assert cfg.get_level(31) == SilenceLevel.GOODBYE

    def test_silence_prompts(self):
        cfg = SilenceConfig()
        assert cfg.get_prompt(SilenceLevel.GENTLE_PROMPT) != ""
        assert cfg.get_prompt(SilenceLevel.CHECK_IN) != ""
        assert cfg.get_prompt(SilenceLevel.FINAL_CHECK) != ""
        assert cfg.get_prompt(SilenceLevel.GOODBYE) != ""
        assert cfg.get_prompt(SilenceLevel.NONE) == ""

    def test_engine_silence_tick_no_silence(self, turn_engine):
        result = turn_engine.on_silence_tick()
        assert result is None  # no silence started yet

    def test_engine_silence_escalation(self):
        engine = TurnTakingEngine(
            silence=SilenceConfig(gentle_prompt_s=0.01, check_in_s=0.02),
        )
        engine.on_agent_speech_end("Hello?")
        engine._silence_start = time.monotonic() - 0.015  # 15ms ago
        result = engine.on_silence_tick()
        assert result is not None
        assert result["action"] == "silence_prompt"
        assert result["level"] == "gentle_prompt"

    def test_silence_goodbye_ends_call(self):
        engine = TurnTakingEngine(
            silence=SilenceConfig(goodbye_s=0.01),
        )
        engine.on_agent_speech_end("Hello?")
        engine._silence_start = time.monotonic() - 0.02
        result = engine.on_silence_tick()
        assert result is not None
        assert result["action"] == "end_call"
        assert result["reason"] == "silence_timeout"

    def test_silence_resets_on_speech(self, turn_engine):
        turn_engine.on_agent_speech_end("Hello?")
        assert turn_engine._silence_start is not None
        turn_engine.on_contact_speech_start()
        assert turn_engine._silence_start is None


class TestTurnTakingState:
    def test_to_dict(self, turn_engine):
        state = turn_engine.to_dict()
        assert "floor" in state
        assert "phase" in state
        assert "endpointing_ms" in state
        assert state["floor"] == "agent"


# ══════════════════════════════════════════════════════════════
#  LATENCY TRACKER TESTS
# ══════════════════════════════════════════════════════════════

class TestStageTracker:
    def test_record_and_stats(self):
        t = StageTracker(PipelineStage.STT)
        for ms in [100, 120, 110, 130, 90]:
            t.record(ms)
        assert t.count == 5
        assert t.avg_ms == pytest.approx(110.0, abs=0.1)
        assert t.min_ms == 90.0
        assert t.max_ms == 130.0

    def test_percentiles(self):
        t = StageTracker(PipelineStage.LLM_TTFB)
        for i in range(100):
            t.record(float(i))
        assert t.p50_ms == pytest.approx(50, abs=2)
        assert t.p90_ms == pytest.approx(90, abs=2)
        assert t.p99_ms == pytest.approx(99, abs=2)

    def test_empty_tracker(self):
        t = StageTracker(PipelineStage.TTS_TTFB)
        assert t.avg_ms == 0.0
        assert t.p50_ms == 0.0
        assert t.count == 0

    def test_to_dict(self):
        t = StageTracker(PipelineStage.TOTAL)
        t.record(500)
        d = t.to_dict()
        assert d["stage"] == "total"
        assert d["count"] == 1
        assert d["avg_ms"] == 500.0


class TestCallLatencyTracker:
    def test_basic_timing(self):
        tracker = CallLatencyTracker("call_1")
        tracker.start(PipelineStage.STT)
        time.sleep(0.01)
        duration = tracker.end(PipelineStage.STT)
        assert duration >= 8  # at least ~10ms

    def test_budget_violation(self):
        budget = LatencyBudget(stt_ms=50)
        tracker = CallLatencyTracker("call_1", budget)
        tracker.start(PipelineStage.STT)
        time.sleep(0.06)  # 60ms > 50ms budget
        tracker.end(PipelineStage.STT)
        assert len(tracker.violations) == 1
        assert tracker.violations[0]["stage"] == "stt"

    def test_no_violation_within_budget(self):
        budget = LatencyBudget(stt_ms=500)
        tracker = CallLatencyTracker("call_1", budget)
        tracker.start(PipelineStage.STT)
        tracker.end(PipelineStage.STT)
        assert len(tracker.violations) == 0

    def test_turn_count(self):
        tracker = CallLatencyTracker("call_1")
        assert tracker._turn_count == 0
        tracker.record_turn()
        tracker.record_turn()
        assert tracker._turn_count == 2

    def test_to_dict(self):
        tracker = CallLatencyTracker("call_1")
        tracker.start(PipelineStage.STT)
        tracker.end(PipelineStage.STT)
        d = tracker.to_dict()
        assert d["call_id"] == "call_1"
        assert "stages" in d
        assert "stt" in d["stages"]

    def test_end_without_start_returns_zero(self):
        tracker = CallLatencyTracker("call_1")
        duration = tracker.end(PipelineStage.LLM_TTFB)
        assert duration == 0.0

    def test_optimization_hints_slow_llm(self):
        budget = LatencyBudget(llm_ttfb_ms=100)
        tracker = CallLatencyTracker("call_1", budget)
        # Simulate 20 slow LLM measurements
        for _ in range(20):
            tracker._measurements.append(
                type("M", (), {"stage": PipelineStage.LLM_TTFB, "duration_ms": 200.0, "timestamp": 0, "call_id": "", "metadata": {}})()
            )
        hints = tracker.get_optimization_hints()
        assert len(hints) >= 1
        assert hints[0]["hint"] == "switch_to_faster_model"

    def test_optimization_hints_slow_tts(self):
        budget = LatencyBudget(tts_ttfb_ms=100)
        tracker = CallLatencyTracker("call_1", budget)
        for _ in range(20):
            tracker._measurements.append(
                type("M", (), {"stage": PipelineStage.TTS_TTFB, "duration_ms": 200.0, "timestamp": 0, "call_id": "", "metadata": {}})()
            )
        hints = tracker.get_optimization_hints()
        assert any(h["hint"] == "increase_streaming_optimization" for h in hints)


class TestAggregateLatencyTracker:
    def test_create_call_tracker(self):
        agg = AggregateLatencyTracker()
        ct = agg.create_call_tracker("call_1")
        assert ct.call_id == "call_1"
        assert agg.get_call_tracker("call_1") is ct

    def test_record_aggregate(self):
        agg = AggregateLatencyTracker()
        agg.record(PipelineStage.STT, 100)
        agg.record(PipelineStage.STT, 120)
        stats = agg.get_stage_stats(PipelineStage.STT)
        assert stats["count"] == 2
        assert stats["avg_ms"] == pytest.approx(110.0, abs=0.1)

    def test_remove_call(self):
        agg = AggregateLatencyTracker()
        agg.create_call_tracker("call_1")
        removed = agg.remove_call("call_1")
        assert removed is not None
        assert agg.get_call_tracker("call_1") is None

    def test_all_stats(self):
        agg = AggregateLatencyTracker()
        agg.record(PipelineStage.TOTAL, 500)
        stats = agg.get_all_stats()
        assert "total" in stats
        assert "budget" in stats
        assert stats["active_calls"] == 0

    def test_is_within_budget(self):
        budget = LatencyBudget(stt_ms=200)
        agg = AggregateLatencyTracker(budget)
        for _ in range(20):
            agg.record(PipelineStage.STT, 100)
        assert agg.is_within_budget(PipelineStage.STT) is True

    def test_exceeds_budget(self):
        budget = LatencyBudget(stt_ms=50)
        agg = AggregateLatencyTracker(budget)
        for _ in range(20):
            agg.record(PipelineStage.STT, 100)
        assert agg.is_within_budget(PipelineStage.STT) is False


class TestLatencyBudget:
    def test_budget_for_stages(self):
        budget = LatencyBudget(stt_ms=100, llm_ttfb_ms=200, tts_ttfb_ms=150, total_ms=500)
        assert budget.budget_for(PipelineStage.STT) == 100
        assert budget.budget_for(PipelineStage.LLM_TTFB) == 200
        assert budget.budget_for(PipelineStage.TTS_TTFB) == 150
        assert budget.budget_for(PipelineStage.TOTAL) == 500


# ══════════════════════════════════════════════════════════════
#  VOICE PROMPT BUILDER TESTS
# ══════════════════════════════════════════════════════════════

class TestVoicePromptBuilder:
    def test_basic_prompt(self, contact):
        prompt = VoicePromptBuilder.build(
            contact=contact,
            business_context={"invoice": "INV-301"},
        )
        assert "Priya Sharma" in prompt
        assert "CFO" in prompt
        assert "Acme Corp" in prompt
        assert "INV-301" in prompt

    def test_voice_rules_present(self, contact):
        prompt = VoicePromptBuilder.build(contact=contact, business_context={})
        assert "1-3 SHORT sentences" in prompt
        assert "contractions" in prompt
        assert "Never use markdown" in prompt
        assert "phone call" in prompt.lower()

    def test_no_markdown_instruction(self, contact):
        prompt = VoicePromptBuilder.build(contact=contact, business_context={})
        assert "bullet" in prompt.lower()
        assert "markdown" in prompt.lower()

    def test_pacing_instructions(self, contact):
        prompt = VoicePromptBuilder.build(contact=contact, business_context={})
        assert "slow down" in prompt.lower() or "Slow down" in prompt
        assert "numbers naturally" in prompt.lower() or "numbers" in prompt.lower()

    def test_listening_instructions(self, contact):
        prompt = VoicePromptBuilder.build(contact=contact, business_context={})
        assert "Take your time" in prompt

    def test_conversation_history(self, contact):
        history = [
            {"role": "agent", "content": "Sent email 3 days ago"},
            {"role": "contact", "content": "Will pay next week"},
        ]
        prompt = VoicePromptBuilder.build(
            contact=contact,
            business_context={},
            conversation_history=history,
        )
        assert "RECENT CONVERSATION" in prompt
        assert "email" in prompt.lower()

    def test_followup_reason(self, contact):
        prompt = VoicePromptBuilder.build(
            contact=contact,
            business_context={},
            followup_reason="Payment reminder for overdue invoice",
        )
        assert "Payment reminder" in prompt

    def test_custom_instructions(self, contact):
        prompt = VoicePromptBuilder.build(
            contact=contact,
            business_context={},
            custom_instructions="Always mention the 5% early payment discount",
        )
        assert "5% early payment discount" in prompt


# ══════════════════════════════════════════════════════════════
#  PIPELINE BUILDER TESTS
# ══════════════════════════════════════════════════════════════

class TestVoicePipelineBuilder:
    def test_build_pipeline(self, registry, contact, call_context):
        builder = VoicePipelineBuilder(registry)
        config = builder.build(contact, call_context, initial_message="Hi Priya!")
        assert isinstance(config, PipelineConfig)

        d = config.to_dict()
        assert d["pipeline"]["stt"]["provider"] == "deepgram"
        assert d["pipeline"]["tts"]["provider"] == "elevenlabs"
        assert d["pipeline"]["llm"]["provider"] == "anthropic"
        assert d["pipeline"]["llm"]["stream"] is True

    def test_behavior_config(self, registry, contact, call_context):
        builder = VoicePipelineBuilder(registry)
        config = builder.build(contact, call_context, initial_message="Hello!")
        assert config.behavior["prefetch_greeting"] is True
        assert config.behavior["llm_sentence_streaming"] is True
        assert config.behavior["interruption_handling"] is True
        assert config.behavior["max_response_words"] == 80

    def test_turn_taking_config(self, registry, contact, call_context):
        builder = VoicePipelineBuilder(registry)
        config = builder.build(contact, call_context)
        assert config.turn_taking["backchannel_enabled"] is True
        assert config.turn_taking["adaptive_endpointing"] is True
        assert config.turn_taking["interruption_strategy"] == "immediate_yield"
        assert "silence_escalation" in config.turn_taking

    def test_latency_config(self, registry, contact, call_context):
        builder = VoicePipelineBuilder(registry)
        config = builder.build(contact, call_context)
        assert config.latency["estimated_pipeline_ms"] > 0
        assert config.latency["monitoring_enabled"] is True
        assert config.latency["stt_budget_ms"] > 0
        assert config.latency["tts_budget_ms"] > 0

    def test_callbacks_present(self, registry, contact, call_context):
        builder = VoicePipelineBuilder(registry)
        config = builder.build(contact, call_context)
        assert "on_transcript" in config.callbacks
        assert "on_barge_in" in config.callbacks
        assert "on_vad" in config.callbacks
        assert "on_latency" in config.callbacks

    def test_system_prompt_in_pipeline(self, registry, contact, call_context):
        builder = VoicePipelineBuilder(registry)
        config = builder.build(contact, call_context)
        prompt = config.llm["system_prompt"]
        assert "Priya Sharma" in prompt
        assert "INV-301" in prompt
        assert "1-3 SHORT sentences" in prompt

    def test_no_providers_raises(self, contact, call_context):
        empty_reg = VoiceProviderRegistry()
        builder = VoicePipelineBuilder(empty_reg)
        with pytest.raises(ValueError, match="STT and TTS"):
            builder.build(contact, call_context)

    def test_adaptive_quality_check(self, registry, contact, call_context):
        agg = AggregateLatencyTracker()
        builder = VoicePipelineBuilder(registry, agg)
        # No call tracker → None
        assert builder.should_degrade_quality("nonexistent") is None

    def test_custom_stt_tts_profile(self, registry, contact, call_context):
        registry.register_stt("fast", STTConfig(
            provider=STTProvider.DEEPGRAM, model="nova-2",
            endpointing_ms=150,
            latency=LatencyProfile(80, 30),
        ))
        registry.register_tts("cartesia", TTSConfig(
            provider=TTSProvider.CARTESIA,
            latency=LatencyProfile(60, 20),
        ))
        builder = VoicePipelineBuilder(registry)
        config = builder.build(contact, call_context, stt_profile="fast", tts_profile="cartesia")
        assert config.stt["provider"] == "deepgram"
        assert config.tts["provider"] == "cartesia"


class TestCreateTurnTakingEngine:
    def test_from_empty_config(self):
        engine = create_turn_taking_engine({})
        assert isinstance(engine, TurnTakingEngine)
        assert engine.floor == FloorHolder.AGENT

    def test_from_pipeline_config(self):
        config = {
            "backchannel_enabled": True,
            "base_endpointing_ms": 350,
            "question_endpointing_ms": 150,
            "filler_endpointing_ms": 900,
            "interruption_strategy": "finish_sentence",
        }
        engine = create_turn_taking_engine(config)
        assert engine.backchannel.enabled is True
        assert engine.endpointing.base_ms == 350
        assert engine.endpointing.question_ms == 150
        assert engine.interruption.default_strategy == InterruptionStrategy.FINISH_SENTENCE


# ══════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY: PipecatPipelineBuilder
# ══════════════════════════════════════════════════════════════

class TestPipecatPipelineBuilderCompat:
    """Ensure the legacy PipecatPipelineBuilder still works."""

    def test_legacy_builder(self, contact):
        from channels.voice_adapter import PipecatPipelineBuilder
        builder = PipecatPipelineBuilder({
            "stt_provider": "deepgram",
            "stt_model": "nova-2",
            "tts_provider": "elevenlabs",
            "tts_voice_id": "voice_abc",
        })
        config = builder.build(contact, {"business_context": {"amount": 5000}}, "Hello!")
        pipeline = config["pipeline"]
        assert pipeline["stt"]["provider"] == "deepgram"
        assert pipeline["tts"]["provider"] == "elevenlabs"
        assert pipeline["llm"]["provider"] == "anthropic"
        assert pipeline["llm"]["stream"] is True
        assert pipeline["behavior"]["prefetch_greeting"] is True
        assert pipeline["behavior"]["llm_sentence_streaming"] is True
        assert pipeline["behavior"]["max_response_words"] == 80

    def test_legacy_prompt_has_voice_rules(self, contact):
        from channels.voice_adapter import PipecatPipelineBuilder
        builder = PipecatPipelineBuilder({})
        config = builder.build(contact, {"business_context": {}})
        prompt = config["pipeline"]["llm"]["system_prompt"]
        assert "1-3 SHORT sentences" in prompt
        assert "markdown" in prompt.lower()


# ══════════════════════════════════════════════════════════════
#  INTEGRATION: Full flow simulation
# ══════════════════════════════════════════════════════════════

class TestFullConversationFlow:
    """Simulate a complete voice conversation flow."""

    def test_complete_turn_cycle(self, turn_engine):
        """Agent greets → contact responds → agent replies cycle."""
        # Agent greets
        turn_engine.on_agent_speech_start()
        assert turn_engine.floor == FloorHolder.AGENT
        turn_engine.on_agent_speech_end("Hi Priya, this is a quick call about invoice INV-301. Do you have a moment?")

        # Floor yielded, agent asked a question
        assert turn_engine.floor == FloorHolder.YIELDED
        assert turn_engine._agent_asked_question is True

        # Contact responds
        result = turn_engine.on_contact_speech_start()
        assert result["action"] == "listen"
        assert turn_engine.floor == FloorHolder.CONTACT

        # Contact finishes with a complete sentence
        result = turn_engine.on_contact_speech_end("Yes, go ahead.")
        assert result["likely_complete"] is True
        # Question asked → shorter endpointing
        assert result["endpointing_ms"] <= 300

    def test_filler_then_response(self, turn_engine):
        """Contact says filler then continues."""
        turn_engine.on_agent_speech_end("What's your account number?")
        turn_engine.on_contact_speech_start()

        # Contact says filler
        filler_result = turn_engine.on_filler_detected("um")
        assert filler_result["action"] == "extend_endpointing"

        # Contact finishes with partial info
        result = turn_engine.on_contact_speech_end("um let me check my")
        assert result["likely_complete"] is False
        # Mid-sentence + filler → long endpointing
        assert result["endpointing_ms"] >= 600

    def test_latency_tracked_through_call(self):
        """Full pipeline latency tracked per turn."""
        agg = AggregateLatencyTracker(LatencyBudget(total_ms=600))
        ct = agg.create_call_tracker("call_1")

        # Simulate a turn
        ct.start(PipelineStage.STT)
        time.sleep(0.01)
        ct.end(PipelineStage.STT)

        ct.start(PipelineStage.LLM_TTFB)
        time.sleep(0.01)
        ct.end(PipelineStage.LLM_TTFB)

        ct.start(PipelineStage.TTS_TTFB)
        time.sleep(0.01)
        ct.end(PipelineStage.TTS_TTFB)

        ct.start(PipelineStage.TOTAL)
        time.sleep(0.03)
        total = ct.end(PipelineStage.TOTAL)
        ct.record_turn()

        assert total >= 25  # at least ~30ms
        assert ct._turn_count == 1
        stats = ct.to_dict()
        assert stats["turns"] == 1
        assert len(stats["stages"]) >= 3

    def test_barge_in_with_latency(self):
        """Contact interrupts agent, engine yields and latency records it."""
        engine = TurnTakingEngine()
        engine.on_agent_speech_start()

        # Barge-in
        result = engine.on_contact_speech_start()
        assert result["action"] == "barge_in"
        assert result["stop_tts"] is True

        # Contact speaks
        end_result = engine.on_contact_speech_end("Wait, I already paid that.")
        assert end_result["likely_complete"] is True
