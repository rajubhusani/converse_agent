"""
Voice Subsystem — Low-latency natural voice conversation engine.

Modules:
- providers: Multi-provider STT/TTS/Telephony config with latency profiles
- turn_taking: Natural conversation turn-taking state machine
- latency: Per-stage latency tracking, budgets, and optimization hints
- pipeline: Streaming pipeline builder with sentence-level LLM streaming
"""
from voice.providers import (
    VoiceProviderRegistry, STTProvider, TTSProvider, TelephonyProvider,
    STTConfig, TTSConfig, TelephonyConfig,
    QualityPreset, AudioCodec, LatencyProfile, QUALITY_PRESETS,
)
from voice.turn_taking import (
    TurnTakingEngine, FloorHolder, ConversationPhase, SilenceLevel,
    BackchannelConfig, EndpointingConfig, SilenceConfig,
    InterruptionConfig, InterruptionStrategy,
    detect_fillers, is_likely_complete, is_question,
)
from voice.latency import (
    PipelineStage, LatencyBudget, CallLatencyTracker,
    AggregateLatencyTracker, StageTracker,
)
from voice.pipeline import (
    VoicePipelineBuilder, VoicePromptBuilder, PipelineConfig,
    create_turn_taking_engine,
)

__all__ = [
    "VoiceProviderRegistry", "STTProvider", "TTSProvider", "TelephonyProvider",
    "STTConfig", "TTSConfig", "TelephonyConfig",
    "QualityPreset", "AudioCodec", "LatencyProfile", "QUALITY_PRESETS",
    "TurnTakingEngine", "FloorHolder", "ConversationPhase", "SilenceLevel",
    "BackchannelConfig", "EndpointingConfig", "SilenceConfig",
    "InterruptionConfig", "InterruptionStrategy",
    "detect_fillers", "is_likely_complete", "is_question",
    "PipelineStage", "LatencyBudget", "CallLatencyTracker",
    "AggregateLatencyTracker", "StageTracker",
    "VoicePipelineBuilder", "VoicePromptBuilder", "PipelineConfig",
    "create_turn_taking_engine",
]
