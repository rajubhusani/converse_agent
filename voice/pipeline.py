"""
Voice Pipeline — Low-latency streaming pipeline builder.

Builds the complete Pipecat pipeline configuration optimized for
natural, low-latency voice conversations. Key optimizations:

- Sentence-level LLM streaming: starts TTS on the first complete
  sentence while the LLM is still generating the rest
- Greeting prefetch: pre-generates the opening TTS audio before
  the call connects, so it plays instantly
- Parallel warmup: establishes STT and TTS WebSocket connections
  concurrently during the ring phase
- Adaptive quality: degrades to faster models when latency spikes
- Context-aware system prompts with voice-specific naturalness rules
"""
from __future__ import annotations

import json
import structlog
from typing import Any, Optional
from dataclasses import dataclass, field

from models.schemas import Contact
from voice.providers import (
    VoiceProviderRegistry,
    STTConfig,
    TTSConfig,
    TelephonyConfig,
    QualityPreset,
    AudioCodec,
)
from voice.turn_taking import (
    TurnTakingEngine,
    BackchannelConfig,
    EndpointingConfig,
    SilenceConfig,
    InterruptionConfig,
    InterruptionStrategy,
)
from voice.latency import (
    LatencyBudget,
    AggregateLatencyTracker,
)

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  PIPELINE CONFIG
# ══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Complete pipeline configuration for a single call."""
    stt: dict[str, Any] = field(default_factory=dict)
    tts: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    telephony: dict[str, Any] = field(default_factory=dict)
    behavior: dict[str, Any] = field(default_factory=dict)
    turn_taking: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    callbacks: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": {
                "stt": self.stt,
                "tts": self.tts,
                "llm": self.llm,
            },
            "telephony": self.telephony,
            "behavior": self.behavior,
            "turn_taking": self.turn_taking,
            "latency": self.latency,
            "callbacks": self.callbacks,
        }


# ══════════════════════════════════════════════════════════════
#  NATURALNESS PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

class VoicePromptBuilder:
    """
    Builds voice-optimized system prompts that produce natural speech.

    The prompts instruct the LLM to:
    - Use conversational sentence structure
    - Speak in natural cadences (short sentences, pauses)
    - Spell out numbers verbally
    - Use contractions and informal but professional language
    - React to emotional cues appropriately
    - Keep responses short for voice (1-3 sentences per turn)
    """

    VOICE_RULES = """VOICE CONVERSATION RULES — Follow these precisely:

DELIVERY:
- Keep every response to 1-3 SHORT sentences. This is a phone call, not an email.
- Use natural spoken language — contractions ("I'll", "don't", "we've"), informal but professional.
- Vary your sentence length. Mix short punchy sentences with slightly longer ones.
- End turns with a clear signal: a question, a pause phrase ("take your time"), or a conclusion.

PACING:
- For important info (amounts, dates, IDs), slow down by breaking into chunks:
  "That's fifty... two... thousand... and four hundred rupees."
- Spell out numbers naturally: say "two lakh fifty thousand" not "250000".
- Say dates conversationally: "January fifteenth" not "01/15".
- Pause implicitly with sentence breaks before key points.

NATURALNESS:
- React to what you hear. If they sound frustrated, acknowledge it: "I understand this is frustrating."
- If they sound busy: "I'll keep this quick" or "Would another time work better?"
- Use transition phrases: "So here's the thing...", "Actually...", "By the way..."
- Mirror their energy — if they're casual, be casual. If formal, match it.
- When they say something you can acknowledge, do: "Right", "Makes sense", "Got it."

LISTENING:
- If they go silent, wait a beat. Don't rush to fill every pause.
- If they seem to be thinking, say "Take your time" and wait.
- If they give a short answer, don't assume you need to fill the silence — ask a follow-up or wrap up.

NEVER DO:
- Never use markdown, bullet points, headers, asterisks, or any text formatting.
- Never list things with numbers or dashes. Say "a few things: first... second... and also..."
- Never say "as an AI" or "as a language model." You're a professional caller.
- Never give a long monologue. Every response gets a reaction checkpoint.
- Never read IDs character by character — group them: "Invoice IN-V three-zero-one"."""

    @classmethod
    def build(
        cls,
        contact: Contact,
        business_context: dict[str, Any],
        conversation_history: list[dict] = None,
        followup_reason: str = "",
        custom_instructions: str = "",
        language: str = "en",
    ) -> str:
        history_block = ""
        if conversation_history:
            recent = conversation_history[-5:]
            lines = []
            for h in recent:
                role = h.get("role", "?")
                content = str(h.get("content", ""))[:120]
                lines.append(f"  {role}: {content}")
            history_block = f"\nRECENT CONVERSATION:\n" + "\n".join(lines) + "\n"

        biz_json = json.dumps(business_context, indent=2, default=str) if business_context else "{}"

        prompt = f"""You are a professional follow-up agent on a live voice call.
You are speaking with {contact.name}{f' ({contact.role})' if contact.role else ''}{f' from {contact.organization}' if contact.organization else ''}.

{cls.VOICE_RULES}

BUSINESS CONTEXT:
{biz_json}
{history_block}
CALL PURPOSE: {followup_reason or 'General follow-up'}
{f'{chr(10)}ADDITIONAL INSTRUCTIONS:{chr(10)}{custom_instructions}' if custom_instructions else ''}

Remember: This is a LIVE PHONE CALL. Be brief, warm, and human."""

        return prompt


# ══════════════════════════════════════════════════════════════
#  PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════

class VoicePipelineBuilder:
    """
    Builds complete pipeline configurations for Pipecat.

    Wires together provider configs, turn-taking, latency budgets,
    and voice-optimized prompts into a single pipeline spec.
    """

    def __init__(
        self,
        provider_registry: VoiceProviderRegistry,
        latency_tracker: AggregateLatencyTracker = None,
    ):
        self.providers = provider_registry
        self.latency_tracker = latency_tracker or AggregateLatencyTracker()
        self._callback_base_url: str = "/api/v1/voice"

    def build(
        self,
        contact: Contact,
        context: dict[str, Any],
        initial_message: str = "",
        stt_profile: str = "",
        tts_profile: str = "",
        quality_preset: QualityPreset = None,
    ) -> PipelineConfig:
        """Build a complete pipeline config for a call."""

        stt_config = self.providers.get_stt(stt_profile)
        tts_config = self.providers.get_tts(tts_profile)
        telephony_config = self.providers.get_telephony()

        if not stt_config or not tts_config:
            raise ValueError("STT and TTS providers must be configured")

        # Build system prompt
        system_prompt = VoicePromptBuilder.build(
            contact=contact,
            business_context=context.get("business_context", {}),
            conversation_history=context.get("conversation_history", []),
            followup_reason=context.get("followup", {}).get("reason", ""),
            custom_instructions=context.get("custom_instructions", ""),
        )

        # LLM config — configurable provider (anthropic or openai)
        llm_provider = context.get("llm_provider", "anthropic")
        llm_api_key = context.get("llm_api_key", "")
        llm_model = context.get("llm_model", "claude-haiku-4-5-20251001")
        llm_max_tokens = context.get("llm_max_tokens", 150)

        pipeline = PipelineConfig(
            stt={
                "provider": stt_config.provider.value,
                **stt_config.to_provider_params(),
            },
            tts={
                "provider": tts_config.provider.value,
                **tts_config.to_provider_params(),
            },
            llm={
                "provider": llm_provider,
                "api_key": llm_api_key,
                "model": llm_model,
                "system_prompt": system_prompt,
                "temperature": 0.7,
                "max_tokens": llm_max_tokens,
                "stream": True,
            },
            telephony=self._build_telephony(telephony_config, contact),
            behavior=self._build_behavior(stt_config, tts_config, initial_message),
            turn_taking=self._build_turn_taking(context),
            latency=self._build_latency(stt_config, tts_config),
            callbacks=self._build_callbacks(),
        )

        return pipeline

    def _build_telephony(
        self, config: Optional[TelephonyConfig], contact: Contact
    ) -> dict[str, Any]:
        if not config:
            return {"provider": "twilio"}
        return {
            "provider": config.provider.value,
            "account_sid": config.account_sid,
            "auth_token": config.auth_token,
            "from_number": config.phone_number,
            "status_callback_url": config.status_callback_url or f"{self._callback_base_url}/status",
            "ring_timeout": config.ring_timeout_s,
            "answering_machine_detection": config.answering_machine_detection,
            "amd_timeout_ms": config.amd_timeout_ms,
            "codec": config.codec.value,
        }

    def _build_behavior(
        self, stt: STTConfig, tts: TTSConfig, initial_message: str
    ) -> dict[str, Any]:
        return {
            "initial_message": initial_message,
            "prefetch_greeting": bool(initial_message),
            "interruption_handling": True,
            "llm_sentence_streaming": True,       # stream TTS per sentence
            "min_speech_duration_ms": 300,         # ignore sub-300ms noises
            "vad_threshold": stt.vad_threshold,
            "silence_timeout_ms": 30000,           # handled by turn-taking engine
            "max_response_words": 80,              # keep voice responses short
            "output_format": tts.output_format.value,
            "sample_rate": tts.sample_rate,
        }

    def _build_turn_taking(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "backchannel_enabled": True,
            "backchannel_min_interval_ms": 4000,
            "adaptive_endpointing": True,
            "base_endpointing_ms": 300,
            "question_endpointing_ms": 200,
            "filler_endpointing_ms": 800,
            "interruption_strategy": "immediate_yield",
            "silence_escalation": {
                "gentle_prompt_s": 5,
                "check_in_s": 10,
                "final_check_s": 20,
                "goodbye_s": 30,
            },
        }

    def _build_latency(
        self, stt: STTConfig, tts: TTSConfig
    ) -> dict[str, Any]:
        estimated = (
            stt.latency.time_to_first_byte_ms
            + 200  # LLM TTFB estimate
            + tts.latency.time_to_first_byte_ms
        )
        return {
            "target_total_ms": max(estimated, 300),
            "stt_budget_ms": stt.latency.time_to_first_byte_ms + 50,
            "llm_budget_ms": 300,
            "tts_budget_ms": tts.latency.time_to_first_byte_ms + 50,
            "estimated_pipeline_ms": estimated,
            "monitoring_enabled": True,
        }

    def _build_callbacks(self) -> dict[str, str]:
        return {
            "on_transcript": f"{self._callback_base_url}/transcript",
            "on_call_status": f"{self._callback_base_url}/status",
            "on_call_end": f"{self._callback_base_url}/call-ended",
            "on_dtmf": f"{self._callback_base_url}/dtmf",
            "on_vad": f"{self._callback_base_url}/vad",
            "on_latency": f"{self._callback_base_url}/latency",
            "on_barge_in": f"{self._callback_base_url}/barge-in",
        }

    # ── Adaptive quality ──────────────────────────────────────

    def should_degrade_quality(self, call_id: str) -> Optional[dict[str, Any]]:
        """
        Check if a call's latency suggests we should degrade quality.
        Returns suggestion dict or None.
        """
        tracker = self.latency_tracker.get_call_tracker(call_id)
        if not tracker:
            return None

        hints = tracker.get_optimization_hints()
        if hints:
            return {
                "action": "degrade_quality",
                "hints": hints,
                "current_violations": len(tracker.violations),
            }
        return None


# ══════════════════════════════════════════════════════════════
#  FACTORY: Create turn-taking engine from config
# ══════════════════════════════════════════════════════════════

def create_turn_taking_engine(config: dict[str, Any] = None) -> TurnTakingEngine:
    """Create a TurnTakingEngine from a config dict (e.g., from pipeline builder)."""
    config = config or {}
    return TurnTakingEngine(
        backchannel=BackchannelConfig(
            enabled=config.get("backchannel_enabled", True),
            min_interval_ms=config.get("backchannel_min_interval_ms", 4000),
        ),
        endpointing=EndpointingConfig(
            base_ms=config.get("base_endpointing_ms", 300),
            question_ms=config.get("question_endpointing_ms", 200),
            after_filler_ms=config.get("filler_endpointing_ms", 800),
        ),
        silence=SilenceConfig(
            **(config.get("silence_escalation", {}))
        ) if "silence_escalation" in config else SilenceConfig(),
        interruption=InterruptionConfig(
            default_strategy=InterruptionStrategy(
                config.get("interruption_strategy", "immediate_yield")
            ),
        ),
    )
