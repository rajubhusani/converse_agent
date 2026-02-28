"""
Voice Pipeline Manager — orchestrates Pipecat pipelines per active call.

This is the runtime that actually handles live voice conversations:
1. Creates a Pipecat pipeline when a call connects
2. Wires Deepgram STT → Claude LLM → Cartesia TTS
3. Integrates turn-taking engine for natural conversation
4. Tracks per-stage latency via CallLatencyTracker
5. Tears down cleanly on call end

Run standalone: python -m voice.server
"""
from __future__ import annotations

import asyncio
import structlog
from typing import Any, Callable, Optional
from datetime import datetime, timezone

from voice.providers import (
    VoiceProviderRegistry, STTConfig, TTSConfig, TelephonyConfig,
    STTProvider, TTSProvider, TelephonyProvider, AudioCodec, QualityPreset,
)
from voice.pipeline import VoicePipelineBuilder, PipelineConfig, VoicePromptBuilder
from voice.turn_taking import (
    TurnTakingEngine, EndpointingConfig, SilenceConfig,
    InterruptionConfig, InterruptionStrategy, BackchannelConfig,
    FloorHolder, create_turn_taking_engine,
)
from voice.latency import (
    AggregateLatencyTracker, CallLatencyTracker, PipelineStage, LatencyBudget,
)
from models.schemas import Contact

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────
#  Per-Call Pipeline State
# ──────────────────────────────────────────────────────────────

class ActivePipeline:
    """
    Holds all per-call state: pipeline config, turn-taking engine,
    latency tracker, and runtime handles.
    """

    def __init__(
        self,
        call_id: str,
        config: PipelineConfig,
        turn_engine: TurnTakingEngine,
        latency_tracker: CallLatencyTracker,
    ):
        self.call_id = call_id
        self.config = config
        self.turn_engine = turn_engine
        self.latency_tracker = latency_tracker
        self.created_at = datetime.now(timezone.utc)

        # Runtime handles (set when Pipecat connects)
        self.stt_ws = None        # Deepgram WebSocket
        self.llm_stream = None    # Anthropic/OpenAI streaming session
        self.tts_ws = None        # Cartesia/ElevenLabs WebSocket
        self.sip_transport = None  # SIP/RTP media transport
        self.pipeline_task: Optional[asyncio.Task] = None

        # State
        self.is_running = False
        self.current_llm_response = ""
        self.greeting_audio: Optional[bytes] = None  # Pre-generated greeting

    async def stop(self):
        """Gracefully tear down all connections."""
        self.is_running = False
        if self.pipeline_task and not self.pipeline_task.done():
            self.pipeline_task.cancel()
            try:
                await self.pipeline_task
            except asyncio.CancelledError:
                pass
        # Close WebSockets
        for ws in [self.stt_ws, self.tts_ws]:
            if ws and hasattr(ws, "close"):
                try:
                    await ws.close()
                except Exception:
                    pass
        logger.info("pipeline_stopped", call_id=self.call_id)


# ──────────────────────────────────────────────────────────────
#  Pipeline Manager
# ──────────────────────────────────────────────────────────────

class VoicePipelineManager:
    """
    Manages lifecycle of voice pipelines across all active calls.

    Usage:
        manager = VoicePipelineManager(registry)
        await manager.start()

        # When a call connects:
        await manager.create_pipeline(call_id, contact, context)

        # When a call ends:
        await manager.destroy_pipeline(call_id)

        await manager.shutdown()
    """

    def __init__(
        self,
        registry: VoiceProviderRegistry,
        on_transcript: Optional[Callable] = None,
        on_call_ended: Optional[Callable] = None,
    ):
        self.registry = registry
        self.aggregate_tracker = AggregateLatencyTracker()
        self._pipelines: dict[str, ActivePipeline] = {}
        self._on_transcript = on_transcript
        self._on_call_ended = on_call_ended
        self._default_quality = QualityPreset.LOW_LATENCY

    @property
    def active_call_count(self) -> int:
        return sum(1 for p in self._pipelines.values() if p.is_running)

    # ── Pipeline Lifecycle ─────────────────────────────────

    async def create_pipeline(
        self,
        call_id: str,
        contact: Contact,
        context: dict[str, Any],
        quality_preset: str = "",
    ) -> dict[str, Any]:
        """
        Create and start a new voice pipeline for a connected call.

        This is called by VoiceAdapter.handle_call_status() when the
        telephony provider reports the call is connected (in-progress).

        Args:
            call_id: Unique call identifier
            contact: The person we're talking to
            context: Business context (invoice details, order info, etc.)
            quality_preset: Override quality preset (default: low_latency)
        """
        if call_id in self._pipelines:
            logger.warning("pipeline_already_exists", call_id=call_id)
            return {"status": "already_exists", "call_id": call_id}

        # 1. Build pipeline configuration
        preset = quality_preset or self._default_quality
        builder = VoicePipelineBuilder(self.registry, self.aggregate_tracker)
        config = builder.build(
            contact=contact,
            context=context,
            initial_message=context.get("initial_message", ""),
            quality_preset=preset,
        )

        # 2. Create turn-taking engine with Indian English tuning
        turn_config = self._build_turn_taking_config()
        turn_engine = create_turn_taking_engine(turn_config)

        # 3. Create per-call latency tracker
        call_tracker = self.aggregate_tracker.create_call_tracker(call_id)

        # 4. Assemble pipeline state
        pipeline = ActivePipeline(
            call_id=call_id,
            config=config,
            turn_engine=turn_engine,
            latency_tracker=call_tracker,
        )

        # 5. Pre-generate greeting if configured
        if config.pipeline.get("prefetch_greeting") and config.initial_greeting:
            pipeline.greeting_audio = await self._prefetch_greeting(config)

        # 6. Start the pipeline
        pipeline.is_running = True
        pipeline.pipeline_task = asyncio.create_task(
            self._run_pipeline(pipeline),
            name=f"pipeline_{call_id}",
        )

        self._pipelines[call_id] = pipeline

        logger.info(
            "pipeline_created",
            call_id=call_id,
            contact=contact.name,
            preset=preset,
            target_latency=config.pipeline.get("target_latency_ms", 500),
        )

        return {
            "status": "pipeline_ready",
            "call_id": call_id,
            "quality_preset": preset,
        }

    async def destroy_pipeline(self, call_id: str) -> dict[str, Any]:
        """Tear down a pipeline when a call ends."""
        pipeline = self._pipelines.pop(call_id, None)
        if not pipeline:
            return {"status": "not_found", "call_id": call_id}

        await pipeline.stop()
        self.aggregate_tracker.remove_call(call_id)

        # Collect final metrics
        metrics = pipeline.latency_tracker.get_report()
        logger.info(
            "pipeline_destroyed",
            call_id=call_id,
            duration_s=(datetime.now(timezone.utc) - pipeline.created_at).total_seconds(),
            total_p90=metrics.get("total", {}).get("p90_ms", 0),
        )

        return {"status": "destroyed", "call_id": call_id, "metrics": metrics}

    # ── Audio Event Handlers ───────────────────────────────
    # These are called by the Pipecat pipeline's event loop

    async def on_stt_transcript(
        self,
        call_id: str,
        text: str,
        confidence: float = 1.0,
        is_final: bool = True,
    ) -> Optional[dict[str, Any]]:
        """
        Called by Deepgram when speech is transcribed.

        For interim results: update turn-taking engine (filler detection).
        For final results: forward to VoiceAdapter.handle_transcript().
        """
        pipeline = self._pipelines.get(call_id)
        if not pipeline or not pipeline.is_running:
            return None

        tracker = pipeline.latency_tracker
        engine = pipeline.turn_engine

        if not is_final:
            # Check for fillers in interim results
            fillers = self._detect_fillers(text)
            if fillers:
                result = engine.on_filler_detected(fillers[0])
                logger.debug("filler_detected", call_id=call_id, filler=fillers[0])
            return None

        # Final transcript — track STT latency
        tracker.end(PipelineStage.STT)

        # Update turn-taking
        result = engine.on_contact_speech_end(text)

        # Generate backchannel if appropriate
        backchannel_action = None
        if "backchannel" in result:
            backchannel_action = {
                "type": "backchannel",
                "text": result["backchannel"],
            }

        # Forward to VoiceAdapter via callback
        if self._on_transcript:
            await self._on_transcript(
                call_id=call_id,
                role="contact",
                text=text,
                confidence=confidence,
                is_final=True,
            )

        # Start LLM timing for the response
        tracker.start(PipelineStage.LLM_TTFB)

        return {
            "text": text,
            "likely_complete": result.get("likely_complete", True),
            "backchannel": backchannel_action,
        }

    async def on_contact_speech_start(self, call_id: str) -> dict[str, str]:
        """Called by Deepgram VAD when contact starts speaking."""
        pipeline = self._pipelines.get(call_id)
        if not pipeline:
            return {"action": "none"}

        result = pipeline.turn_engine.on_contact_speech_start()

        # Track STT start time
        pipeline.latency_tracker.start(PipelineStage.STT)

        if result.get("stop_tts", False):
            return {"action": "stop_tts"}
        return {"action": "continue"}

    async def on_llm_first_token(self, call_id: str) -> None:
        """Called when Claude returns first token of response."""
        pipeline = self._pipelines.get(call_id)
        if pipeline:
            pipeline.latency_tracker.end(PipelineStage.LLM_TTFB)
            pipeline.latency_tracker.start(PipelineStage.TTS_TTFB)

    async def on_tts_first_audio(self, call_id: str) -> None:
        """Called when TTS returns first audio chunk."""
        pipeline = self._pipelines.get(call_id)
        if pipeline:
            pipeline.latency_tracker.end(PipelineStage.TTS_TTFB)
            pipeline.latency_tracker.end(PipelineStage.TOTAL)

            # Check latency budget
            hints = pipeline.latency_tracker.get_optimization_hints()
            if hints:
                logger.warning("latency_budget_exceeded", call_id=call_id, hints=hints)

    async def on_silence_tick(self, call_id: str, silence_duration_s: float) -> Optional[dict]:
        """Called periodically when no speech detected."""
        pipeline = self._pipelines.get(call_id)
        if not pipeline:
            return None

        result = pipeline.turn_engine.on_silence_tick()
        if not result:
            return None

        action = result.get("action")
        if action == "silence_prompt":
            return {"action": "speak", "text": result["prompt"]}
        elif action == "end_call":
            return {"action": "end_call", "reason": "silence_timeout"}
        return None

    # ── Internal Pipeline Logic ────────────────────────────

    async def _run_pipeline(self, pipeline: ActivePipeline) -> None:
        """
        Main pipeline loop — connects to providers and processes audio.

        In production, this creates Pipecat pipeline processors:
        - DeepgramSTTService (WebSocket)
        - AnthropicLLMService (streaming)
        - CartesiaTTSService or ElevenLabsTTSService (WebSocket)
        - SIPTransport (audio I/O)

        The processors are wired in a pipeline:
            SIP Input → STT → LLM → TTS → SIP Output
        with event callbacks to our handlers above.
        """
        call_id = pipeline.call_id
        config = pipeline.config

        try:
            logger.info("pipeline_starting", call_id=call_id)

            # === PIPECAT INTEGRATION POINT ===
            #
            # This is where you instantiate the actual Pipecat pipeline.
            # The config object has everything needed:
            #
            #   config.stt   → {"provider": "deepgram", "api_key": "...", "model": "nova-2", ...}
            #   config.llm   → {"model": "claude-haiku-4-5-20251001", "system_prompt": "...", ...}
            #   config.tts   → {"provider": "cartesia", "api_key": "...", "voice_id": "...", ...}
            #   config.pipeline → {"target_latency_ms": 500, "llm_sentence_streaming": True, ...}
            #
            # Example Pipecat wiring:
            #
            # from pipecat.pipeline.pipeline import Pipeline
            # from pipecat.services.deepgram import DeepgramSTTService
            # from pipecat.services.anthropic import AnthropicLLMService
            # from pipecat.services.cartesia import CartesiaTTSService
            # from pipecat.transports.network.sip import SIPTransport
            #
            # stt = DeepgramSTTService(api_key=config.stt["api_key"], ...)
            # llm = AnthropicLLMService(
            #     api_key=config.llm["api_key"],
            #     model=config.llm["model"],
            #     system_prompt=config.llm["system_prompt"],
            # )
            # tts = CartesiaTTSService(api_key=config.tts["api_key"], ...)
            # transport = SIPTransport(...)
            #
            # pipeline = Pipeline([
            #     transport.input(),
            #     stt,          # → triggers on_stt_transcript
            #     llm,          # → triggers on_llm_first_token
            #     tts,          # → triggers on_tts_first_audio
            #     transport.output(),
            # ])
            #
            # # Register our event handlers
            # stt.on("transcript", lambda t: self.on_stt_transcript(call_id, t.text, t.confidence, t.is_final))
            # stt.on("vad_start", lambda: self.on_contact_speech_start(call_id))
            # llm.on("first_token", lambda: self.on_llm_first_token(call_id))
            # tts.on("first_audio", lambda: self.on_tts_first_audio(call_id))
            #
            # await pipeline.run()

            # Play greeting if pre-generated
            if pipeline.greeting_audio:
                logger.info("playing_prefetched_greeting", call_id=call_id)
                # In production: send greeting_audio to SIP transport

            # For now, keep the task alive until stopped
            while pipeline.is_running:
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("pipeline_cancelled", call_id=call_id)
        except Exception as e:
            logger.error("pipeline_error", call_id=call_id, error=str(e))
        finally:
            pipeline.is_running = False

    async def _prefetch_greeting(self, config: PipelineConfig) -> Optional[bytes]:
        """Pre-generate greeting audio before the call connects."""
        greeting_text = config.initial_greeting
        if not greeting_text:
            return None

        # In production: call TTS provider to generate audio
        # tts_client = CartesiaTTSService(...)
        # audio = await tts_client.synthesize(greeting_text)
        # return audio
        logger.debug("prefetch_greeting", text=greeting_text[:50])
        return None  # Placeholder

    # ── Turn-Taking Configuration ──────────────────────────

    def _build_turn_taking_config(self) -> dict[str, Any]:
        """
        Build turn-taking config tuned for Indian English conversations.
        """
        return {
            "endpointing": EndpointingConfig(
                base_ms=350,             # Wider than US English default (300)
                question_ms=250,
                after_filler_ms=1000,    # Hindi fillers need more time
                number_dictation_ms=1500,  # "Paanch lakh baavan hazaar" is long
                mid_sentence_ms=700,
                emotional_ms=600,
            ),
            "silence": SilenceConfig(
                gentle_prompt_s=8,       # More patience than default 5s
                check_in_s=15,
                final_check_s=25,
                goodbye_s=35,
                gentle_prompt_text="Sir, are you still there?",
                check_in_text="I'm still on the line. Should I continue?",
                final_check_text="It seems we've lost connection. I'll try again shortly.",
            ),
            "interruption": InterruptionConfig(
                strategy=InterruptionStrategy.IMMEDIATE_YIELD,
                min_agent_words_before_yield=3,
                max_ignores_before_forced_yield=2,
            ),
            "backchannel": BackchannelConfig(
                enabled=True,
                min_words_before_backchannel=15,
                phrases=["Mm-hmm", "I see", "Right", "Okay", "Theek hai"],
            ),
        }

    # ── Filler Detection ───────────────────────────────────

    FILLER_WORDS = {
        # English
        "um", "uh", "like", "you know", "i mean", "sort of", "well",
        "so", "actually", "let me think", "hold on",
        # Hindi / Hinglish
        "matlab", "woh", "acha", "haan", "toh", "basically",
        "dekho", "ek minute", "ruko", "theek hai", "suno",
        "abhi", "haan ji", "bas",
    }

    def _detect_fillers(self, text: str) -> list[str]:
        text_lower = text.lower().strip()
        found = []
        for filler in self.FILLER_WORDS:
            if filler in text_lower:
                found.append(filler)
        return found

    # ── Metrics ────────────────────────────────────────────

    def get_metrics(self) -> dict[str, Any]:
        """Aggregate metrics across all active and recent calls."""
        return {
            "active_calls": self.active_call_count,
            "total_pipelines_created": len(self._pipelines),
            "latency": self.aggregate_tracker.get_all_stats(),
        }

    def get_call_metrics(self, call_id: str) -> Optional[dict[str, Any]]:
        pipeline = self._pipelines.get(call_id)
        if not pipeline:
            return None
        return pipeline.latency_tracker.get_report()

    # ── Lifecycle ──────────────────────────────────────────

    async def start(self) -> None:
        logger.info("voice_pipeline_manager_started")

    async def shutdown(self) -> None:
        """Gracefully shut down all active pipelines."""
        call_ids = list(self._pipelines.keys())
        logger.info("shutting_down_pipelines", count=len(call_ids))
        for call_id in call_ids:
            await self.destroy_pipeline(call_id)
        logger.info("voice_pipeline_manager_stopped")
