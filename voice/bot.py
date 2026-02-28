"""
Voice Bot — Pipecat pipeline for a single Twilio call.

Handles the complete voice conversation loop:
  Twilio Audio → Deepgram STT → OpenAI LLM → Cartesia TTS → Twilio Audio

Used by the FastAPI WebSocket endpoint to run a pipeline per call.
"""
from __future__ import annotations

import os
import json
import asyncio
import structlog

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import EndFrame

logger = structlog.get_logger()


async def run_voice_bot(websocket, call_sid: str, system_prompt: str = ""):
    """
    Run a Pipecat voice pipeline connected to a Twilio Media Stream.

    Called by the FastAPI WebSocket handler when Twilio connects.
    The websocket must already be accepted.

    Args:
        websocket: FastAPI WebSocket connection (already accepted)
        call_sid: Twilio CallSid for correlation
        system_prompt: The LLM system prompt for this call
    """
    # --- Config from environment ---
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    cartesia_api_key = os.getenv("CARTESIA_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    cartesia_voice_id = os.getenv(
        "CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"
    )
    twilio_account_sid = os.getenv("TWILIO_SID", "")
    twilio_auth_token = os.getenv("TWILIO_AUTH", "")

    if not all([deepgram_api_key, openai_api_key, cartesia_api_key]):
        logger.error(
            "voice_bot_missing_api_keys",
            has_deepgram=bool(deepgram_api_key),
            has_openai=bool(openai_api_key),
            has_cartesia=bool(cartesia_api_key),
        )
        return

    if not system_prompt:
        system_prompt = (
            "You are a professional and friendly voice agent on a live phone call. "
            "Keep your responses short — 1 to 3 sentences. Use natural spoken language. "
            "Never use markdown, bullet points, or text formatting. "
            "Be warm, concise, and conversational."
        )

    # --- Wait for Twilio's 'connected' event to get the stream_sid ---
    stream_sid = ""
    try:
        # Twilio sends a few JSON messages before media: connected, start
        # We need the stream_sid from the 'start' event
        while True:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            msg = json.loads(raw)
            twilio_event = msg.get("event", "")
            logger.info("twilio_ws_event", twilio_event=twilio_event, call_sid=call_sid)

            if twilio_event == "connected":
                logger.info("twilio_stream_connected_event", call_sid=call_sid)
            elif twilio_event == "start":
                stream_sid = msg.get("streamSid", msg.get("start", {}).get("streamSid", ""))
                actual_call_sid = msg.get("start", {}).get("callSid", call_sid)
                logger.info(
                    "twilio_stream_started",
                    stream_sid=stream_sid,
                    actual_call_sid=actual_call_sid,
                )
                # Use the actual call_sid from Twilio if available
                if actual_call_sid:
                    call_sid = actual_call_sid
                break
            elif twilio_event == "media":
                # Shouldn't happen before 'start', but just in case
                logger.warning("twilio_media_before_start", call_sid=call_sid)
                break
    except asyncio.TimeoutError:
        logger.error("twilio_stream_timeout", call_sid=call_sid)
        return
    except Exception as e:
        logger.error("twilio_stream_init_error", call_sid=call_sid, error=str(e))
        return

    if not stream_sid:
        logger.error("twilio_no_stream_sid", call_sid=call_sid)
        return

    logger.info(
        "voice_bot_starting",
        call_sid=call_sid,
        stream_sid=stream_sid,
        model=openai_model,
        voice_id=cartesia_voice_id[:12],
    )

    # --- Transport: Twilio ↔ Pipecat via WebSocket ---
    serializer_params = TwilioFrameSerializer.InputParams(
        twilio_sample_rate=8000,
        sample_rate=8000,
        auto_hang_up=bool(twilio_account_sid and twilio_auth_token),
    )

    serializer_kwargs = {
        "stream_sid": stream_sid,
        "params": serializer_params,
    }
    # Only pass Twilio credentials if auto_hang_up is enabled
    if twilio_account_sid and twilio_auth_token:
        serializer_kwargs["call_sid"] = call_sid
        serializer_kwargs["account_sid"] = twilio_account_sid
        serializer_kwargs["auth_token"] = twilio_auth_token

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            serializer=TwilioFrameSerializer(**serializer_kwargs),
            vad_enabled=True,
            vad_audio_passthrough=True,
        ),
    )

    # --- STT: Deepgram Nova-2 ---
    stt = DeepgramSTTService(api_key=deepgram_api_key, sample_rate=8000)

    # --- LLM: OpenAI GPT-4 Turbo ---
    llm = OpenAILLMService(api_key=openai_api_key, model=openai_model)

    # --- TTS: Cartesia Sonic ---
    # Use pcm_s16le encoding — TwilioFrameSerializer handles PCM ↔ μ-law conversion
    tts = CartesiaTTSService(
        api_key=cartesia_api_key,
        voice_id=cartesia_voice_id,
        sample_rate=8000,
        encoding="pcm_s16le",
        container="raw",
    )

    # --- LLM Context with conversation history ---
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Please greet the person on the phone warmly and ask how you can help them today.",
        },
    ]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # --- Pipeline: STT → UserAgg → LLM → TTS → AssistantAgg ---
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            context_aggregator.assistant(),
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Trigger initial greeting when pipeline starts
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("pipecat_client_connected", call_sid=call_sid)
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("pipecat_client_disconnected", call_sid=call_sid)
        await task.queue_frames([EndFrame()])

    # --- Run ---
    runner = PipelineRunner()
    try:
        logger.info("voice_bot_pipeline_running", call_sid=call_sid)
        await runner.run(task)
    except Exception as e:
        logger.error("voice_bot_error", call_sid=call_sid, error=str(e), exc_info=True)
    finally:
        logger.info("voice_bot_ended", call_sid=call_sid)
