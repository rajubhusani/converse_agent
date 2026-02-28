"""
FastAPI Application — REST API + WebSocket + Webhooks.

Provides:
- REST API for management and dashboard
- WebSocket endpoint for real-time chat
- Webhook endpoints for WhatsApp, Email, and Voice callbacks
- Channel health monitoring and diagnostics
- Scheduler for automated follow-up processing
"""
from __future__ import annotations

import json
import asyncio
import structlog
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

# Load .env before any config is read
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import get_settings, load_settings
from models.schemas import ChannelType, FollowUpStatus, ConversationStatus
from context.tracker import ContextTracker, ContextStore, ContextChange
from context.state_machine import BusinessStateMachine
from core.engine import ConversationEngine
from core.orchestrator import Orchestrator
from rules.engine import RulesEngine
from backend.connector import create_backend_connector
from channels.base import ChannelRegistry
from channels.chat_adapter import ChatAdapter
from channels.email_adapter import EmailAdapter
from channels.whatsapp_adapter import WhatsAppAdapter
from channels.voice_adapter import VoiceAdapter, CallEndReason
from database.adapter import create_configured_store
from job_queue.message_queue import create_message_queue, get_message_queue, Queues
from job_queue.consumer import FollowUpConsumer, DelayedJobPromoter

logger = structlog.get_logger()

# ──────────────────────────────────────────────────────────────
#  Bootstrap
# ──────────────────────────────────────────────────────────────

context_store = create_configured_store()
context_tracker = ContextTracker(context_store)
conversation_engine = ConversationEngine()
rules_engine = RulesEngine(context_tracker)
state_machine = BusinessStateMachine()
channel_registry = ChannelRegistry()
backend_connector = create_backend_connector()

_settings_boot = get_settings()
message_queue = create_message_queue({
    "backend": _settings_boot.queue.backend,
    "redis_url": _settings_boot.queue.redis_url,
})

# Register channel adapters
chat_adapter = ChatAdapter()
email_adapter = EmailAdapter()
whatsapp_adapter = WhatsAppAdapter()
voice_adapter = VoiceAdapter()

channel_registry.register(chat_adapter)
channel_registry.register(email_adapter)
channel_registry.register(whatsapp_adapter)
channel_registry.register(voice_adapter)

orchestrator = Orchestrator(
    context=context_tracker,
    engine=conversation_engine,
    rules=rules_engine,
    state_machine=state_machine,
    channels=channel_registry,
    backend=backend_connector,
    queue=message_queue,
)

followup_consumer = FollowUpConsumer(
    orchestrator, message_queue,
    consumer_group=_settings_boot.queue.consumer_group,
    concurrency=_settings_boot.queue.consumer_concurrency,
)
delayed_promoter = DelayedJobPromoter(
    message_queue,
    interval_seconds=_settings_boot.queue.delayed_promote_interval,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Initialize channels
    await channel_registry.initialize_all(settings.channels)

    # Load rules and state maps from config
    try:
        import yaml
        from pathlib import Path
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            if "rules" in raw:
                rules_engine.load_rules(raw["rules"])
            if "state_maps" in raw:
                state_machine.register_maps_from_config(raw["state_maps"])
    except Exception as e:
        logger.warning("config_load_failed", error=str(e))

    await message_queue.connect()
    _consumer_task = await followup_consumer.start_background()
    _promoter_task = await delayed_promoter.start_background()

    logger.info("converse_agent_started",
                 domain=settings.business_domain,
                 queue_backend=type(message_queue).__name__)
    yield

    await followup_consumer.stop()
    await delayed_promoter.stop()
    await message_queue.close()
    await channel_registry.shutdown_all()
    logger.info("converse_agent_stopped")


# ──────────────────────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ConverseAgent API",
    description="Multi-channel follow-up conversational agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
#  Request/Response Models
# ──────────────────────────────────────────────────────────────

class InboundMessageRequest(BaseModel):
    channel: str
    sender_address: str
    content: str
    metadata: dict[str, Any] = {}


class BackendEventRequest(BaseModel):
    event_name: str
    data: dict[str, Any]


class TriggerFollowUpRequest(BaseModel):
    contact_id: str
    reason: str
    business_context: dict[str, Any] = {}
    channel_priority: list[str] = ["whatsapp", "email"]
    template: str = ""


class ContactCreateRequest(BaseModel):
    name: str
    role: str = ""
    organization: str = ""
    email: str = ""
    phone: str = ""
    whatsapp: str = ""
    metadata: dict[str, Any] = {}


class ContextChangeRequest(BaseModel):
    conversation_id: str
    trigger_type: str
    trigger_value: str
    extra_data: dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════
#  HEALTH & DIAGNOSTICS
# ══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "channels": [c.value for c in channel_registry.get_available()],
        "healthy_channels": [c.value for c in channel_registry.get_healthy_channels()],
        "rules_count": len(rules_engine.list_rules()),
    }


@app.get("/api/v1/stats")
async def get_stats():
    stats = await context_tracker.get_stats()
    stats["rules"] = len(rules_engine.list_rules())
    stats["available_channels"] = [c.value for c in channel_registry.get_available()]
    return stats


@app.get("/api/v1/channels/health")
async def channel_health():
    """Comprehensive health report for all channels."""
    return await channel_registry.get_health_report()


# ══════════════════════════════════════════════════════════════
#  INBOUND MESSAGES
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/messages/inbound")
async def receive_inbound_message(req: InboundMessageRequest):
    result = await orchestrator.handle_inbound_message(
        channel=ChannelType(req.channel),
        sender_address=req.sender_address,
        content=req.content,
        metadata=req.metadata,
    )
    return result


# ══════════════════════════════════════════════════════════════
#  BACKEND EVENTS
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/events")
async def receive_backend_event(req: BackendEventRequest):
    results = await orchestrator.handle_backend_event(req.event_name, req.data)
    return {"event": req.event_name, "results": results}


# ══════════════════════════════════════════════════════════════
#  FOLLOW-UPS
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/followups/trigger")
async def trigger_followup(req: TriggerFollowUpRequest):
    from models.schemas import FollowUp
    followup = FollowUp(
        rule_id="manual",
        contact_id=req.contact_id,
        reason=req.reason,
        business_context={"template": req.template, **req.business_context},
        channel_priority=[ChannelType(c) for c in req.channel_priority],
    )
    followup = await context_store.create_followup(followup)
    job = await orchestrator.enqueue_followup(followup)
    return {
        "status": "enqueued",
        "followup_id": followup.id,
        "job_id": job.job_id,
    }


@app.get("/api/v1/followups")
async def list_followups(
    status: str = None,
    contact_id: str = None,
    limit: int = Query(50, le=200),
):
    followups = await context_tracker.store.list_followups(
        status=FollowUpStatus(status) if status else None,
        contact_id=contact_id,
        limit=limit,
    )
    return [f.model_dump() for f in followups]


# ══════════════════════════════════════════════════════════════
#  CONVERSATIONS
# ══════════════════════════════════════════════════════════════

@app.get("/api/v1/conversations")
async def list_conversations(
    status: str = None,
    channel: str = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
):
    convs = await context_tracker.store.list_conversations(
        status=ConversationStatus(status) if status else None,
        channel=ChannelType(channel) if channel else None,
        limit=limit,
        offset=offset,
    )
    return [c.model_dump() for c in convs]


@app.get("/api/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conv = await context_tracker.store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return await context_tracker.get_full_context(conversation_id)


@app.post("/api/v1/conversations/{conversation_id}/resolve")
async def resolve_conversation(conversation_id: str, outcome: str = ""):
    conv = await context_tracker.resolve_conversation(conversation_id, outcome)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return {"status": "resolved"}


# ══════════════════════════════════════════════════════════════
#  CONTACTS
# ══════════════════════════════════════════════════════════════

@app.get("/api/v1/contacts")
async def list_contacts():
    return [c.model_dump() for c in await context_tracker.store.list_contacts()]


@app.post("/api/v1/contacts")
async def create_contact(req: ContactCreateRequest):
    from models.schemas import Contact, ContactChannel
    channels = []
    if req.email:
        channels.append(ContactChannel(channel=ChannelType.EMAIL, address=req.email))
    if req.whatsapp:
        channels.append(ContactChannel(channel=ChannelType.WHATSAPP, address=req.whatsapp, preferred=True))
    if req.phone:
        channels.append(ContactChannel(channel=ChannelType.VOICE, address=req.phone))

    contact = Contact(
        name=req.name, role=req.role, organization=req.organization,
        channels=channels, metadata=req.metadata,
    )
    contact = await context_store.upsert_contact(contact)
    return contact.model_dump()


# ══════════════════════════════════════════════════════════════
#  STATE MACHINE
# ══════════════════════════════════════════════════════════════

@app.get("/api/v1/state-maps")
async def list_state_maps():
    return [sm.model_dump() for sm in state_machine.list_maps()]


@app.post("/api/v1/context-change")
async def apply_context_change(req: ContextChangeRequest):
    change = await context_tracker.build_context_change(
        conversation_id=req.conversation_id,
        trigger_type=req.trigger_type,
        trigger_value=req.trigger_value,
        extra_data=req.extra_data,
    )
    results = await orchestrator.apply_context_change(change)
    return {"conversation_id": req.conversation_id, "transitions": results}


# ══════════════════════════════════════════════════════════════
#  QUEUE
# ══════════════════════════════════════════════════════════════

@app.get("/api/v1/queue/stats")
async def queue_stats():
    dispatch_len = await message_queue.queue_length(Queues.DISPATCH)
    dlq_len = await message_queue.queue_length(Queues.DLQ)
    return {
        "dispatch_queue_depth": dispatch_len,
        "dlq_depth": dlq_len,
        "consumer_running": followup_consumer._running,
    }


@app.post("/api/v1/process")
async def trigger_processing():
    backend_data = await backend_connector.get_pending_followups()
    new_followups = await rules_engine.evaluate_scheduled_rules(backend_data)
    jobs = await orchestrator.enqueue_batch(new_followups)
    return {
        "backend_items_evaluated": len(backend_data),
        "followups_enqueued": len(jobs),
    }


# ══════════════════════════════════════════════════════════════
#  WEBHOOKS — WhatsApp
# ══════════════════════════════════════════════════════════════

@app.get("/webhooks/whatsapp")
async def whatsapp_verify(request: Request):
    params = dict(request.query_params)
    challenge = whatsapp_adapter.verify_webhook(params)
    if challenge:
        return JSONResponse(content=int(challenge))
    raise HTTPException(403, "Verification failed")


@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(request: Request):
    """Receive WhatsApp messages with signature verification."""
    body_bytes = await request.body()

    # Verify webhook signature if app_secret is configured
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not whatsapp_adapter.verify_webhook_signature(body_bytes, signature):
        logger.warning("whatsapp_webhook_signature_invalid")
        raise HTTPException(403, "Invalid signature")

    body = json.loads(body_bytes)
    parsed = await whatsapp_adapter.handle_inbound(body)

    if parsed and parsed.get("content"):
        # Auto mark-read
        msg_id = parsed.get("metadata", {}).get("whatsapp_message_id", "")
        if msg_id:
            asyncio.create_task(whatsapp_adapter.mark_read(msg_id))

        result = await orchestrator.handle_inbound_message(
            channel=ChannelType.WHATSAPP,
            sender_address=parsed["sender_address"],
            content=parsed["content"],
            metadata=parsed.get("metadata", {}),
        )
        return result
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════
#  WEBHOOKS — Email
# ══════════════════════════════════════════════════════════════

@app.post("/webhooks/email")
async def email_webhook(request: Request):
    """Receive inbound emails (SendGrid / Mailgun / SES)."""
    body = await request.json()
    parsed = await email_adapter.handle_inbound(body)
    if parsed and parsed.get("content"):
        result = await orchestrator.handle_inbound_message(
            channel=ChannelType.EMAIL,
            sender_address=parsed["sender_address"],
            content=parsed["content"],
            metadata=parsed.get("metadata", {}),
        )
        return result
    return {"status": "ok"}


@app.post("/webhooks/email/bounce")
async def email_bounce(request: Request):
    """Handle email bounce notifications."""
    body = await request.json()
    email_addr = body.get("email", body.get("recipient", ""))
    bounce_type = body.get("type", body.get("bounce_type", "hard"))
    reason = body.get("reason", body.get("diagnostic", ""))
    await email_adapter.handle_bounce(email_addr, bounce_type, reason)
    return {"status": "processed"}


@app.post("/webhooks/email/complaint")
async def email_complaint(request: Request):
    """Handle email spam complaints."""
    body = await request.json()
    email_addr = body.get("email", body.get("complainant", ""))
    await email_adapter.handle_complaint(email_addr)
    return {"status": "processed"}


# ══════════════════════════════════════════════════════════════
#  VOICE — Call Management API
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/voice/transcript")
async def voice_transcript(request: Request):
    """Real-time transcript from Pipecat STT."""
    body = await request.json()
    parsed = await voice_adapter.handle_transcript(
        call_id=body.get("call_id", ""),
        role=body.get("role", "contact"),
        text=body.get("text", ""),
        confidence=body.get("confidence", 1.0),
        is_final=body.get("is_final", True),
    )
    if parsed and body.get("role") == "contact" and body.get("is_final", True):
        result = await orchestrator.handle_inbound_message(
            channel=ChannelType.VOICE,
            sender_address=parsed["sender_address"],
            content=parsed["content"],
            metadata=parsed.get("metadata", {}),
        )
        return result
    return {"status": "ok"}


@app.post("/api/v1/voice/status")
async def voice_call_status(request: Request):
    """
    Universal call status endpoint — auto-detects Twilio/Exotel/Plivo format.

    Accepts JSON or form-encoded payloads from any supported telephony provider.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        body = dict(await request.form())

    # Auto-detect provider from payload keys
    from channels.telephony.factory import TelephonyFactory
    detected = TelephonyFactory.detect_provider_from_webhook(body)

    if detected:
        parser = TelephonyFactory.get_webhook_parser(detected)
        normalized = parser(body)
        call_id = normalized.get("call_id", "")
        status = normalized.get("status", "")

        # Correlate via custom_field for Exotel/Plivo
        custom_field = body.get("custom_field", body.get("CustomField", ""))
        if custom_field and custom_field.startswith("call_"):
            call_id = custom_field
    else:
        call_id = body.get("call_id", body.get("CallSid", ""))
        status = body.get("status", body.get("CallStatus", ""))
        normalized = body

    result = await voice_adapter.handle_call_status(call_id, status, normalized)
    return result or {"status": "ok"}


# ── Provider-Specific Webhook Endpoints ───────────────────────
# These handle provider-specific answer URL and status callback routing.
# The universal /api/v1/voice/status above handles most cases, but
# providers that need separate answer URLs use these.

@app.post("/webhooks/twilio/status")
async def twilio_status_webhook(request: Request):
    """Twilio status callback — form-encoded."""
    body = dict(await request.form())
    from channels.telephony.twilio_client import TwilioClient
    normalized = TwilioClient.parse_status_webhook(body)
    result = await voice_adapter.handle_call_status(
        normalized["call_id"], normalized["status"], normalized
    )
    return result or {"status": "ok"}


@app.post("/webhooks/twilio/answer")
async def twilio_answer_webhook(request: Request):
    """
    Twilio answer URL — returns TwiML to connect audio via Media Stream.

    When contact answers, Twilio fetches this URL for instructions.
    We return <Connect><Stream> to bridge audio to our Pipecat WebSocket.
    """
    body = dict(await request.form())
    call_sid = body.get("CallSid", "")

    # Determine our base URL for the WebSocket connection
    base_url = voice_adapter._config.get("base_url", "http://localhost:8000")
    # Convert http(s) to ws(s) for the WebSocket URL
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    stream_url = f"{ws_url}/ws/twilio-stream/{call_sid}"

    logger.info("twilio_answer", call_sid=call_sid, stream_url=stream_url)

    from fastapi.responses import Response
    return Response(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{stream_url}" />
    </Connect>
</Response>""",
        media_type="application/xml",
    )


@app.websocket("/ws/twilio-stream/{call_sid}")
async def twilio_media_stream(websocket: WebSocket, call_sid: str):
    """
    WebSocket endpoint for Twilio Media Streams.

    Twilio connects here after the TwiML <Connect><Stream> instruction.
    We run a Pipecat pipeline that processes audio in real-time:
      Twilio Audio → Deepgram STT → OpenAI LLM → Cartesia TTS → Twilio Audio
    """
    await websocket.accept()
    logger.info("twilio_ws_connected", call_sid=call_sid)

    from voice.bot import run_voice_bot

    # Build system prompt from any stored call context
    system_prompt = ""
    call_state = voice_adapter._active_calls.get(call_sid)
    if call_state and hasattr(call_state, "tts_queue") and call_state.tts_queue:
        # Use the initial message as context for the greeting
        initial_msg = call_state.tts_queue[0]
        system_prompt = (
            f"You are a professional follow-up agent on a live phone call. "
            f"Your opening line should convey this message: '{initial_msg}'. "
            f"Keep responses to 1-3 short sentences. Use natural spoken language. "
            f"Never use markdown or text formatting. Be warm and conversational."
        )

    try:
        await run_voice_bot(
            websocket=websocket,
            call_sid=call_sid,
            system_prompt=system_prompt,
        )
    except Exception as e:
        logger.error("twilio_ws_error", call_sid=call_sid, error=str(e))
    finally:
        logger.info("twilio_ws_closed", call_sid=call_sid)


@app.post("/webhooks/exotel/status")
async def exotel_status_webhook(request: Request):
    """Exotel status callback."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        body = dict(await request.form())

    from channels.telephony.exotel import ExotelClient
    normalized = ExotelClient.parse_status_webhook(body)

    # Correlate with our internal call_id
    custom_field = body.get("custom_field", body.get("CustomField", ""))
    call_id = custom_field if custom_field.startswith("call_") else normalized["call_id"]

    result = await voice_adapter.handle_call_status(call_id, normalized["status"], normalized)
    return result or {"status": "ok"}


@app.post("/webhooks/exotel/answer")
async def exotel_answer_webhook(request: Request):
    """Exotel answer URL — returns applet XML to bridge to SIP."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        body = dict(await request.form())

    custom_field = body.get("custom_field", body.get("CustomField", ""))
    sip_domain = voice_adapter._config.get("sip_domain", "voice.example.com")
    call_id = custom_field if custom_field.startswith("call_") else body.get("CallSid", "unknown")

    from fastapi.responses import Response
    return Response(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Sip>sip:{call_id}@{sip_domain}</Sip>
    </Dial>
</Response>""",
        media_type="application/xml",
    )


@app.post("/webhooks/plivo/status")
async def plivo_status_webhook(request: Request):
    """Plivo status callback — form-encoded."""
    body = dict(await request.form())
    from channels.telephony.plivo_client import PlivoClient
    normalized = PlivoClient.parse_status_webhook(body)
    result = await voice_adapter.handle_call_status(
        normalized["call_id"], normalized["status"], normalized
    )
    return result or {"status": "ok"}


@app.post("/webhooks/plivo/answer")
async def plivo_answer_webhook(request: Request):
    """Plivo answer URL — returns XML to bridge to SIP."""
    body = dict(await request.form())
    call_uuid = body.get("CallUUID", "")
    sip_domain = voice_adapter._config.get("sip_domain", "voice.example.com")

    from fastapi.responses import Response
    return Response(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <User>sip:{call_uuid}@{sip_domain}</User>
    </Dial>
</Response>""",
        media_type="application/xml",
    )


@app.post("/api/v1/voice/call-ended")
async def voice_call_ended(request: Request):
    """Handle end-of-call callback."""
    body = await request.json()
    call_id = body.get("call_id", body.get("CallSid", ""))
    reason_str = body.get("reason", "completed")
    try:
        reason = CallEndReason(reason_str)
    except ValueError:
        reason = CallEndReason.COMPLETED

    summary = await voice_adapter.handle_call_ended(call_id, reason)
    if summary:
        conv_id = summary.get("conversation_id", "")
        if conv_id:
            conv = await context_tracker.store.get_conversation(conv_id)
            if conv:
                conv.metadata["call_summary"] = summary
                conv.metadata["call_duration"] = summary.get("duration_seconds", 0)
                conv.metadata["call_end_reason"] = summary.get("end_reason", "")
                await context_tracker.store.update_conversation(conv)
    return summary or {"status": "ok"}


@app.post("/api/v1/voice/dtmf")
async def voice_dtmf(request: Request):
    """Handle DTMF tone input during a call."""
    body = await request.json()
    parsed = await voice_adapter.handle_dtmf(
        body.get("call_id", ""),
        body.get("digit", body.get("Digits", "")),
    )
    if parsed:
        result = await orchestrator.handle_inbound_message(
            channel=ChannelType.VOICE,
            sender_address=parsed["sender_address"],
            content=parsed["content"],
            metadata=parsed.get("metadata", {}),
        )
        return result
    return {"status": "ok"}


@app.post("/api/v1/voice/voicemail")
async def voice_voicemail(request: Request):
    """Handle voicemail detection."""
    body = await request.json()
    result = await voice_adapter.handle_voicemail_detected(body.get("call_id", ""))
    return result or {"status": "ok"}


@app.post("/api/v1/voice/quality")
async def voice_quality(request: Request):
    """Receive call quality metrics from Pipecat."""
    body = await request.json()
    await voice_adapter.handle_call_quality(body.get("call_id", ""), body)
    return {"status": "ok"}


@app.get("/api/v1/voice/calls")
async def list_active_calls():
    """List all active voice calls."""
    return voice_adapter.get_active_calls()


class OutboundCallRequest(BaseModel):
    to: str  # Phone number in E.164 format (e.g. +919966990732)
    greeting: str = "Hello! I'm calling to follow up with you. How can I help you today?"
    system_prompt: str = ""


@app.post("/api/v1/voice/outbound")
async def initiate_outbound_call(req: OutboundCallRequest):
    """
    Initiate an outbound voice call via Twilio.

    Flow:
    1. Twilio dials the destination number
    2. When answered, Twilio fetches /webhooks/twilio/answer for TwiML
    3. TwiML instructs Twilio to open a Media Stream to /ws/twilio-stream/{call_sid}
    4. Pipecat pipeline runs: audio → Deepgram STT → OpenAI LLM → Cartesia TTS → audio
    """
    from channels.telephony.twilio_client import TwilioClient

    # Read from voice adapter config (already resolved from settings.yaml + env)
    creds = getattr(voice_adapter, "_config", None) or {}
    twilio_sid = creds.get("telephony_account_sid", "")
    twilio_auth = creds.get("telephony_auth_token", "")
    twilio_phone = creds.get("telephony_phone_number", "")
    base_url = creds.get("base_url", "http://localhost:8000")

    logger.info("outbound_call_config",
                 has_sid=bool(twilio_sid),
                 has_auth=bool(twilio_auth),
                 has_phone=bool(twilio_phone),
                 base_url=base_url)

    if not all([twilio_sid, twilio_auth, twilio_phone]):
        raise HTTPException(
            status_code=500,
            detail=f"Twilio credentials not configured (sid={bool(twilio_sid)}, auth={bool(twilio_auth)}, phone={bool(twilio_phone)})"
        )

    client = TwilioClient(
        account_sid=twilio_sid,
        auth_token=twilio_auth,
        from_number=twilio_phone,
    )

    try:
        logger.info("twilio_call_attempt",
                     to=req.to,
                     answer_url=f"{base_url}/webhooks/twilio/answer",
                     status_url=f"{base_url}/webhooks/twilio/status")
        result = await client.initiate_call(
            to=req.to,
            status_callback_url=f"{base_url}/webhooks/twilio/status",
            sip_endpoint=f"{base_url}/webhooks/twilio/answer",
            ring_timeout=30,
        )
        call_sid = result.get("sid", "")

        # Store call context so the WS handler can pick it up
        voice_adapter._active_calls[call_sid] = type("CallState", (), {
            "tts_queue": [req.greeting],
            "system_prompt": req.system_prompt or "",
        })()

        logger.info("outbound_call_initiated",
                     call_sid=call_sid,
                     to=req.to,
                     from_number=twilio_phone)

        return {
            "status": "call_initiated",
            "call_sid": call_sid,
            "to": req.to,
            "from": twilio_phone,
            "provider": "twilio",
            "answer_url": f"{base_url}/webhooks/twilio/answer",
            "stream_url": f"{base_url}/ws/twilio-stream/{call_sid}",
        }
    except Exception as e:
        error_detail = str(e)
        # Try to extract Twilio-specific error from nested exception
        if hasattr(e, '__cause__') and e.__cause__:
            error_detail = str(e.__cause__)
        elif hasattr(e, 'args') and e.args:
            for arg in e.args:
                if hasattr(arg, 'response'):
                    error_detail = f"Twilio API: {arg.response.text[:300]}"
                    break
        logger.error("outbound_call_failed", error=error_detail)
        raise HTTPException(status_code=500, detail=f"Call initiation failed: {error_detail}")


@app.post("/api/v1/voice/calls/{call_id}/hold")
async def hold_call(call_id: str, request: Request):
    body = await request.json() if await request.body() else {}
    return await voice_adapter.hold_call(call_id, body.get("message", ""))


@app.post("/api/v1/voice/calls/{call_id}/resume")
async def resume_call(call_id: str, request: Request):
    body = await request.json() if await request.body() else {}
    return await voice_adapter.resume_call(call_id, body.get("message", ""))


@app.post("/api/v1/voice/calls/{call_id}/end")
async def end_call(call_id: str):
    return await voice_adapter.end_call(call_id, CallEndReason.AGENT_HANGUP)


# ══════════════════════════════════════════════════════════════
#  WEBSOCKET — Real-time Chat
# ══════════════════════════════════════════════════════════════

@app.websocket("/ws/chat/{contact_id}")
async def websocket_chat(websocket: WebSocket, contact_id: str):
    """
    Real-time chat via WebSocket with structured event handling.

    Client sends JSON events:
      {"type": "message", "content": "hello", "sender_id": "..."}
      {"type": "typing", "is_typing": true, "sender_id": "..."}
      {"type": "pong", "sender_id": "..."}
      {"type": "ack", "message_id": "...", "sender_id": "..."}
      {"type": "presence", "status": "away", "sender_id": "..."}
    """
    await websocket.accept()

    # Extract auth token from query params
    auth_token = websocket.query_params.get("token", "")
    registered = await chat_adapter.register_connection(contact_id, websocket, auth_token)

    if not registered:
        await websocket.close(code=4003, reason="Authentication failed")
        return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                event = {"type": "message", "content": raw, "sender_id": contact_id}

            event.setdefault("sender_id", contact_id)

            # Route through the adapter's inbound parsing
            parsed = await chat_adapter.handle_inbound(event)

            if parsed and parsed.get("content"):
                # Send typing indicator
                await chat_adapter.send_typing(contact_id, True)

                result = await orchestrator.handle_inbound_message(
                    channel=ChannelType.CHAT,
                    sender_address=contact_id,
                    content=parsed["content"],
                    metadata=parsed.get("metadata", {}),
                )

                await chat_adapter.send_typing(contact_id, False)

    except WebSocketDisconnect:
        await chat_adapter.unregister_connection(contact_id)
    except asyncio.CancelledError:
        await chat_adapter.unregister_connection(contact_id)
    except Exception as e:
        logger.error("websocket_error", contact_id=contact_id, error=str(e))
        await chat_adapter.unregister_connection(contact_id)


# ══════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
