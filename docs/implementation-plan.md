# ConverseAgent — Implementation Plan

## What Exists vs What Needs Building

The codebase has **complete application logic** (16,683 lines, 359 tests) but is currently a **simulation layer** — the adapters manage state and produce correct data structures, but don't make real network calls. Every external integration point has a comment like `# Production: Pipecat session creation here` or uses `MockBackendConnector`.

This plan converts the simulation into a live system, organized into 8 sprints. Each sprint is independently deployable — you can go live with text channels first while voice is still in development.

---

## Sprint 0 — Local Development Environment (Days 1-3)

### Goal: Run the full stack locally with real API keys

**0.1 — Environment Setup**

```bash
git clone <repo>
cd converse-agent
cp config/settings.example.yaml config/settings.yaml
cp .env.example .env
```

Edit `.env` with real API keys:

```env
# LLM
CONVERSE_LLM_API_KEY=sk-ant-...

# STT / TTS (voice — can defer to Sprint 5)
STT_API_KEY=deepgram_...
TTS_API_KEY=cartesia_...

# Telephony (can defer to Sprint 5)
EXOTEL_SID=...
EXOTEL_TOKEN=...
EXOTEL_CALLER_ID=+919...

# Channels
WHATSAPP_PHONE_ID=...
WHATSAPP_ACCESS_TOKEN=...
WHATSAPP_VERIFY_TOKEN=...
EMAIL_SMTP_PASSWORD=...

# Backend
BACKEND_API_TOKEN=...
```

**0.2 — Start Services**

```bash
docker-compose up -d redis
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --port 8000
```

**0.3 — Verify Baseline**

```bash
# All 359 tests pass
python -m pytest tests/ -q

# Health check
curl http://localhost:8000/health

# Stats
curl http://localhost:8000/api/v1/stats
```

**0.4 — Deliverable**: Local stack running, health endpoint returning channel list.

---

## Sprint 1 — Database Persistence (Days 4-8)

### Goal: Replace in-memory stores with PostgreSQL so data survives restarts

Currently `ContextStore` is an in-memory dict. Every restart loses all conversations, contacts, and follow-ups.

**1.1 — Add PostgreSQL Dependencies**

```bash
# requirements.txt — add:
asyncpg>=0.29.0
alembic>=1.13.0
```

```yaml
# docker-compose.yml — add:
  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: converse_agent
      POSTGRES_USER: converse
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-dev_password}
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U converse"]
      interval: 10s
```

**1.2 — Create SQLAlchemy Models**

New file: `database/models.py`

Map existing Pydantic schemas to SQLAlchemy ORM models:

| Pydantic Schema | SQL Table | Key Columns |
|---|---|---|
| `Contact` | `contacts` | id, name, role, organization, channels (JSONB), metadata (JSONB) |
| `Conversation` | `conversations` | id, contact_id (FK), status, active_channel, messages (JSONB), created_at |
| `FollowUp` | `followups` | id, contact_id (FK), conversation_id (FK), rule_id, status, priority, business_context (JSONB), attempt_count, next_attempt_at |
| `StateBinding` | `state_bindings` | id, conversation_id (FK), process_type, entity_type, entity_id, current_state, state_data (JSONB) |
| `Message` | `messages` | id, conversation_id (FK), direction, channel, content, timestamp, metadata (JSONB) |

Use JSONB for flexible fields — avoids schema migrations every time business context changes.

**1.3 — Create Database Session Manager**

New file: `database/session.py`

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

engine = create_async_engine(settings.database.url, echo=settings.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session
```

**1.4 — Replace ContextStore**

Modify `context/tracker.py` — replace the dict-based `ContextStore` with `PostgresContextStore`:

| Method | Current (In-Memory) | New (PostgreSQL) |
|---|---|---|
| `get_contact(id)` | `self._contacts[id]` | `SELECT * FROM contacts WHERE id = $1` |
| `get_conversation(id)` | `self._conversations[id]` | `SELECT * FROM conversations WHERE id = $1` |
| `find_conversation_by_address(addr)` | Linear scan | `SELECT c.* FROM conversations c JOIN contacts ct ON ... WHERE ct.channels @> ...` |
| `update_followup(f)` | `self._followups[f.id] = f` | `UPDATE followups SET ... WHERE id = $1` |
| `get_conversation_bindings(conv_id)` | Filter list | `SELECT * FROM state_bindings WHERE conversation_id = $1` |

Keep the abstract `ContextStore` interface — the `ContextTracker` doesn't change, only the store implementation swaps.

**1.5 — Alembic Migrations**

```bash
alembic init database/migrations
alembic revision --autogenerate -m "initial_schema"
alembic upgrade head
```

**1.6 — Deliverable**: Restart the server, previous conversations and contacts persist. All 359 tests still pass (tests use a test-specific in-memory store fixture).

---

## Sprint 2 — Backend System Integration (Days 9-15)

### Goal: Connect to your actual ERP/CRM so follow-ups come from real business data

The `RESTBackendConnector` exists but needs to be wired to your specific backend.

**2.1 — Define Your Backend Endpoints**

Edit `config/settings.yaml` to point at your real system:

```yaml
backend:
  type: "rest"
  base_url: "https://your-erp.example.com/api/v1"
  auth_type: "bearer"
  auth_credentials:
    token: "${BACKEND_API_TOKEN}"

  endpoints:
    # Required — these feed the follow-up system
    get_followups: "/followups/pending"
    get_contact: "/contacts/{contact_id}"
    get_context: "/followups/{followup_id}/context"
    update_followup: "/followups/{followup_id}/status"
    search_contacts: "/contacts/search"

    # Optional — used by dialogue flow tool_call steps
    check_payment: "/payments/{invoice_id}/status"
    check_stock: "/inventory/check"
    track_shipment: "/shipments/{shipment_id}/track"
    get_order_issues: "/orders/{order_id}/issues"
```

**2.2 — Backend Data Contract**

Your backend needs to return data in this shape (or you adapt the connector):

```json
// GET /followups/pending
[
  {
    "id": "f_001",
    "contact_id": "c_123",
    "reason": "Payment overdue 15 days",
    "business_context": {
      "invoice_number": "INV-2026-0042",
      "amount": 52400,
      "currency": "INR",
      "due_date": "2026-01-30",
      "days_overdue": 17
    },
    "priority": "high",
    "channel_priority": ["voice", "whatsapp", "email"],
    "process_type": "payment_collection",
    "entity_type": "invoice",
    "entity_id": "INV-2026-0042"
  }
]
```

```json
// GET /contacts/{contact_id}
{
  "id": "c_123",
  "name": "Priya Sharma",
  "role": "CFO",
  "organization": "Acme Corp",
  "channels": [
    {"channel": "voice", "address": "+919876543210"},
    {"channel": "whatsapp", "address": "+919876543210"},
    {"channel": "email", "address": "priya@acme.com"}
  ]
}
```

If your backend uses different field names, add a mapping layer in the connector:

```python
# backend/connector.py — in RESTBackendConnector
async def get_contact(self, contact_id: str) -> Optional[dict]:
    raw = await self._get(f"/contacts/{contact_id}")
    # Map your backend's field names to ConverseAgent's schema
    return {
        "id": raw["customer_id"],           # your field → our field
        "name": raw["full_name"],
        "role": raw.get("designation", ""),
        "organization": raw.get("company_name", ""),
        "channels": self._map_channels(raw),
    }
```

**2.3 — Status Writeback**

When a conversation resolves (payment confirmed, order acknowledged), the orchestrator calls `update_followup_status`. This writes back to your backend:

```python
# This already happens automatically via the state machine.
# When state transitions to "confirmed" or "closed",
# the action executor calls:
await self.backend.update_followup_status(
    followup_id="f_001",
    status="resolved",
    outcome="Payment confirmed by contact on call"
)
```

Make sure your backend endpoint accepts this and updates the invoice/order accordingly.

**2.4 — Backend Event Ingestion**

Your backend pushes events to ConverseAgent when things happen (order placed, shipment dispatched, payment received):

```bash
# Your backend calls this when an invoice is created:
curl -X POST http://converse-agent:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_name": "invoice_overdue",
    "data": {
      "invoice_number": "INV-2026-0042",
      "contact_id": "c_123",
      "amount": 52400,
      "days_overdue": 3
    }
  }'
```

The Rules Engine evaluates this event against your configured rules and creates follow-ups automatically.

**2.5 — Cron-Based Polling (Alternative)**

If your backend can't push events, add a periodic poller:

```python
# New: backend/poller.py
class BackendPoller:
    """Periodically fetch pending follow-ups from the backend."""
    
    async def poll_cycle(self):
        raw_followups = await self.backend.get_pending_followups()
        for raw in raw_followups:
            # Check if we already have this follow-up
            existing = await self.context.store.get_followup_by_external_id(raw["id"])
            if existing:
                continue
            # Create new follow-up and enqueue
            followup = self._map_to_followup(raw)
            contact = await self._ensure_contact(raw["contact_id"])
            conversation = await self.context.create_conversation(contact)
            await self.orchestrator.enqueue_followup(followup)
```

Add to the FastAPI lifespan as a background task running every N minutes.

**2.6 — Deliverable**: Follow-ups from your real backend flow through the system. Status updates write back. Manual test: trigger a follow-up from your ERP, see it dispatched via email/WhatsApp, reply, watch the state machine transition, verify the status update reaches your ERP.

---

## Sprint 3 — Text Channels Live (Days 16-22)

### Goal: WhatsApp, Email, and Chat channels handling real conversations

**3.1 — WhatsApp**

The `WhatsAppAdapter` is fully built with Meta Graph API integration, webhook verification, message deduplication, and typing indicators. To go live:

1. Set up a Meta Business account and WhatsApp Business API
2. Configure webhook URL: `https://your-domain.com/webhooks/whatsapp`
3. Set verify token in settings
4. Test with the Meta webhook test tool

The adapter handles: text messages, template messages (HSM), media messages, read receipts, delivery status, and interactive buttons.

**3.2 — Email**

The `EmailAdapter` uses `aiosmtplib` for outbound and expects webhook callbacks for inbound (SendGrid/Mailgun):

1. Configure SMTP credentials in settings
2. Set up SendGrid/Mailgun inbound parse webhook → `https://your-domain.com/webhooks/email`
3. Configure bounce handler → `https://your-domain.com/webhooks/email/bounce`

**3.3 — Chat (WebSocket)**

The `ChatAdapter` manages WebSocket connections with reconnection, heartbeat, and presence tracking:

1. Connect from your frontend: `ws://your-domain.com/ws/chat/{contact_id}?token=...`
2. Send JSON messages: `{"type": "message", "content": "hello"}`
3. Receive structured responses with typing indicators

**3.4 — End-to-End Verification**

For each channel, verify the full loop:

```
Backend event → Rules Engine → Follow-up created → Queue → Consumer dispatches
→ Channel adapter sends message → Contact replies → Webhook receives reply
→ Orchestrator routes to Conversation → LLM generates response → Reply sent
→ State machine transitions → Backend status updated
```

**3.5 — Deliverable**: Real conversations happening over WhatsApp/Email/Chat. Dialogue flows executing correctly. State transitions tracked.

---

## Sprint 4 — State Machine & Rules Tuning (Days 23-28)

### Goal: Business logic matches your actual processes

**4.1 — Customize State Maps**

Edit `config/settings.yaml` state_maps section. The example has `payment_collection` and `order_fulfillment`. Add your specific processes:

```yaml
state_maps:
  - process_type: "payment_collection"
    states: [pending, reminded, acknowledged, promised, partial_paid, confirmed, overdue, escalated, closed]
    initial_state: "pending"
    terminal_states: ["confirmed", "closed"]
    transitions:
      # Customize these to match your business rules
      - from_states: ["reminded"]
        to_state: "promised"
        trigger: { type: "intent", values: ["promise_to_pay", "will_pay"] }
        conditions:
          - field: "extracted_info.payment_date"
            operator: "exists"
        actions:
          - type: "update_backend"
            endpoint: "record_promise"
            payload:
              invoice_id: "{{entity_id}}"
              promised_date: "{{extracted_info.payment_date}}"
          - type: "schedule_check"
            delay_minutes: 4320  # Check in 3 days
```

**4.2 — Customize Dialogue Flows**

Edit `config/flows/dialogue_flows.yaml`. These are multi-step conversation templates:

```yaml
flows:
  - id: payment_reminder
    steps:
      - id: check_payment
        type: tool_call
        tool:
          endpoint: check_payment
          payload_template: { invoice_id: "business_context.invoice_number" }
          result_key: payment_status
      - id: payment_branch
        type: branch
        arms:
          - conditions:
              - field: payment_status.paid
                operator: eq
                value: true
            goto: already_paid
          - goto: send_reminder
      - id: send_reminder
        type: generate
        system_prompt: |
          Call {{contact_name}} about invoice {{invoice_number}}.
          Amount: {{amount}} {{currency}}. Due: {{due_date}}.
          Be professional but firm. Ask for a specific payment date.
```

**4.3 — Test State Transitions**

Use the API to simulate transitions and verify correct behavior:

```bash
# Simulate an intent trigger
curl -X POST http://localhost:8000/api/v1/context/trigger \
  -d '{"conversation_id": "conv_1", "trigger_type": "intent", "trigger_value": "promise_to_pay", "extra_data": {"extracted_info": {"payment_date": "2026-02-20"}}}'

# Check state
curl http://localhost:8000/api/v1/conversations/conv_1
```

**4.4 — Deliverable**: State maps match your business processes. Dialogue flows produce correct messages. Escalation paths work.

---

## Sprint 5 — Voice Pipeline Integration (Days 29-40)

### Goal: Live voice calls working end-to-end at ₹4/min

This is the biggest sprint. It connects the voice subsystem modules (already built and tested) to real providers.

**5.1 — Exotel Telephony Integration**

New file: `channels/telephony/exotel.py`

```python
class ExotelClient:
    """Exotel SIP trunking client."""

    def __init__(self, sid: str, token: str, caller_id: str):
        self.base_url = f"https://api.exotel.com/v2/accounts/{sid}"
        self._auth = (sid, token)
        self.caller_id = caller_id

    async def initiate_call(
        self, to: str, status_callback: str, sip_endpoint: str
    ) -> dict:
        """
        Place outbound call via Exotel.
        
        Flow:
        1. Exotel dials `to` number
        2. On answer, connects to our Pipecat SIP endpoint
        3. Status callbacks hit our /api/v1/voice/status endpoint
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/calls",
                auth=self._auth,
                json={
                    "from": self.caller_id,
                    "to": to,
                    "url": sip_endpoint,            # Pipecat SIP URI
                    "status_callback": status_callback,
                    "status_callback_events": ["initiated", "ringing", "in-progress", "completed"],
                    "ring_timeout": 30,
                    "record": False,                 # We handle recording ourselves
                },
            )
            return resp.json()

    async def end_call(self, call_sid: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/calls/{call_sid}",
                auth=self._auth,
                json={"status": "completed"},
            )
            return resp.json()
```

**5.2 — Pipecat Process Manager**

New file: `voice/server.py`

This is the actual voice pipeline runtime — one Pipecat process per call:

```python
class VoicePipelineManager:
    """
    Manages Pipecat pipeline instances per active call.
    
    Lifecycle:
    1. Orchestrator calls VoiceAdapter._initiate_outbound_call()
    2. VoiceAdapter calls ExotelClient.initiate_call()
    3. Exotel dials contact, on answer sends SIP INVITE to our SIP endpoint
    4. SIP endpoint triggers create_pipeline(call_id, contact, context)
    5. Pipeline starts: STT WebSocket + LLM stream + TTS WebSocket
    6. Audio flows: Exotel RTP ↔ Pipecat ↔ STT/TTS providers
    7. Transcript callbacks hit /api/v1/voice/transcript
    8. On call end, pipeline tears down, summary posted to /api/v1/voice/call-ended
    """

    def __init__(self, registry: VoiceProviderRegistry, latency_tracker: AggregateLatencyTracker):
        self.registry = registry
        self.latency_tracker = latency_tracker
        self._pipelines: dict[str, PipecatPipeline] = {}

    async def create_pipeline(self, call_id: str, contact: Contact, context: dict) -> dict:
        """Create and start a new voice pipeline for a call."""
        builder = VoicePipelineBuilder(self.registry, self.latency_tracker)
        config = builder.build(
            contact=contact,
            context=context,
            initial_message=context.get("initial_message", ""),
            quality_preset="low_latency",
        )

        # Create Pipecat pipeline from config
        pipeline = await self._build_pipecat_pipeline(config)
        self._pipelines[call_id] = pipeline

        # Start pipeline (connects WebSockets to STT, LLM, TTS)
        await pipeline.start()

        # Create turn-taking engine for this call
        turn_engine = create_turn_taking_engine(config.turn_taking)

        # Create latency tracker for this call
        call_tracker = self.latency_tracker.create_call_tracker(call_id)

        return {"status": "pipeline_ready", "call_id": call_id}

    async def _build_pipecat_pipeline(self, config: PipelineConfig):
        """
        Wire Pipecat processors from our PipelineConfig.
        
        Uses pipecat-ai framework:
        - DeepgramSTTService (WebSocket to Deepgram)
        - AnthropicLLMService (streaming to Claude)
        - CartesiaTTSService or ElevenLabsTTSService (WebSocket to TTS)
        - SIPInputTransport / SIPOutputTransport (audio from/to Exotel)
        """
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.services.deepgram import DeepgramSTTService
        from pipecat.services.anthropic import AnthropicLLMService
        from pipecat.services.cartesia import CartesiaTTSService
        from pipecat.transports.network.sip import SIPTransport

        stt = DeepgramSTTService(
            api_key=config.stt["api_key"],
            **{k: v for k, v in config.stt.items() if k not in ("api_key", "provider")},
        )
        llm = AnthropicLLMService(
            api_key=config.llm["api_key"],
            model=config.llm["model"],
            system_prompt=config.llm["system_prompt"],
        )
        tts = CartesiaTTSService(
            api_key=config.tts["api_key"],
            voice_id=config.tts.get("voice_id", ""),
        )
        transport = SIPTransport(...)

        pipeline = Pipeline([
            transport.input(),
            stt,
            llm,
            tts,
            transport.output(),
        ])
        return pipeline

    async def destroy_pipeline(self, call_id: str):
        pipeline = self._pipelines.pop(call_id, None)
        if pipeline:
            await pipeline.stop()
            self.latency_tracker.remove_call(call_id)
```

**5.3 — Wire VoiceAdapter to Real Providers**

Update `channels/voice_adapter.py` — replace the simulation stubs:

```python
# In _initiate_outbound_call():

# OLD (simulation):
# call_state.set_status(CallStatus.RINGING)

# NEW (real):
exotel_result = await self._exotel.initiate_call(
    to=phone,
    status_callback=f"{self._webhook_base}/api/v1/voice/status",
    sip_endpoint=f"sip:{call_id}@{self._pipecat_sip_host}",
)
call_state.sip_call_id = exotel_result["sid"]
call_state.set_status(CallStatus.RINGING)

# Pipeline will be created when Exotel connects (via status callback)
```

```python
# In handle_call_status(), when status == "in-progress":

# Create the Pipecat pipeline now that the call is connected
contact = await self._context_store.get_contact(call_state.contact_id)
context = {
    "business_context": call_state.metadata.get("business_context", {}),
    "initial_message": call_state.tts_queue[0] if call_state.tts_queue else "",
}
await self._pipeline_manager.create_pipeline(
    call_state.call_id, contact, context
)
call_state.set_status(CallStatus.CONNECTED)
```

**5.4 — Turn-Taking Integration**

Wire the turn-taking engine into the Pipecat pipeline's event handlers:

```python
# In the Pipecat pipeline event loop:

async def on_stt_transcript(self, call_id: str, text: str, is_final: bool):
    """Called by Deepgram when speech is transcribed."""
    engine = self._turn_engines[call_id]
    tracker = self._latency_tracker.get_call_tracker(call_id)

    # Track STT latency
    tracker.end(PipelineStage.STT)

    if not is_final:
        # Check for fillers in interim results
        fillers = detect_fillers(text)
        if fillers:
            result = engine.on_filler_detected(fillers[0])
            # Extend endpointing — tell Deepgram to wait longer
            await self._update_endpointing(call_id, result["new_endpointing_ms"])
        return

    # Final transcript
    result = engine.on_contact_speech_end(text)

    # Should we generate a backchannel?
    if "backchannel" in result:
        await self._quick_tts(call_id, result["backchannel"])

    # Should we respond? (endpointing timer manages this)
    if result.get("likely_complete"):
        # Start LLM generation
        tracker.start(PipelineStage.LLM_TTFB)
        # Feed to LLM via Pipecat's pipeline
        # LLM → TTS happens automatically through the pipeline

async def on_contact_speech_start(self, call_id: str):
    """Called by Deepgram VAD when contact starts speaking."""
    engine = self._turn_engines[call_id]
    result = engine.on_contact_speech_start()

    if result["action"] == "barge_in" and result["stop_tts"]:
        # Immediately stop TTS playback
        await self._stop_tts(call_id)

async def on_silence(self, call_id: str, duration_s: float):
    """Called periodically when no speech is detected."""
    engine = self._turn_engines[call_id]
    result = engine.on_silence_tick()

    if result and result["action"] == "silence_prompt":
        await self._speak(call_id, result["prompt"])
    elif result and result["action"] == "end_call":
        await self.end_call(call_id, CallEndReason.SILENCE_TIMEOUT)
```

**5.5 — Provider Registry Setup**

In `voice/server.py` startup:

```python
# Initialize provider registry with ₹4/min stack
registry = VoiceProviderRegistry()

registry.register_stt("default", STTConfig(
    provider=STTProvider.DEEPGRAM,
    model="nova-2",
    api_key=settings.channels["voice"].credentials["stt_api_key"],
    endpointing_ms=250,
    vad_enabled=True,
    interim_results=True,
    smart_format=True,
    filler_words=True,
    keywords=["invoice", "payment", "rupees", "lakh", "crore"],  # domain keywords
    keyword_boost=1.5,
))

registry.register_tts("default", TTSConfig(
    provider=TTSProvider.CARTESIA,
    api_key=settings.channels["voice"].credentials["tts_api_key"],
    voice_id="<hindi-english-bilingual-voice-id>",
    model="sonic-3",
    speed=1.0,
    output_format=AudioCodec.PCM_16K,
))

registry.set_telephony(TelephonyConfig(
    provider=TelephonyProvider.EXOTEL,
    account_sid=settings.channels["voice"].credentials["telephony_account_sid"],
    auth_token=settings.channels["voice"].credentials["telephony_auth_token"],
    phone_number=settings.channels["voice"].credentials["telephony_phone_number"],
    status_callback_url=f"{settings.webhook_base_url}/api/v1/voice/status",
))
```

**5.6 — Test Voice Pipeline**

```bash
# 1. Unit test — verify pipeline config generation
python -m pytest tests/test_voice_subsystem.py -v

# 2. Integration test — call yourself
curl -X POST http://localhost:8000/api/v1/followups/trigger \
  -d '{"contact_id": "your_test_contact", "reason": "Test call", "channel_priority": ["voice"]}'

# 3. Monitor latency
curl http://localhost:8000/api/v1/voice/calls
```

**5.7 — Deliverable**: Make a real voice call to your own phone. Agent greets you, you respond, agent replies. Latency tracked per-stage. Transcript saved to database.

---

## Sprint 6 — Voice Quality Tuning (Days 41-48)

### Goal: Natural conversations, not robotic back-and-forth

**6.1 — Tune Endpointing for Indian English**

Indian English speech patterns have different pausing characteristics than American English. Adjust:

```python
endpointing = EndpointingConfig(
    base_ms=350,            # Slightly longer than default 300 — Indian English has more mid-sentence pauses
    question_ms=250,        # Still short after agent asks a question
    after_filler_ms=1000,   # Hindi fillers ("matlab", "woh", "acha") — give more time
    number_dictation_ms=1500, # "Paanch lakh baavan hazaar" takes longer than "five lakh fifty two"
    mid_sentence_ms=700,
    emotional_ms=600,
)
```

**6.2 — Add Hindi Fillers**

Update `voice/turn_taking.py`:

```python
FILLER_WORDS = {
    # English
    "um", "uh", "like", "you know", "I mean", "sort of", "well", "so", "actually", "let me think",
    # Hindi / Hinglish
    "matlab", "woh", "acha", "haan", "toh", "basically", "actually", "dekho",
    "ek minute", "ruko", "theek hai",
}
```

**6.3 — Voice Prompt Tuning**

Update `VoicePromptBuilder` for Indian business context:

```python
# Add to VOICE_RULES in voice/pipeline.py:
INDIAN_BUSINESS_RULES = """
CULTURAL CONTEXT:
- Address as "Sir" / "Ma'am" or by name with "ji" suffix if appropriate
- State amounts in Indian numbering: "two lakh fifty-two thousand four hundred" not "two hundred fifty-two thousand four hundred"
- Reference dates in Indian format: "twentieth February" not "February twentieth"
- If contact speaks Hindi/Hinglish, mirror their language naturally
- Be respectful of time — acknowledge if calling during lunch or evening hours
"""
```

**6.4 — Latency Optimization**

Monitor `CallLatencyTracker` output and tune:

```python
# If LLM TTFB consistently > 300ms, switch to Haiku
if tracker.get_optimization_hints():
    for hint in hints:
        if hint["hint"] == "switch_to_faster_model":
            # Already on Haiku — check prompt size
            # Reduce system prompt, enable prompt caching
            pass
        elif hint["hint"] == "increase_streaming_optimization":
            # Increase Cartesia streaming buffer
            pass
```

**6.5 — Silence Escalation Tuning**

Indian callers may take longer to respond (checking files, asking colleagues). Widen the silence thresholds:

```python
silence = SilenceConfig(
    gentle_prompt_s=8,    # Was 5 — give more time before "Are you still there?"
    check_in_s=15,        # Was 10
    final_check_s=25,     # Was 20
    goodbye_s=35,         # Was 30
)
```

**6.6 — Deliverable**: Natural-sounding voice conversations. Filler detection working for Hindi. Latency consistently under 600ms. Silence handling feels human-like.

---

## Sprint 7 — Production Deployment (Days 49-56)

### Goal: Running on AWS with monitoring and alerting

**7.1 — Docker Images**

Create three images from the same codebase:

```dockerfile
# Dockerfile.api — same as existing Dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Dockerfile.worker
CMD ["python", "-m", "job_queue.worker"]

# Dockerfile.voice
CMD ["python", "-m", "voice.server"]
```

Or use a single Dockerfile with CMD override in ECS task definitions.

**7.2 — AWS CDK Deployment**

Follow the architecture from the AWS deployment doc. For the ₹4/min budget stack:

```python
# cdk/compute_stack.py (simplified)

# API Service
api_service = ecs.FargateService(self, "ApiService",
    cluster=cluster,
    task_definition=api_task,
    desired_count=2,
    assign_public_ip=True,  # No NAT Gateway
)

# Worker Service
worker_service = ecs.FargateService(self, "WorkerService",
    cluster=cluster,
    task_definition=worker_task,
    desired_count=2,
)

# Voice Service (EC2 for cost, Spot for savings)
voice_asg = autoscaling.AutoScalingGroup(self, "VoiceASG",
    instance_type=ec2.InstanceType("c7g.large"),
    spot_price="0.03",  # Spot instances
    min_capacity=1,
    max_capacity=5,
)
```

**7.3 — Monitoring Setup**

Push custom metrics from `AggregateLatencyTracker` to CloudWatch:

```python
# voice/monitoring.py
import boto3
cloudwatch = boto3.client("cloudwatch", region_name="ap-south-1")

async def publish_voice_metrics(tracker: AggregateLatencyTracker):
    stats = tracker.get_all_stats()
    cloudwatch.put_metric_data(
        Namespace="ConverseAgent/Voice",
        MetricData=[
            {"MetricName": "PipelineLatencyP90", "Value": stats["total"]["p90_ms"], "Unit": "Milliseconds"},
            {"MetricName": "ActiveCalls", "Value": stats["active_calls"], "Unit": "Count"},
            {"MetricName": "STT_TTFB_P90", "Value": stats["stt"]["p90_ms"], "Unit": "Milliseconds"},
            {"MetricName": "LLM_TTFB_P90", "Value": stats["llm_ttfb"]["p90_ms"], "Unit": "Milliseconds"},
            {"MetricName": "TTS_TTFB_P90", "Value": stats["tts_ttfb"]["p90_ms"], "Unit": "Milliseconds"},
        ],
    )
```

**7.4 — Alarms**

```python
# CDK alarm definitions
cloudwatch.Alarm(self, "LatencyAlarm",
    metric=voice_latency_p90,
    threshold=800,
    evaluation_periods=3,
    alarm_actions=[sns_pagerduty_topic],
)
```

**7.5 — Deliverable**: System running on AWS. Monitoring dashboard showing call metrics. Alerts configured. Cost tracking enabled.

---

## Sprint 8 — Scale & Harden (Days 57-65)

### Goal: Production-ready for 50+ concurrent calls

**8.1 — Load Testing**

```python
# tests/load/voice_load_test.py
# Simulate 50 concurrent calls using Locust or custom async script
# Each simulated call:
# 1. POST /api/v1/followups/trigger (initiate call)
# 2. Simulate 10 transcript callbacks (contact speech)
# 3. Verify latency metrics stay under budget
# 4. POST /api/v1/voice/call-ended
```

**8.2 — Graceful Degradation**

```python
# In VoicePipelineBuilder.build():
# If latency tracker shows degradation, auto-switch to faster providers
if builder.should_degrade_quality(call_id):
    # Switch from Sonnet → Haiku
    # Reduce max_response_words from 80 → 50
    # Increase endpointing to reduce false triggers
```

**8.3 — Circuit Breaker Per Provider**

The channel adapters already have circuit breakers. Add provider-level breakers:

```python
# If Cartesia TTS is down, fall back to ElevenLabs
# If Deepgram STT is down, fall back to Google
# VoiceProviderRegistry already supports named profiles — 
# the fallback is: registry.get_tts("fallback_elevenlabs")
```

**8.4 — Call Recording & Compliance**

For Indian regulatory compliance (TRAI), you may need call recordings:

```python
# Enable recording via Exotel (stored on their side)
# Or record locally in Pipecat pipeline and push to S3
recording_config = {
    "enabled": True,
    "storage": "s3",
    "bucket": "converse-agent-recordings",
    "retention_days": 90,
    "encryption": "aws:kms",
}
```

**8.5 — Deliverable**: System handles 50 concurrent calls without degradation. Fallback providers configured. Recording compliance handled. Load test report showing p90 latency under target.

---

## Integration Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR BACKEND (ERP/CRM)                      │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ POST events │  │ GET contacts │  │ PUT followup/status    │  │
│  │ to /events  │  │ GET context  │  │ (writeback on resolve) │  │
│  └──────┬──────┘  └──────▲───────┘  └────────────▲───────────┘  │
│         │                │                       │               │
└─────────│────────────────│───────────────────────│───────────────┘
          │                │                       │
          ▼                │                       │
┌─────────────────────────────────────────────────────────────────┐
│                      CONVERSE AGENT                              │
│                                                                  │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────┐    │
│  │  FastAPI  │───▶│ Orchestrator│───▶│ BackendConnector     │────┤
│  │  (api/)   │    │  (core/)    │    │ (REST/GraphQL/gRPC)  │    │
│  └────┬─────┘    └──────┬──────┘    └──────────────────────┘    │
│       │                 │                                        │
│  ┌────▼─────┐    ┌──────▼──────┐    ┌──────────────────────┐    │
│  │  Rules   │    │   State     │    │   Dialogue Flows     │    │
│  │  Engine  │    │   Machine   │    │   (templates/)       │    │
│  │ (rules/) │    │ (context/)  │    │   Agentic Planner    │    │
│  └──────────┘    └─────────────┘    └──────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  CHANNEL ADAPTERS                         │    │
│  │                                                          │    │
│  │  ┌─────────┐ ┌──────────┐ ┌──────┐ ┌─────┐ ┌────────┐  │    │
│  │  │WhatsApp │ │  Email   │ │ Chat │ │ SMS │ │ Voice  │  │    │
│  │  │ (Meta)  │ │(SendGrid)│ │ (WS) │ │(SNS)│ │(below) │  │    │
│  │  └─────────┘ └──────────┘ └──────┘ └─────┘ └────┬───┘  │    │
│  └──────────────────────────────────────────────────│──────┘    │
│                                                     │            │
│  ┌──────────────────────────────────────────────────▼────────┐  │
│  │                    VOICE SUBSYSTEM                         │  │
│  │                                                           │  │
│  │  ┌───────────────┐   ┌───────────────┐  ┌─────────────┐  │  │
│  │  │VoiceProvider  │   │VoicePipeline  │  │  Pipecat    │  │  │
│  │  │Registry       │──▶│Builder        │─▶│  Runtime    │  │  │
│  │  │               │   │               │  │             │  │  │
│  │  │• Deepgram STT │   │• Voice prompts│  │• STT (WS)   │  │  │
│  │  │• Cartesia TTS │   │• Turn-taking  │  │• LLM (WS)   │  │  │
│  │  │• Exotel Tel.  │   │• Latency cfg  │  │• TTS (WS)   │  │  │
│  │  └───────────────┘   └───────────────┘  │• SIP I/O    │  │  │
│  │                                         └──────┬──────┘  │  │
│  │  ┌───────────────┐   ┌───────────────┐         │         │  │
│  │  │TurnTaking     │   │Latency        │◄────────┘         │  │
│  │  │Engine         │   │Tracker        │                   │  │
│  │  │               │   │               │                   │  │
│  │  │• Floor mgmt   │   │• Per-stage    │                   │  │
│  │  │• Barge-in     │   │• p50/p90/p99  │                   │  │
│  │  │• Backchannel  │   │• Budget check │                   │  │
│  │  │• Silence esc. │   │• Opt. hints   │                   │  │
│  │  └───────────────┘   └───────────────┘                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐                    │
│  │  Redis Streams   │   │  PostgreSQL      │                    │
│  │  (Job Queue)     │   │  (Persistence)   │                    │
│  └──────────────────┘   └──────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
          │                           │
          ▼                           ▼
┌─────────────────┐          ┌─────────────────┐
│  Exotel / PSTN  │          │  Deepgram       │
│  ₹0.60/min      │          │  $0.0036/min    │
├─────────────────┤          ├─────────────────┤
│  Contact's      │          │  Cartesia       │
│  Phone          │          │  ~$0.014/min    │
└─────────────────┘          ├─────────────────┤
                             │  Claude Haiku   │
                             │  ~$0.002/min    │
                             └─────────────────┘
```

---

## Timeline Summary

| Sprint | Days | What | Dependency |
|---|---|---|---|
| 0 | 1-3 | Local dev environment | None |
| 1 | 4-8 | PostgreSQL persistence | Sprint 0 |
| 2 | 9-15 | Backend integration | Sprint 1 |
| 3 | 16-22 | Text channels live | Sprint 2 |
| 4 | 23-28 | State machine tuning | Sprint 3 |
| 5 | 29-40 | Voice pipeline live | Sprint 2 |
| 6 | 41-48 | Voice quality tuning | Sprint 5 |
| 7 | 49-56 | AWS production deploy | Sprints 3+6 |
| 8 | 57-65 | Scale and harden | Sprint 7 |

Sprints 3-4 (text channels) and 5-6 (voice) can run in parallel with two engineers. Critical path is Sprint 2 (backend integration) because both text and voice depend on real business data flowing in.

**Total: ~65 working days (13 weeks) to full production.** Text channels go live at week 5, voice at week 10, production-hardened by week 13.
