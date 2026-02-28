# Implementation Plan: ₹4/min Voice Agent Stack

**Target:** Reduce voice call cost from ~₹6.8/min (premium stack) to ≤ ₹4/min  
**Primary lever:** Indian telephony provider (Exotel) — saves ₹1.10/min alone  
**Secondary levers:** Cartesia TTS default, Haiku 4.5 for voice, prompt caching  

---

## Architecture Overview

```
Current Flow:
  Twilio (₹1.70/min) → ALB → Pipecat → Deepgram STT → Claude Sonnet → ElevenLabs TTS

Target Flow:
  Exotel (₹0.60/min) → ALB → Pipecat → Deepgram STT → Claude Haiku → Cartesia TTS
  ↑                      ↑                                ↑              ↑
  SIP trunk bridge       Webhook normalization             Prompt cache   Sonic 3
```

### Cost Breakdown (Target)

| Component       | Provider           | ₹/min  | % of total |
|-----------------|--------------------|--------|------------|
| Telephony       | Exotel             | ₹0.60  | 22%        |
| TTS             | Cartesia Sonic 3   | ₹1.19  | 43%        |
| STT             | Deepgram Nova-2    | ₹0.31  | 11%        |
| LLM             | Haiku 4.5 + cache  | ₹0.17  | 6%         |
| AWS infra       | Bootstrap EC2      | ₹0.43  | 16%        |
| **Buffer**      | Retries, overhead  | ₹1.30  | (headroom) |
| **Total**       |                    | **₹2.70** | **≤ ₹4** |

---

## Phase 1: Telephony Provider Factory (Days 1-3)

### 1.1 Create TelephonyFactory

The codebase already has `ExotelClient` and `PlivoClient` in `channels/telephony/`. What's missing is a **factory** that lets `VoiceAdapter` use them interchangeably based on config.

**New file:** `channels/telephony/factory.py`

```python
"""
Telephony Provider Factory — instantiates the right client from config.

Used by VoiceAdapter to abstract away Twilio/Exotel/Plivo differences.
The factory normalizes all provider APIs into a common interface:
  - initiate_call(to, callbacks, sip_endpoint) → {sid, status, provider}
  - end_call(call_id) → {sid, status}
  - parse_status_webhook(payload) → normalized dict
"""
from __future__ import annotations

import structlog
from typing import Any, Optional, Protocol

from voice.providers import TelephonyProvider, TelephonyConfig

logger = structlog.get_logger()


class TelephonyClient(Protocol):
    """Common interface all telephony providers must implement."""

    async def initiate_call(
        self, to: str, status_callback_url: str, sip_endpoint: str,
        ring_timeout: int = 30, record: bool = False, custom_field: str = "",
    ) -> dict[str, Any]: ...

    async def end_call(self, call_id: str) -> dict[str, Any]: ...

    @staticmethod
    def parse_status_webhook(payload: dict[str, Any]) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class TelephonyFactory:
    """Creates telephony client from TelephonyConfig."""

    @staticmethod
    def create(config: TelephonyConfig) -> TelephonyClient:
        if config.provider == TelephonyProvider.EXOTEL:
            from channels.telephony.exotel import ExotelClient
            return ExotelClient(
                account_sid=config.account_sid,
                auth_token=config.auth_token,
                caller_id=config.phone_number,
            )

        elif config.provider == TelephonyProvider.PLIVO:
            from channels.telephony.plivo_client import PlivoClient
            return PlivoClient(
                auth_id=config.account_sid,
                auth_token=config.auth_token,
                caller_id=config.phone_number,
            )

        elif config.provider == TelephonyProvider.TWILIO:
            from channels.telephony.twilio_client import TwilioClient
            return TwilioClient(
                account_sid=config.account_sid,
                auth_token=config.auth_token,
                from_number=config.phone_number,
            )

        else:
            raise ValueError(f"Unsupported telephony provider: {config.provider}")

    @staticmethod
    def get_webhook_parser(provider: TelephonyProvider):
        """Returns the static webhook parser for a provider."""
        if provider == TelephonyProvider.EXOTEL:
            from channels.telephony.exotel import ExotelClient
            return ExotelClient.parse_status_webhook
        elif provider == TelephonyProvider.PLIVO:
            from channels.telephony.plivo_client import PlivoClient
            return PlivoClient.parse_status_webhook
        elif provider == TelephonyProvider.TWILIO:
            from channels.telephony.twilio_client import TwilioClient
            return TwilioClient.parse_status_webhook
        raise ValueError(f"No webhook parser for: {provider}")
```

### 1.2 Create TwilioClient (for parity)

**New file:** `channels/telephony/twilio_client.py`

The existing codebase uses Twilio implicitly through Pipecat. We need an explicit client matching the same interface as ExotelClient/PlivoClient so the factory works uniformly.

```python
"""
Twilio Telephony Client — Global PSTN provider.

Matches the same interface as ExotelClient and PlivoClient
so TelephonyFactory can instantiate any provider transparently.
"""
from __future__ import annotations

import structlog
from typing import Any, Optional
from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class TwilioClient:
    """Twilio REST API client for voice call management."""

    BASE_URL = "https://api.twilio.com/2010-04-01/Accounts"

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.base_url = f"{self.BASE_URL}/{account_sid}"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                auth=(self.account_sid, self.auth_token),
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        client = await self._get_client()
        url = f"{self.base_url}{path}.json"
        resp = await client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            logger.error("twilio_api_error", status=resp.status_code, body=resp.text[:500])
            resp.raise_for_status()
        return resp.json()

    async def initiate_call(
        self,
        to: str,
        status_callback_url: str,
        sip_endpoint: str,
        ring_timeout: int = 30,
        record: bool = False,
        custom_field: str = "",
    ) -> dict[str, Any]:
        payload = {
            "From": self.from_number,
            "To": to,
            "Url": sip_endpoint,           # TwiML URL or SIP endpoint
            "StatusCallback": status_callback_url,
            "StatusCallbackEvent": "initiated ringing answered completed",
            "Timeout": ring_timeout,
            "Record": str(record).lower(),
            "MachineDetection": "Enable",
            "MachineDetectionTimeout": "3",
        }
        logger.info("twilio_initiate_call", to=to)
        result = await self._request("POST", "/Calls", data=payload)
        return {
            "sid": result.get("sid", ""),
            "status": result.get("status", "queued"),
            "to": to,
            "from": self.from_number,
            "provider": "twilio",
        }

    async def end_call(self, call_sid: str) -> dict[str, Any]:
        logger.info("twilio_end_call", call_sid=call_sid)
        await self._request("POST", f"/Calls/{call_sid}", data={"Status": "completed"})
        return {"sid": call_sid, "status": "completed"}

    @staticmethod
    def parse_status_webhook(payload: dict[str, Any]) -> dict[str, Any]:
        STATUS_MAP = {
            "queued": "initiating",
            "ringing": "ringing",
            "in-progress": "in-progress",
            "completed": "completed",
            "busy": "busy",
            "no-answer": "no-answer",
            "failed": "failed",
            "canceled": "cancelled",
        }
        status_raw = payload.get("CallStatus", payload.get("Status", "")).lower()
        return {
            "call_id": payload.get("CallSid", ""),
            "status": STATUS_MAP.get(status_raw, status_raw),
            "direction": payload.get("Direction", "outbound-api").split("-")[0],
            "from": payload.get("From", ""),
            "to": payload.get("To", ""),
            "duration": int(payload.get("CallDuration", payload.get("Duration", 0))),
            "recording_url": payload.get("RecordingUrl", ""),
            "answered_by": payload.get("AnsweredBy", ""),
            "timestamp": payload.get("Timestamp", datetime.now(timezone.utc).isoformat()),
            "raw": payload,
        }

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
```

### 1.3 Update `channels/telephony/__init__.py`

```python
"""
Telephony provider clients for PSTN call management.
Supports: Exotel (India), Plivo (India/Global), Twilio (Global).
"""
from channels.telephony.exotel import ExotelClient
from channels.telephony.plivo_client import PlivoClient
from channels.telephony.twilio_client import TwilioClient
from channels.telephony.factory import TelephonyFactory, TelephonyClient

__all__ = [
    "ExotelClient", "PlivoClient", "TwilioClient",
    "TelephonyFactory", "TelephonyClient",
]
```

### 1.4 Wire Factory into VoiceAdapter

**File:** `channels/voice_adapter.py`

**Change 1:** Add telephony client to `__init__` and `initialize`:

```python
# In VoiceAdapter.__init__(), add:
self._telephony_client: Optional[TelephonyClient] = None
self._telephony_provider: Optional[TelephonyProvider] = None

# In VoiceAdapter.initialize(), add after existing config loading:
from channels.telephony.factory import TelephonyFactory
from voice.providers import TelephonyProvider, TelephonyConfig

provider_name = config.get("telephony_provider", "twilio")
self._telephony_provider = TelephonyProvider(provider_name)
telephony_config = TelephonyConfig(
    provider=self._telephony_provider,
    account_sid=config.get("telephony_account_sid", ""),
    auth_token=config.get("telephony_auth_token", ""),
    phone_number=config.get("telephony_phone_number", ""),
    status_callback_url=config.get("telephony_status_callback_url", ""),
)
self._telephony_client = TelephonyFactory.create(telephony_config)
logger.info("telephony_client_initialized", provider=provider_name)
```

**Change 2:** Use client in `_initiate_outbound_call`:

Replace the current stub (`# Production: Pipecat session creation here`) with actual call initiation:

```python
async def _initiate_outbound_call(
    self, contact: Contact, initial_message: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    phone = self.get_address(contact)
    call_id = f"call_{uuid.uuid4().hex[:12]}"
    conversation_id = metadata.get("conversation_id", "")

    call_state = VoiceCallState(
        call_id=call_id,
        contact_id=contact.id,
        conversation_id=conversation_id,
        direction="outbound",
    )
    call_state.attempt_number = metadata.get("attempt_number", 1)
    call_state.add_transcript("agent", initial_message)
    call_state.tts_queue.append(initial_message)
    self._active_calls[call_id] = call_state

    # Initiate via telephony provider
    if self._telephony_client:
        try:
            sip_endpoint = self._config.get(
                "sip_endpoint",
                f"sip:{call_id}@{self._config.get('sip_domain', 'voice.example.com')}"
            )
            result = await self._telephony_client.initiate_call(
                to=phone,
                status_callback_url=self._config.get(
                    "telephony_status_callback_url",
                    f"{self._config.get('base_url', '')}/api/v1/voice/status"
                ),
                sip_endpoint=sip_endpoint,
                ring_timeout=self._ring_timeout_s,
                record=self._config.get("recording_enabled", False),
                custom_field=call_id,  # pass our call_id for webhook correlation
            )
            call_state.sip_call_id = result.get("sid", "")
            call_state.set_status(CallStatus.RINGING)
            logger.info("voice_call_initiated",
                call_id=call_id, provider_sid=result.get("sid"),
                provider=result.get("provider"), phone=phone)
        except Exception as e:
            call_state.set_status(CallStatus.FAILED)
            call_state.disposition = CallDisposition.FAILED
            logger.error("call_initiation_failed", call_id=call_id, error=str(e))
            return {"status": "failed", "error": str(e), "call_id": call_id}
    else:
        call_state.set_status(CallStatus.RINGING)

    return {
        "status": "call_initiated",
        "channel_message_id": call_id,
        "call_id": call_id,
        "to": phone,
    }
```

**Change 3:** Add provider-aware webhook parsing in `handle_call_status`:

```python
async def handle_call_status(
    self, call_id: str, status: str, metadata: dict[str, Any] = None
) -> Optional[dict[str, Any]]:
    """Handle telephony status webhook — auto-detects provider format."""
    metadata = metadata or {}

    # If raw webhook payload, normalize via provider parser
    if not status and metadata:
        from channels.telephony.factory import TelephonyFactory
        parser = TelephonyFactory.get_webhook_parser(self._telephony_provider)
        normalized = parser(metadata)
        call_id = normalized.get("call_id", call_id)
        status = normalized.get("status", "")
        metadata = {**metadata, **normalized}

    # ... rest of existing handle_call_status logic unchanged ...
```

**Change 4:** Clean up on shutdown:

```python
async def shutdown(self):
    # ... existing cleanup ...
    if self._telephony_client:
        await self._telephony_client.close()
```

---

## Phase 2: Webhook Routing (Days 2-3)

### 2.1 Provider-Specific Webhook Endpoints

**File:** `api/main.py`

Add provider-specific webhook endpoints that normalize payloads before passing to `VoiceAdapter`:

```python
# ── Exotel Webhooks ──────────────────────────────────────────
@app.post("/webhooks/exotel/status")
async def exotel_status_webhook(request: Request):
    """
    Exotel sends status callbacks here.
    Payload format differs from Twilio — we normalize before processing.
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") \
        else dict(await request.form())

    from channels.telephony.exotel import ExotelClient
    normalized = ExotelClient.parse_status_webhook(body)

    call_id = normalized["call_id"]
    status = normalized["status"]

    # Correlate with our internal call_id via custom_field
    custom_field = body.get("custom_field", body.get("CustomField", ""))
    if custom_field and custom_field.startswith("call_"):
        call_id = custom_field

    result = await voice_adapter.handle_call_status(call_id, status, normalized)
    return {"status": "ok", "result": result}


@app.post("/webhooks/exotel/answer")
async def exotel_answer_webhook(request: Request):
    """
    Exotel calls this URL when the contact answers.
    Returns SIP bridge instructions to connect to our Pipecat pipeline.
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") \
        else dict(await request.form())

    call_sid = body.get("CallSid", body.get("call_sid", ""))
    custom_field = body.get("custom_field", body.get("CustomField", ""))

    # Return Exotel applet XML to bridge to SIP
    sip_domain = voice_adapter._config.get("sip_domain", "voice.example.com")
    call_id = custom_field if custom_field.startswith("call_") else call_sid

    return Response(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Sip>sip:{call_id}@{sip_domain}</Sip>
    </Dial>
</Response>""",
        media_type="application/xml",
    )


# ── Plivo Webhooks ───────────────────────────────────────────
@app.post("/webhooks/plivo/status")
async def plivo_status_webhook(request: Request):
    """Plivo status callback — normalize and forward."""
    body = dict(await request.form())  # Plivo sends form-encoded

    from channels.telephony.plivo_client import PlivoClient
    normalized = PlivoClient.parse_status_webhook(body)

    call_id = normalized["call_id"]
    status = normalized["status"]

    result = await voice_adapter.handle_call_status(call_id, status, normalized)
    return {"status": "ok", "result": result}


@app.post("/webhooks/plivo/answer")
async def plivo_answer_webhook(request: Request):
    """Returns Plivo XML to bridge call to SIP endpoint."""
    body = dict(await request.form())
    call_uuid = body.get("CallUUID", "")
    sip_domain = voice_adapter._config.get("sip_domain", "voice.example.com")

    return Response(
        content=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <User>sip:{call_uuid}@{sip_domain}</User>
    </Dial>
</Response>""",
        media_type="application/xml",
    )
```

### 2.2 Update Existing Voice Status Endpoint

Modify the existing `/api/v1/voice/status` to auto-detect provider format:

```python
@app.post("/api/v1/voice/status")
async def voice_call_status(request: Request):
    """
    Universal call status endpoint.
    Auto-detects Twilio/Exotel/Plivo format from payload shape.
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") \
        else dict(await request.form())

    # Auto-detect provider from payload keys
    if "CallSid" in body and "CallStatus" in body:
        # Twilio format
        call_id = body.get("CallSid", "")
        status = body.get("CallStatus", "")
    elif "CallSid" in body and "Status" in body:
        # Exotel format
        from channels.telephony.exotel import ExotelClient
        normalized = ExotelClient.parse_status_webhook(body)
        call_id = body.get("custom_field", normalized["call_id"])
        status = normalized["status"]
        body = normalized
    elif "CallUUID" in body:
        # Plivo format
        from channels.telephony.plivo_client import PlivoClient
        normalized = PlivoClient.parse_status_webhook(body)
        call_id = normalized["call_id"]
        status = normalized["status"]
        body = normalized
    else:
        call_id = body.get("call_id", "")
        status = body.get("status", "")

    result = await voice_adapter.handle_call_status(call_id, status, body)
    return result or {"status": "ok"}
```

---

## Phase 3: SIP Bridge Configuration (Days 3-4)

### 3.1 Exotel SIP Trunk Setup

Exotel supports SIP trunking where their PSTN leg bridges to your SIP endpoint. The Pipecat voice service needs to expose a SIP listener.

**File:** `voice/server.py` — add SIP configuration:

```python
# In VoicePipelineManager.__init__, add SIP config:
self._sip_config = {
    "listen_host": "0.0.0.0",
    "listen_port": 5060,            # SIP signaling
    "rtp_port_range": (10000, 10100),  # RTP media
    "external_ip": "",              # Set from env: PUBLIC_IP
    "codec": "PCMU",               # μ-law 8kHz for PSTN
}
```

**docker-compose.yml update** — expose SIP ports on voice service:

```yaml
voice:
    build: .
    ports:
      - "8765:8765"     # WebSocket (existing)
      - "5060:5060/udp" # SIP signaling
      - "5060:5060/tcp" # SIP signaling (TCP fallback)
      - "10000-10100:10000-10100/udp"  # RTP media
    environment:
      - PUBLIC_IP=${PUBLIC_IP}  # EC2 public IP for SIP
      - SIP_DOMAIN=${SIP_DOMAIN:-voice.example.com}
```

### 3.2 Exotel Configuration Steps (Manual)

These are one-time setup steps in Exotel's dashboard:

1. **Create Exotel account** → get `account_sid` and `auth_token`
2. **Purchase Indian DID number** → this becomes your caller ID
3. **Configure SIP endpoint**:
   - Go to Exotel Dashboard → Settings → SIP
   - Add your server's public IP as trusted SIP endpoint
   - Set SIP URI format: `sip:{call_id}@{your_public_ip}:5060`
4. **Configure status webhooks**:
   - Status callback URL: `https://{your_domain}/webhooks/exotel/status`
   - Answer URL: `https://{your_domain}/webhooks/exotel/answer`
5. **TRAI compliance** (India regulatory):
   - Upload business registration documents
   - Submit DLT registration for outbound calling
   - Configure calling hours (9 AM - 9 PM IST for non-transactional)

### 3.3 Pipecat SIP Transport Integration

Pipecat uses `pipecat-ai[sip]` for SIP transport. The key connection point is where an incoming SIP INVITE (from Exotel) maps to a pipeline.

**File:** `voice/server.py` — add to `_run_pipeline`:

```python
async def _run_pipeline(self, pipeline: ActivePipeline):
    """
    Main pipeline loop — connects SIP audio to STT → LLM → TTS.

    When Exotel bridges a call to our SIP endpoint, Pipecat receives
    the SIP INVITE, establishes an RTP stream, and feeds audio into
    this pipeline.
    """
    try:
        pipeline.is_running = True
        call_id = pipeline.call_id
        config = pipeline.config

        # Pipecat transport handles SIP negotiation
        # The telephony provider (Exotel/Twilio) has already dialed the contact
        # and is bridging audio to our SIP endpoint.
        #
        # Audio flow:
        #   Contact → PSTN → Exotel → SIP/RTP → Pipecat → Deepgram STT
        #   Claude Haiku → Cartesia TTS → Pipecat → SIP/RTP → Exotel → PSTN → Contact

        stt_params = config.stt
        tts_params = config.tts
        llm_params = config.llm

        # ... existing pipeline loop with STT → LLM → TTS ...

    except asyncio.CancelledError:
        logger.info("pipeline_cancelled", call_id=pipeline.call_id)
    except Exception as e:
        logger.error("pipeline_error", call_id=pipeline.call_id, error=str(e))
    finally:
        pipeline.is_running = False
```

---

## Phase 4: TTS Provider Default Switch (Day 4)

### 4.1 Update Quality Presets

**File:** `voice/providers.py`

Change `ULTRA_LOW_LATENCY` and `LOW_LATENCY` presets to use Cartesia as default TTS:

```python
# ULTRA_LOW_LATENCY preset — already uses Cartesia, no change needed

# LOW_LATENCY preset — change from ElevenLabs to Cartesia:
QualityPreset.LOW_LATENCY: {
    "stt": { ... },  # unchanged
    "tts": {
        "provider": TTSProvider.CARTESIA,           # was: ELEVENLABS
        "model": "sonic-english",                    # was: eleven_turbo_v2_5
        "optimize_streaming_latency": 4,
        "output_format": AudioCodec.PCM_16K,
        "chunk_length_schedule": [50],
        "latency": LatencyProfile(
            time_to_first_byte_ms=60,               # was: 150 (Cartesia is faster)
            processing_per_second_ms=20,             # was: 30
            websocket=True,
        ),
    },
    "llm": { ... },  # see Phase 5 below
    "pipeline": { ... },  # unchanged
},
```

### 4.2 Add Cost-Optimized Preset

Add a new preset specifically for the ₹4/min stack:

```python
# Add to QUALITY_PRESETS dict:
QualityPreset.COST_OPTIMIZED: {  # ← new enum value needed
    "stt": {
        "provider": STTProvider.DEEPGRAM,
        "model": "nova-2",
        "endpointing_ms": 200,
        "vad_threshold": 0.55,
        "utterance_end_ms": 700,
        "latency": LatencyProfile(time_to_first_byte_ms=80, processing_per_second_ms=30, websocket=True),
    },
    "tts": {
        "provider": TTSProvider.CARTESIA,
        "model": "sonic-english",
        "optimize_streaming_latency": 4,
        "output_format": AudioCodec.PCM_16K,
        "chunk_length_schedule": [30],
        "latency": LatencyProfile(time_to_first_byte_ms=60, processing_per_second_ms=20, websocket=True),
    },
    "llm": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 150,
        "temperature": 0.7,
        "prompt_caching": True,          # ← new field
    },
    "pipeline": {
        "target_latency_ms": 400,
        "llm_sentence_streaming": True,
        "prefetch_greeting": True,
    },
},
```

Add the new enum value:

```python
class QualityPreset(str, Enum):
    ULTRA_LOW_LATENCY = "ultra_low_latency"
    LOW_LATENCY = "low_latency"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"
    COST_OPTIMIZED = "cost_optimized"     # ← add this
```

### 4.3 Update PipecatPipelineBuilder Default

**File:** `channels/voice_adapter.py`

In `PipecatPipelineBuilder.build()`, change defaults:

```python
"tts": {
    "provider": self._config.get("tts_provider", "cartesia"),  # was: elevenlabs
    "voice_id": self._config.get("tts_voice_id", "a0e99841-438c-4a64-b679-ae501e7d6091"),
    "model": self._config.get("tts_model", "sonic-english"),  # was: eleven_turbo_v2_5
    "output_format": "pcm_16000",
    "optimize_streaming_latency": 4,
},
```

---

## Phase 5: LLM Cost Optimization (Days 4-5)

### 5.1 Haiku for Voice, Sonnet for Text

The key insight: voice responses are 1-3 sentences. Haiku is 80% cheaper and has faster TTFB — it's actually *better* for voice because lower latency matters more than reasoning depth.

**File:** `voice/pipeline.py` — in `VoicePipelineBuilder.build()`:

```python
# LLM config — use Haiku for voice unless overridden
llm_config = preset_config.get("llm", {})
llm_model = llm_config.get("model", "claude-haiku-4-5-20251001")  # was: claude-sonnet
pipeline_config.llm = {
    "provider": "anthropic",
    "model": llm_model,
    "system_prompt": system_prompt,
    "temperature": llm_config.get("temperature", 0.7),
    "max_tokens": llm_config.get("max_tokens", 150),  # short for voice
    "stream": True,
    "extra_headers": {},  # prompt caching headers added below
}

# Enable prompt caching if configured
if llm_config.get("prompt_caching", False):
    pipeline_config.llm["cache_control"] = {
        "system_prompt": True,    # Cache the voice system prompt
        "type": "ephemeral",      # Anthropic prompt caching
    }
```

### 5.2 Prompt Caching Setup

**File:** `voice/pipeline.py` — in `VoicePromptBuilder.build()`:

The system prompt is ~500 tokens and identical across all turns in a call. With Anthropic's prompt caching, pay full price once, then 90% discount on subsequent turns.

```python
@staticmethod
def build(
    contact: Contact,
    business_context: dict[str, Any] = None,
    conversation_history: list[dict] = None,
    followup_reason: str = "",
    enable_cache: bool = True,    # ← add parameter
) -> str | dict:
    """
    Build voice-optimized system prompt.

    If enable_cache is True, returns a structured prompt with
    cache_control markers for Anthropic's prompt caching API.
    """
    prompt_text = VoicePromptBuilder._build_text(
        contact, business_context, conversation_history, followup_reason
    )

    if enable_cache:
        # Return structured format for Anthropic API with caching
        return {
            "type": "text",
            "text": prompt_text,
            "cache_control": {"type": "ephemeral"},
        }
    return prompt_text
```

### 5.3 Orchestrator-Level LLM Routing

**File:** `core/orchestrator.py`

The Orchestrator currently uses a single LLM model for all channels. Add channel-aware model selection:

```python
def _get_llm_model(self, channel: str) -> str:
    """
    Route to different LLM models based on channel.

    Voice → Haiku (faster TTFB, cheaper, short responses)
    Text channels → Sonnet (deeper reasoning, longer responses)
    """
    if channel == "voice":
        return self._config.get("voice_llm_model", "claude-haiku-4-5-20251001")
    return self._config.get("llm_model", "claude-sonnet-4-5-20250929")
```

---

## Phase 6: Configuration Updates (Day 5)

### 6.1 Settings YAML

**File:** `config/settings.example.yaml` — update voice section:

```yaml
channels:
  voice:
    enabled: true
    credentials:
      # ── Telephony (choose one) ──────────────────────────
      telephony_provider: "exotel"              # exotel | plivo | twilio
      telephony_account_sid: "${EXOTEL_SID}"
      telephony_auth_token: "${EXOTEL_TOKEN}"
      telephony_phone_number: "+919XXXXXXXXX"   # Indian DID from Exotel
      telephony_status_callback_url: "${BASE_URL}/webhooks/exotel/status"

      # ── SIP Bridge ─────────────────────────────────────
      sip_domain: "${SIP_DOMAIN}"               # your-server.example.com
      sip_endpoint: "sip:{call_id}@${SIP_DOMAIN}:5060"
      base_url: "${BASE_URL}"                   # https://your-server.example.com

      # ── STT ─────────────────────────────────────────────
      stt_provider: "deepgram"
      stt_api_key: "${DEEPGRAM_API_KEY}"
      stt_model: "nova-2"

      # ── TTS ─────────────────────────────────────────────
      tts_provider: "cartesia"                  # was: elevenlabs
      tts_api_key: "${CARTESIA_API_KEY}"
      tts_voice_id: "a0e99841-438c-4a64-b679-ae501e7d6091"
      tts_model: "sonic-english"

      # ── LLM (voice-specific) ────────────────────────────
      voice_llm_model: "claude-haiku-4-5-20251001"
      prompt_caching: true

      # ── Quality Preset ──────────────────────────────────
      quality_preset: "cost_optimized"          # new preset

      # ── Call Settings ───────────────────────────────────
      max_concurrent_calls: 20
      max_call_duration_s: 300
      silence_timeout_ms: 15000
      ring_timeout_s: 30
      recording_enabled: false
```

### 6.2 Environment Variables

**File:** `.env.example`

```bash
# ── Exotel (Indian Telephony) ──────────────────────────
EXOTEL_SID=your_exotel_account_sid
EXOTEL_TOKEN=your_exotel_auth_token
EXOTEL_PHONE=+919XXXXXXXXX

# ── Deepgram (STT) ────────────────────────────────────
DEEPGRAM_API_KEY=your_deepgram_api_key

# ── Cartesia (TTS) ────────────────────────────────────
CARTESIA_API_KEY=your_cartesia_api_key

# ── Anthropic (LLM) ───────────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_api_key

# ── Infrastructure ─────────────────────────────────────
BASE_URL=https://voice.yourdomain.com
SIP_DOMAIN=voice.yourdomain.com
PUBLIC_IP=x.x.x.x                  # EC2 Elastic IP

# ── Database ───────────────────────────────────────────
POSTGRES_PASSWORD=your_secure_password
```

### 6.3 Docker Compose (Production Bootstrap)

**File:** `docker-compose.prod.yml`

```yaml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      - CONVERSE_CONFIG=/app/config/settings.yaml
      - PYTHONPATH=/app
    volumes:
      - ./config:/app/config
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

  worker:
    build: .
    env_file: .env
    environment:
      - CONVERSE_CONFIG=/app/config/settings.yaml
      - PYTHONPATH=/app
    volumes:
      - ./config:/app/config
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
    command: python -m job_queue.consumer

  voice:
    build: .
    ports:
      - "8765:8765"         # WebSocket
      - "5060:5060/udp"     # SIP signaling
      - "5060:5060/tcp"     # SIP signaling (TCP)
      - "10000-10100:10000-10100/udp"  # RTP media
    env_file: .env
    environment:
      - CONVERSE_CONFIG=/app/config/settings.yaml
      - PYTHONPATH=/app
      - PUBLIC_IP=${PUBLIC_IP}
      - SIP_DOMAIN=${SIP_DOMAIN}
    volumes:
      - ./config:/app/config
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
    command: python -m voice.server

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: converse_agent
      POSTGRES_USER: converse
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U converse -d converse_agent"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G

volumes:
  redis_data:
  pg_data:
```

---

## Phase 7: Testing (Days 5-7)

### 7.1 Unit Tests for Telephony Factory

**File:** `tests/test_telephony_factory.py`

```python
"""Tests for TelephonyFactory and provider-specific webhook normalization."""
import pytest
from channels.telephony.factory import TelephonyFactory
from channels.telephony.exotel import ExotelClient
from channels.telephony.plivo_client import PlivoClient
from channels.telephony.twilio_client import TwilioClient
from voice.providers import TelephonyProvider, TelephonyConfig


class TestTelephonyFactory:
    """Test factory creates correct client types."""

    def test_create_exotel(self):
        config = TelephonyConfig(
            provider=TelephonyProvider.EXOTEL,
            account_sid="test_sid",
            auth_token="test_token",
            phone_number="+919876543210",
        )
        client = TelephonyFactory.create(config)
        assert isinstance(client, ExotelClient)

    def test_create_plivo(self):
        config = TelephonyConfig(
            provider=TelephonyProvider.PLIVO,
            account_sid="test_id",
            auth_token="test_token",
            phone_number="+919876543210",
        )
        client = TelephonyFactory.create(config)
        assert isinstance(client, PlivoClient)

    def test_create_twilio(self):
        config = TelephonyConfig(
            provider=TelephonyProvider.TWILIO,
            account_sid="AC_test",
            auth_token="test_token",
            phone_number="+14155551234",
        )
        client = TelephonyFactory.create(config)
        assert isinstance(client, TwilioClient)

    def test_create_unsupported_raises(self):
        config = TelephonyConfig(provider=TelephonyProvider.DAILY)
        with pytest.raises(ValueError, match="Unsupported"):
            TelephonyFactory.create(config)


class TestExotelWebhookParsing:
    """Test Exotel webhook normalization."""

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

    def test_ringing_status(self):
        payload = {"CallSid": "exo_123", "Status": "ringing", "Direction": "outbound"}
        result = ExotelClient.parse_status_webhook(payload)
        assert result["status"] == "ringing"

    def test_busy_status(self):
        payload = {"CallSid": "exo_123", "Status": "busy"}
        result = ExotelClient.parse_status_webhook(payload)
        assert result["status"] == "busy"

    def test_lowercase_keys(self):
        """Exotel sometimes uses lowercase keys."""
        payload = {"call_sid": "exo_123", "status": "in-progress"}
        result = ExotelClient.parse_status_webhook(payload)
        assert result["status"] == "in-progress"


class TestPlivoWebhookParsing:
    """Test Plivo webhook normalization."""

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


class TestTwilioWebhookParsing:
    """Test Twilio webhook normalization."""

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


class TestExotelNumberNormalization:
    """Test Indian phone number normalization."""

    def test_10_digit(self):
        assert ExotelClient._normalize_indian_number("9876543210") == "+919876543210"

    def test_with_plus91(self):
        assert ExotelClient._normalize_indian_number("+919876543210") == "+919876543210"

    def test_with_91_no_plus(self):
        assert ExotelClient._normalize_indian_number("919876543210") == "+919876543210"

    def test_with_spaces(self):
        assert ExotelClient._normalize_indian_number("98765 43210") == "+919876543210"

    def test_non_indian(self):
        assert ExotelClient._normalize_indian_number("+14155551234") == "+14155551234"
```

### 7.2 Integration Test for Provider Switching

**File:** `tests/test_voice_provider_switch.py`

```python
"""Integration tests: switch telephony provider without breaking call flow."""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from channels.voice_adapter import VoiceAdapter, CallStatus, CallDisposition
from voice.providers import TelephonyProvider


@pytest.fixture
def voice_adapter():
    adapter = VoiceAdapter()
    return adapter


class TestProviderSwitch:
    """Verify call flow works identically across telephony providers."""

    @pytest.mark.parametrize("provider", ["exotel", "plivo", "twilio"])
    async def test_initialize_with_provider(self, voice_adapter, provider):
        config = {
            "telephony_provider": provider,
            "telephony_account_sid": "test_sid",
            "telephony_auth_token": "test_token",
            "telephony_phone_number": "+919876543210",
            "stt_provider": "deepgram",
            "tts_provider": "cartesia",
        }
        await voice_adapter.initialize(config)
        assert voice_adapter._telephony_provider == TelephonyProvider(provider)
        assert voice_adapter._telephony_client is not None

    async def test_exotel_webhook_maps_to_call_status(self, voice_adapter):
        """Exotel webhook payloads correctly update VoiceCallState."""
        config = {
            "telephony_provider": "exotel",
            "telephony_account_sid": "test",
            "telephony_auth_token": "test",
            "telephony_phone_number": "+919876543210",
        }
        await voice_adapter.initialize(config)

        # Simulate an existing call
        from models.schemas import Contact
        contact = Contact(id="c1", name="Test", phone="+919123456789")
        result = await voice_adapter._initiate_outbound_call(
            contact, "Hello!", {"conversation_id": "conv1"}
        )
        call_id = result["call_id"]

        # Simulate Exotel "in-progress" webhook
        status_result = await voice_adapter.handle_call_status(
            call_id, "in-progress", {}
        )
        call_state = voice_adapter.get_active_call(call_id)
        assert call_state.status == CallStatus.CONNECTED

    async def test_cost_optimized_preset_uses_correct_providers(self):
        from voice.providers import VoiceProviderRegistry, QualityPreset
        registry = VoiceProviderRegistry()
        registry.load_preset(QualityPreset.COST_OPTIMIZED)

        stt = registry.get_stt()
        tts = registry.get_tts()
        assert stt.provider.value == "deepgram"
        assert tts.provider.value == "cartesia"
```

### 7.3 Run All Tests

```bash
cd /home/claude/converse-agent
python -m pytest tests/ -v --tb=short

# Specifically test telephony:
python -m pytest tests/test_telephony_factory.py -v
python -m pytest tests/test_voice_provider_switch.py -v

# Existing voice tests should still pass:
python -m pytest tests/test_voice_subsystem.py -v
```

---

## Phase 8: AWS Deployment (Days 6-7)

### 8.1 Bootstrap Stack (₹4/min target)

Single EC2 instance running Docker Compose:

```bash
# 1. Launch EC2
#    Instance: t4g.large (4 vCPU ARM, 8GB) — $47/mo reserved
#    AMI: Ubuntu 24.04 ARM
#    Region: ap-south-1 (Mumbai)
#    Security Groups:
#      - 8000/tcp (API from ALB only)
#      - 8765/tcp (WebSocket from ALB only)
#      - 5060/udp+tcp (SIP from Exotel IPs only)
#      - 10000-10100/udp (RTP from Exotel IPs only)
#      - 22/tcp (SSH from your IP only)
#    Elastic IP: allocate and associate

# 2. Install Docker
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu

# 3. Clone and configure
git clone your-repo /opt/converse-agent
cd /opt/converse-agent
cp .env.example .env
# Edit .env with real credentials

# 4. Launch
docker compose -f docker-compose.prod.yml up -d

# 5. Set up ALB (via AWS CLI or console)
#    - HTTPS listener on 443 → target group on 8000
#    - WebSocket support enabled
#    - Health check: GET /health → 200
#    - ACM certificate for your domain
```

### 8.2 Security Groups for Exotel

Exotel SIP traffic comes from specific IP ranges. Restrict inbound SIP/RTP to only Exotel's IP ranges:

```bash
# Exotel SIP IPs (verify current list with Exotel support):
# 54.254.73.0/24
# 13.235.X.X/24 (Mumbai region)

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol udp \
  --port 5060 \
  --cidr 54.254.73.0/24

aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol udp \
  --port 10000-10100 \
  --cidr 54.254.73.0/24
```

### 8.3 DNS & TLS

```bash
# Route 53 A record
voice.yourdomain.com → ALB DNS name (ALIAS record)

# For SIP, point directly to EC2 Elastic IP (SIP doesn't go through ALB)
sip.yourdomain.com → EC2 Elastic IP (A record)
```

---

## Implementation Checklist

### Files to Create (New)

| # | File | Purpose |
|---|------|---------|
| 1 | `channels/telephony/factory.py` | TelephonyFactory + TelephonyClient protocol |
| 2 | `channels/telephony/twilio_client.py` | Twilio client matching Exotel/Plivo interface |
| 3 | `docker-compose.prod.yml` | Production Docker Compose with resource limits |
| 4 | `.env.example` | Environment variable template |
| 5 | `tests/test_telephony_factory.py` | Unit tests for factory + webhook parsing |
| 6 | `tests/test_voice_provider_switch.py` | Integration tests for provider switching |

### Files to Modify (Existing)

| # | File | Changes |
|---|------|---------|
| 1 | `channels/telephony/__init__.py` | Add TwilioClient, TelephonyFactory exports |
| 2 | `channels/voice_adapter.py` | Wire TelephonyFactory into init + outbound calls + webhook parsing |
| 3 | `voice/providers.py` | Add COST_OPTIMIZED preset + enum, update LOW_LATENCY TTS default |
| 4 | `voice/pipeline.py` | Add prompt caching support in VoicePromptBuilder |
| 5 | `voice/server.py` | Add SIP config, update pipeline runner |
| 6 | `api/main.py` | Add Exotel/Plivo webhook endpoints, update universal status endpoint |
| 7 | `config/settings.example.yaml` | Update voice section with Exotel/Cartesia/Haiku defaults |
| 8 | `docker-compose.yml` | Add SIP ports to voice service |
| 9 | `core/orchestrator.py` | Add channel-aware LLM model routing |

### External Setup (Manual)

| # | Task | Provider | Time |
|---|------|----------|------|
| 1 | Create Exotel account + KYC | Exotel | 1-2 days |
| 2 | Purchase Indian DID number | Exotel | Same day |
| 3 | Configure SIP endpoint in Exotel dashboard | Exotel | 1 hour |
| 4 | DLT registration for outbound calling | TRAI/Exotel | 3-5 days |
| 5 | Create Cartesia account + get API key | Cartesia | Same day |
| 6 | Create Deepgram Growth account | Deepgram | Same day |
| 7 | Enable prompt caching on Anthropic API key | Anthropic | Same day |
| 8 | Launch EC2 + Elastic IP in ap-south-1 | AWS | 1 hour |
| 9 | Set up ALB + ACM certificate | AWS | 1-2 hours |
| 10 | Configure Route 53 DNS | AWS | 30 min |

---

## Execution Timeline

```
Day 1:  Phase 1.1-1.3  — TelephonyFactory, TwilioClient, __init__ update
Day 2:  Phase 1.4      — Wire factory into VoiceAdapter
        Phase 2        — Webhook routing in api/main.py
Day 3:  Phase 3        — SIP bridge config, Docker Compose SIP ports
Day 4:  Phase 4        — Cartesia default in presets, COST_OPTIMIZED preset
        Phase 5.1-5.2  — Haiku for voice, prompt caching
Day 5:  Phase 5.3      — Orchestrator LLM routing
        Phase 6        — Config files, env vars
Day 6:  Phase 7        — Tests (unit + integration)
Day 7:  Phase 8        — AWS deployment, Exotel SIP testing

Parallel: External setup (Exotel account, DLT, Cartesia account) — start Day 1
```

### Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Exotel SIP connectivity issues | Keep Twilio as fallback, factory makes switching one config change |
| DLT registration delay (can take 5+ days) | Start registration on Day 1, test with manual calls |
| Cartesia voice quality not matching ElevenLabs | A/B test with VoiceProviderRegistry (register both, switch per-call) |
| Prompt caching not reducing costs as expected | Monitor Anthropic API usage dashboard, caching benefit compounds with call volume |
| SIP NAT traversal on EC2 | Use Elastic IP, set `external_ip` in SIP config, verify with `sngrep` |

---

## Validation Criteria

### Cost Target Met

```
Per-minute cost calculation after deployment:
  Exotel:   ₹0.60   (verify from Exotel billing dashboard)
  Cartesia: ₹1.19   (verify from Cartesia usage API)
  Deepgram: ₹0.31   (verify from Deepgram console)
  Haiku:    ₹0.17   (verify from Anthropic usage dashboard, confirm caching active)
  AWS:      ₹0.43   (EC2 $47/mo ÷ estimated minutes)
  ─────────────────
  Total:    ₹2.70   (target: ≤ ₹4.00 ✓)
```

### Quality Not Degraded

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| End-to-end latency (p50) | ~500ms | ≤ 500ms | CallLatencyTracker reports |
| End-to-end latency (p90) | ~800ms | ≤ 700ms | Cartesia TTFB is lower than ElevenLabs |
| TTS naturalness | ElevenLabs | Comparable | Manual A/B listening test, 20 calls |
| LLM response quality | Sonnet | Adequate for voice | Review transcripts of 50 calls |
| Call completion rate | Baseline | Same ± 2% | Monitor disposition=completed / total |

### All Tests Pass

```bash
python -m pytest tests/ -v
# Expected: 359 existing + ~30 new = ~389 tests passing
```
