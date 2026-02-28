"""
Twilio Telephony Client — Default global PSTN provider.

Provides the same interface as ExotelClient and PlivoClient so
TelephonyFactory can instantiate any provider transparently.

Call flow:
1. initiate_call() → Twilio dials contact via PSTN
2. On answer, Twilio executes TwiML at answer_url (SIP bridge to Pipecat)
3. Status webhooks arrive at status_callback_url
4. end_call() terminates active call

API Docs: https://www.twilio.com/docs/voice/api
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
            logger.error(
                "twilio_api_error",
                status=resp.status_code,
                body=resp.text[:500],
                path=path,
            )
            resp.raise_for_status()
        return resp.json()

    # ── Call Management ─────────────────────────────────────

    async def initiate_call(
        self,
        to: str,
        status_callback_url: str,
        sip_endpoint: str,
        ring_timeout: int = 30,
        record: bool = False,
        custom_field: str = "",
    ) -> dict[str, Any]:
        """
        Place an outbound call via Twilio.

        Args:
            to: Destination phone number (E.164)
            status_callback_url: Webhook URL for call status events
            sip_endpoint: TwiML URL that returns <Dial><Sip> for Pipecat bridge
            ring_timeout: Seconds to wait for answer
            record: Whether to record the call
            custom_field: Passed as SipHeader for webhook correlation
        """
        # Twilio uses form-encoded POST, not JSON
        payload = {
            "From": self.from_number,
            "To": to,
            "Url": sip_endpoint,
            "StatusCallback": status_callback_url,
            "StatusCallbackEvent": "initiated ringing answered completed",
            "StatusCallbackMethod": "POST",
            "Timeout": str(ring_timeout),
            "Record": "true" if record else "false",
            "MachineDetection": "Enable",
            "MachineDetectionTimeout": "3",
            "AsyncAmd": "true",
        }

        if custom_field:
            # Twilio passes SipHeaders through to the TwiML app
            payload["SipHeaders"] = f"X-Custom-CallId={custom_field}"

        logger.info("twilio_initiate_call", to=to, sip_endpoint=sip_endpoint)
        result = await self._request("POST", "/Calls", data=payload)

        return {
            "sid": result.get("sid", ""),
            "status": result.get("status", "queued"),
            "to": to,
            "from": self.from_number,
            "provider": "twilio",
        }

    async def end_call(self, call_sid: str) -> dict[str, Any]:
        """Terminate an active call."""
        logger.info("twilio_end_call", call_sid=call_sid)
        result = await self._request(
            "POST",
            f"/Calls/{call_sid}",
            data={"Status": "completed"},
        )
        return {"sid": call_sid, "status": "completed"}

    async def get_call_details(self, call_sid: str) -> dict[str, Any]:
        """Fetch current call state from Twilio."""
        result = await self._request("GET", f"/Calls/{call_sid}")
        return result

    # ── Webhook Parsing ─────────────────────────────────────

    @staticmethod
    def parse_status_webhook(payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize Twilio status webhook into our internal format.

        Twilio sends:
          - CallSid, CallStatus, Direction, From, To, Duration,
            RecordingUrl, AnsweredBy, etc.
        """
        STATUS_MAP = {
            "queued": "initiating",
            "initiated": "initiating",
            "ringing": "ringing",
            "in-progress": "in-progress",
            "completed": "completed",
            "busy": "busy",
            "no-answer": "no-answer",
            "failed": "failed",
            "canceled": "cancelled",
        }

        status_raw = payload.get(
            "CallStatus", payload.get("Status", "")
        ).lower()

        # Twilio direction format: "outbound-api", "inbound", "outbound-dial"
        direction = payload.get("Direction", "outbound-api")
        if "-" in direction:
            direction = direction.split("-")[0]

        return {
            "call_id": payload.get("CallSid", ""),
            "status": STATUS_MAP.get(status_raw, status_raw),
            "direction": direction.lower(),
            "from": payload.get("From", ""),
            "to": payload.get("To", ""),
            "duration": int(
                payload.get("CallDuration", payload.get("Duration", 0))
            ),
            "recording_url": payload.get("RecordingUrl", ""),
            "answered_by": payload.get("AnsweredBy", ""),
            "timestamp": payload.get(
                "Timestamp", datetime.now(timezone.utc).isoformat()
            ),
            "raw": payload,
        }

    # ── Helpers ─────────────────────────────────────────────

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
