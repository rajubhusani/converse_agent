"""
Exotel Telephony Client — Indian PSTN provider.

Handles outbound/inbound call management via Exotel's REST API (V1).
SIP trunking connects Exotel's PSTN leg to our Pipecat SIP endpoint.

Pricing: ~₹0.50-0.70/min outbound to Indian mobile (vs Twilio ~₹1.20/min).

API Docs: https://developer.exotel.com/api/
"""
from __future__ import annotations

import structlog
from typing import Any, Optional
from datetime import datetime, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class ExotelClient:
    """
    Exotel REST API client for voice call management (V1 API).

    Call flow:
    1. initiate_call() → Exotel dials the From number first
    2. On answer, Exotel connects to the To number
    3. Status webhooks arrive at our /webhooks/exotel/status endpoint
    4. On completion, end_call() or natural hangup

    Auth: Basic Auth with api_key:api_token.
          The account_sid is used in the URL path.
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        caller_id: str,
        api_key: str = "",
        subdomain: str = "api.exotel.com",
    ):
        self.account_sid = account_sid
        self.api_key = api_key or account_sid
        self.auth_token = auth_token
        self.caller_id = caller_id
        self.subdomain = subdomain
        self.base_url = f"https://{subdomain}/v1/Accounts/{account_sid}"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                auth=(self.api_key, self.auth_token),
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        resp = await client.request(method, url, **kwargs)

        if resp.status_code >= 400:
            logger.error("exotel_api_error",
                         status=resp.status_code,
                         body=resp.text[:500],
                         url=url,
                         path=path)
            resp.raise_for_status()

        # V1 API can return XML or JSON; request JSON via .json suffix
        try:
            return resp.json()
        except Exception:
            return {"raw_response": resp.text}

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
        Place an outbound call via Exotel V1 API.

        Exotel V1 connect API calls the From number first, then bridges
        to the To number once From picks up.

        Args:
            to: Destination phone number (E.164 or 10-digit Indian)
            status_callback_url: Webhook URL for call status events
            sip_endpoint: URL or SIP URI for call flow / answer
            ring_timeout: Seconds to wait for answer
            record: Whether to record the call
            custom_field: Arbitrary string passed through to callbacks

        Returns:
            {"sid": "...", "status": "queued", "to": "...", "from": "..."}
        """
        to = self._normalize_indian_number(to)

        # V1 API uses form-encoded POST to /Calls/connect.json
        form_data = {
            "From": to,
            "To": self.caller_id,
            "CallerId": self.caller_id,
            "TimeLimit": "300",
            "TimeOut": str(ring_timeout),
        }

        if status_callback_url:
            form_data["StatusCallback"] = status_callback_url
            form_data["StatusCallbackEvents[0]"] = "terminal"

        if record:
            form_data["Record"] = "true"

        if custom_field:
            form_data["CustomField"] = custom_field

        logger.info("exotel_initiate_call",
                     to=to,
                     caller_id=self.caller_id,
                     url=f"{self.base_url}/Calls/connect.json")

        result = await self._request(
            "POST",
            "/Calls/connect.json",
            data=form_data,
        )

        # V1 response: {"Call": {"Sid": "...", "Status": "queued", ...}}
        call_data = result.get("Call", result.get("call", result))
        sid = call_data.get("Sid", call_data.get("sid", ""))
        status = call_data.get("Status", call_data.get("status", "queued"))

        logger.info("exotel_call_initiated",
                     sid=sid,
                     status=status,
                     to=to)

        return {
            "sid": sid,
            "status": status,
            "to": to,
            "from": self.caller_id,
            "provider": "exotel",
        }

    async def end_call(self, call_sid: str) -> dict[str, Any]:
        """Terminate an active call."""
        logger.info("exotel_end_call", call_sid=call_sid)
        result = await self._request(
            "POST",
            f"/Calls/{call_sid}.json",
            data={"Status": "completed"},
        )
        return {"sid": call_sid, "status": "completed"}

    async def get_call_details(self, call_sid: str) -> dict[str, Any]:
        """Fetch current call state from Exotel."""
        result = await self._request("GET", f"/Calls/{call_sid}.json")
        return result.get("Call", result.get("call", result))

    # ── Webhook Parsing ─────────────────────────────────────

    @staticmethod
    def parse_status_webhook(payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize Exotel status webhook into our internal format.

        Exotel sends:
          - CallSid, Status, Direction, From, To, DateUpdated, RecordingUrl, etc.

        We normalize to match VoiceAdapter.handle_call_status() expectations.
        """
        call_sid = payload.get("CallSid", payload.get("call_sid", payload.get("sid", "")))
        status_raw = payload.get("Status", payload.get("status", "")).lower()

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

        return {
            "call_id": call_sid,
            "status": STATUS_MAP.get(status_raw, status_raw),
            "direction": payload.get("Direction", payload.get("direction", "outbound")).lower(),
            "from": payload.get("From", payload.get("from", "")),
            "to": payload.get("To", payload.get("to", "")),
            "duration": int(payload.get("Duration", payload.get("duration", 0))),
            "recording_url": payload.get("RecordingUrl", payload.get("recording_url", "")),
            "answered_by": payload.get("AnsweredBy", payload.get("answered_by", "")),
            "timestamp": payload.get("DateUpdated", datetime.now(timezone.utc).isoformat()),
            "raw": payload,
        }

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize_indian_number(number: str) -> str:
        """Normalize to Indian format that Exotel expects."""
        number = number.strip().replace(" ", "").replace("-", "")

        # Already E.164 Indian
        if number.startswith("+91") and len(number) == 13:
            return number

        # 10-digit Indian
        if len(number) == 10 and number[0] in "6789":
            return f"+91{number}"

        # 91XXXXXXXXXX (without +)
        if number.startswith("91") and len(number) == 12:
            return f"+{number}"

        return number

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
