"""
Plivo Telephony Client — Indian/Global PSTN provider.

Near drop-in replacement for Twilio with better India pricing (~₹0.60-0.80/min).
API is very similar to Twilio's, making migration straightforward.

API Docs: https://www.plivo.com/docs/voice/
"""
from __future__ import annotations

import structlog
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class PlivoClient:
    """Plivo REST API client for voice call management."""

    BASE_URL = "https://api.plivo.com/v1/Account"

    def __init__(self, auth_id: str, auth_token: str, caller_id: str):
        self.auth_id = auth_id
        self.auth_token = auth_token
        self.caller_id = caller_id
        self.base_url = f"{self.BASE_URL}/{auth_id}"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                auth=(self.auth_id, self.auth_token),
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        client = await self._get_client()
        url = f"{self.base_url}{path}/"
        resp = await client.request(method, url, **kwargs)
        if resp.status_code >= 400:
            logger.error("plivo_api_error", status=resp.status_code, body=resp.text[:500])
            resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def initiate_call(
        self,
        to: str,
        status_callback_url: str,
        answer_url: str,
        ring_timeout: int = 30,
        record: bool = False,
        custom_field: str = "",
    ) -> dict[str, Any]:
        """
        Place an outbound call via Plivo.

        Args:
            to: Destination phone (E.164)
            status_callback_url: Webhook for status events
            answer_url: URL returning Plivo XML for call handling (SIP bridge)
            ring_timeout: Seconds to wait
            record: Record the call
        """
        payload = {
            "from": self.caller_id,
            "to": to,
            "answer_url": answer_url,
            "answer_method": "POST",
            "hangup_url": status_callback_url,
            "hangup_method": "POST",
            "ring_timeout": ring_timeout,
            "time_limit": 300,
            "record": record,
            "machine_detection": "true",
            "machine_detection_time": 3000,
        }

        logger.info("plivo_initiate_call", to=to)
        result = await self._request("POST", "/Call", json=payload)

        return {
            "sid": result.get("request_uuid", ""),
            "status": "queued",
            "to": to,
            "from": self.caller_id,
            "provider": "plivo",
        }

    async def end_call(self, call_uuid: str) -> dict[str, Any]:
        logger.info("plivo_end_call", call_uuid=call_uuid)
        await self._request("DELETE", f"/Call/{call_uuid}")
        return {"sid": call_uuid, "status": "completed"}

    @staticmethod
    def parse_status_webhook(payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize Plivo webhook to our internal format."""
        STATUS_MAP = {
            "ring": "ringing",
            "answer": "in-progress",
            "hangup": "completed",
            "busy": "busy",
            "timeout": "no-answer",
            "reject": "failed",
            "cancel": "cancelled",
        }
        event = payload.get("Event", payload.get("event", "")).lower()
        return {
            "call_id": payload.get("CallUUID", payload.get("call_uuid", "")),
            "status": STATUS_MAP.get(event, event),
            "direction": payload.get("Direction", "outbound").lower(),
            "from": payload.get("From", ""),
            "to": payload.get("To", ""),
            "duration": int(payload.get("Duration", 0)),
            "recording_url": payload.get("RecordUrl", ""),
            "answered_by": payload.get("Machine", ""),
            "raw": payload,
        }

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
