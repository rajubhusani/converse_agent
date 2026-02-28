"""
Telephony Provider Factory — instantiates the right client from config.

Used by VoiceAdapter to abstract away Twilio/Exotel/Plivo differences.
The factory normalizes all provider APIs into a common interface:
  - initiate_call(to, callbacks, sip_endpoint) → {sid, status, provider}
  - end_call(call_id) → {sid, status}
  - parse_status_webhook(payload) → normalized dict
  - close() → clean up HTTP clients

Twilio is the default provider. Switch to Exotel/Plivo for Indian PSTN
savings (~60% cheaper for India-to-India calls).
"""
from __future__ import annotations

import structlog
from typing import Any, Optional, Protocol, runtime_checkable

from voice.providers import TelephonyProvider, TelephonyConfig

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  PROTOCOL — Common interface all providers implement
# ══════════════════════════════════════════════════════════════

@runtime_checkable
class TelephonyClient(Protocol):
    """
    Common interface for all telephony providers.

    Each provider client (TwilioClient, ExotelClient, PlivoClient)
    implements these methods with provider-specific API calls, but
    returns normalized dicts so the caller doesn't need to know
    which provider is active.
    """

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
        Place an outbound call.

        Returns:
            {"sid": "...", "status": "queued", "to": "...", "from": "...", "provider": "..."}
        """
        ...

    async def end_call(self, call_id: str) -> dict[str, Any]:
        """
        Terminate an active call.

        Returns:
            {"sid": "...", "status": "completed"}
        """
        ...

    @staticmethod
    def parse_status_webhook(payload: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize provider-specific webhook payload into standard format.

        Returns:
            {
                "call_id": str,
                "status": str,    # initiating|ringing|in-progress|completed|busy|no-answer|failed|cancelled
                "direction": str, # inbound|outbound
                "from": str,
                "to": str,
                "duration": int,
                "recording_url": str,
                "answered_by": str,
                "timestamp": str,
                "raw": dict,
            }
        """
        ...

    async def close(self) -> None:
        """Clean up HTTP clients and connections."""
        ...


# ══════════════════════════════════════════════════════════════
#  FACTORY
# ══════════════════════════════════════════════════════════════

class TelephonyFactory:
    """
    Creates telephony client from TelephonyConfig.

    Usage:
        config = TelephonyConfig(provider=TelephonyProvider.TWILIO, ...)
        client = TelephonyFactory.create(config)
        result = await client.initiate_call(...)
    """

    # Supported providers and their lazy-loaded classes
    _PROVIDERS = {
        TelephonyProvider.TWILIO: (
            "channels.telephony.twilio_client", "TwilioClient"
        ),
        TelephonyProvider.EXOTEL: (
            "channels.telephony.exotel", "ExotelClient"
        ),
        TelephonyProvider.PLIVO: (
            "channels.telephony.plivo_client", "PlivoClient"
        ),
    }

    @staticmethod
    def create(config: TelephonyConfig) -> TelephonyClient:
        """
        Create a telephony client from config.

        Args:
            config: TelephonyConfig with provider, credentials, and settings.

        Returns:
            Provider client implementing TelephonyClient protocol.

        Raises:
            ValueError: If provider is not supported (e.g., Daily is WebRTC only).
        """
        if config.provider == TelephonyProvider.TWILIO:
            from channels.telephony.twilio_client import TwilioClient
            client = TwilioClient(
                account_sid=config.account_sid,
                auth_token=config.auth_token,
                from_number=config.phone_number,
            )
            logger.info("telephony_client_created", provider="twilio")
            return client

        elif config.provider == TelephonyProvider.EXOTEL:
            from channels.telephony.exotel import ExotelClient
            client = ExotelClient(
                account_sid=config.account_sid,
                api_key=config.api_key,
                auth_token=config.auth_token,
                caller_id=config.phone_number,
                subdomain=config.subdomain or "api.exotel.com",
            )
            logger.info("telephony_client_created", provider="exotel",
                         account_sid=config.account_sid)
            return client

        elif config.provider == TelephonyProvider.PLIVO:
            from channels.telephony.plivo_client import PlivoClient
            client = PlivoClient(
                auth_id=config.account_sid,
                auth_token=config.auth_token,
                caller_id=config.phone_number,
            )
            logger.info("telephony_client_created", provider="plivo")
            return client

        else:
            raise ValueError(
                f"Unsupported telephony provider: {config.provider}. "
                f"Supported: {', '.join(p.value for p in TelephonyFactory._PROVIDERS)}"
            )

    @staticmethod
    def get_webhook_parser(provider: TelephonyProvider):
        """
        Returns the static webhook parser for a provider.

        Usage:
            parser = TelephonyFactory.get_webhook_parser(TelephonyProvider.TWILIO)
            normalized = parser(raw_webhook_payload)
        """
        if provider == TelephonyProvider.TWILIO:
            from channels.telephony.twilio_client import TwilioClient
            return TwilioClient.parse_status_webhook

        elif provider == TelephonyProvider.EXOTEL:
            from channels.telephony.exotel import ExotelClient
            return ExotelClient.parse_status_webhook

        elif provider == TelephonyProvider.PLIVO:
            from channels.telephony.plivo_client import PlivoClient
            return PlivoClient.parse_status_webhook

        raise ValueError(f"No webhook parser for: {provider}")

    @staticmethod
    def detect_provider_from_webhook(payload: dict[str, Any]) -> Optional[TelephonyProvider]:
        """
        Auto-detect which provider sent a webhook based on payload shape.

        Twilio:  has CallSid + CallStatus
        Exotel:  has CallSid + Status (no "Call" prefix on Status)
        Plivo:   has CallUUID

        Returns None if format is unrecognized.
        """
        if "CallUUID" in payload:
            return TelephonyProvider.PLIVO

        if "CallSid" in payload or "call_sid" in payload:
            # Twilio uses "CallStatus", Exotel uses "Status"
            if "CallStatus" in payload:
                return TelephonyProvider.TWILIO
            return TelephonyProvider.EXOTEL

        return None
