"""
Telephony provider clients for PSTN call management.

Supports: Twilio (Global, default), Exotel (India), Plivo (India/Global).
Each client provides: initiate_call, end_call, parse_status_webhook, close.

Usage:
    from channels.telephony import TelephonyFactory
    client = TelephonyFactory.create(config)
    result = await client.initiate_call(to="+919876543210", ...)
"""
from channels.telephony.twilio_client import TwilioClient
from channels.telephony.exotel import ExotelClient
from channels.telephony.plivo_client import PlivoClient
from channels.telephony.factory import TelephonyFactory, TelephonyClient

__all__ = [
    "TwilioClient", "ExotelClient", "PlivoClient",
    "TelephonyFactory", "TelephonyClient",
]
