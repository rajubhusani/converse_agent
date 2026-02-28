"""Channel adapters for all supported communication channels."""
from channels.base import (
    ChannelAdapter,
    ChannelRegistry,
    ChannelError,
    DeliveryStatus,
    DeliveryRecord,
    TokenBucketRateLimiter,
    CircuitBreaker,
    ChannelMetrics,
)
from channels.chat_adapter import ChatAdapter
from channels.email_adapter import EmailAdapter
from channels.whatsapp_adapter import WhatsAppAdapter
from channels.voice_adapter import VoiceAdapter, VoiceCallState, CallStatus, CallDisposition
from channels.sms_adapter import SMSAdapter

__all__ = [
    "ChannelAdapter", "ChannelRegistry", "ChannelError",
    "DeliveryStatus", "DeliveryRecord",
    "TokenBucketRateLimiter", "CircuitBreaker", "ChannelMetrics",
    "ChatAdapter", "EmailAdapter", "WhatsAppAdapter", "VoiceAdapter", "SMSAdapter",
    "VoiceCallState", "CallStatus", "CallDisposition",
]
