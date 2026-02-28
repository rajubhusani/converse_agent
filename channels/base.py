"""
Channel Adapters — Production-grade base infrastructure for all channels.

Provides:
- ChannelError: structured error hierarchy
- TokenBucketRateLimiter: async token bucket with configurable burst
- CircuitBreaker: failure-counting breaker with half-open probe
- ChannelMetrics: per-channel send/fail/delivery/latency tracking
- DeliveryStatus / DeliveryRecord: full message lifecycle with history
- ChannelAdapter: abstract base wrapping every send with resilience
- ChannelRegistry: adapter lookup, health checks, channel selection
"""
from __future__ import annotations

import abc
import asyncio
import hashlib
import time
import uuid
import structlog
from enum import Enum
from typing import Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

from models.schemas import ChannelType, Contact

logger = structlog.get_logger()


# ══════════════════════════════════════════════════════════════
#  ERRORS
# ══════════════════════════════════════════════════════════════

class ChannelError(Exception):
    """Base exception for all channel operations."""

    def __init__(self, message: str, channel: str = "", retryable: bool = False):
        self.channel = channel
        self.retryable = retryable
        super().__init__(message)


class RateLimitedError(ChannelError):
    def __init__(self, channel: str = ""):
        super().__init__(f"Rate limit exceeded for {channel}", channel, retryable=True)


class CircuitOpenError(ChannelError):
    def __init__(self, channel: str = ""):
        super().__init__(f"Circuit breaker open for {channel}", channel, retryable=True)


# ══════════════════════════════════════════════════════════════
#  TOKEN BUCKET RATE LIMITER
# ══════════════════════════════════════════════════════════════

class TokenBucketRateLimiter:
    """
    Async token bucket rate limiter.
    Tokens refill at `rate` per second up to `burst` capacity.
    """

    def __init__(self, rate: float = 10.0, burst: int = 10):
        self.rate = rate
        self.burst = burst
        self._tokens: float = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            wait = min(1.0 / max(self.rate, 0.001), remaining)
            await asyncio.sleep(wait)

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now


# ══════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER
# ══════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Synchronous circuit breaker with failure counting.

    closed → open (after threshold failures) → half_open (after timeout) →
    closed (on success) or open (on failure).
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = "closed"
        self._failure_count = 0
        self._opened_at: float = 0.0
        self._total_failures = 0
        self._total_successes = 0

    @property
    def state(self) -> str:
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                return "half_open"
        return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def record_failure(self):
        self._total_failures += 1
        self._failure_count += 1
        if self._state == "half_open" or self.state == "half_open":
            self._open()
        elif self._failure_count >= self.failure_threshold:
            self._open()

    def record_success(self):
        self._total_successes += 1
        if self._state == "half_open" or self.state == "half_open":
            self._close()
        else:
            self._failure_count = 0

    def _open(self):
        self._state = "open"
        self._opened_at = time.monotonic()
        logger.warning("circuit_opened", failures=self._failure_count)

    def _close(self):
        self._state = "closed"
        self._failure_count = 0

    def reset(self):
        self._close()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self._failure_count,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
        }


# ══════════════════════════════════════════════════════════════
#  CHANNEL METRICS
# ══════════════════════════════════════════════════════════════

class ChannelMetrics:
    """Tracks per-channel send, failure, delivery, and latency metrics."""

    def __init__(self, channel: ChannelType):
        self.channel = channel
        self.messages_sent: int = 0
        self.messages_failed: int = 0
        self.messages_delivered: int = 0
        self.messages_read: int = 0
        self._latencies: list[float] = []
        self._errors: list[str] = []

    def record_send(self, latency_ms: float = 0.0):
        self.messages_sent += 1
        if latency_ms > 0:
            self._latencies.append(latency_ms)

    def record_failure(self, error: str = ""):
        self.messages_failed += 1
        if error:
            self._errors.append(error)

    def record_delivery(self):
        self.messages_delivered += 1

    def record_read(self):
        self.messages_read += 1

    @property
    def avg_latency_ms(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    @property
    def failure_rate(self) -> float:
        total = self.messages_sent + self.messages_failed
        return self.messages_failed / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel.value,
            "sent": self.messages_sent,
            "failed": self.messages_failed,
            "delivered": self.messages_delivered,
            "read": self.messages_read,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "failure_rate": round(self.failure_rate, 4),
            "recent_errors": self._errors[-10:],
        }


# ══════════════════════════════════════════════════════════════
#  DELIVERY STATUS & RECORD
# ══════════════════════════════════════════════════════════════

class DeliveryStatus(str, Enum):
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    BOUNCED = "bounced"


class DeliveryRecord:
    """Tracks full lifecycle of an outbound message with status history."""

    def __init__(self, message_id: str, channel: ChannelType, recipient: str):
        self.message_id = message_id
        self.channel = channel
        self.recipient = recipient
        self.status = DeliveryStatus.QUEUED
        self.status_history: list[dict[str, Any]] = []
        self.channel_message_id: str = ""
        self.error: str = ""
        self.created_at = datetime.now(timezone.utc)
        self.sent_at: Optional[datetime] = None
        self.delivered_at: Optional[datetime] = None
        self.read_at: Optional[datetime] = None

    def update_status(self, new_status: DeliveryStatus, error: str = ""):
        self.status_history.append({
            "from": self.status.value,
            "to": new_status.value,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        self.status = new_status
        if error:
            self.error = error
        if new_status == DeliveryStatus.SENT:
            self.sent_at = datetime.now(timezone.utc)
        elif new_status == DeliveryStatus.DELIVERED:
            self.delivered_at = datetime.now(timezone.utc)
        elif new_status == DeliveryStatus.READ:
            self.read_at = datetime.now(timezone.utc)


# ══════════════════════════════════════════════════════════════
#  MESSAGE DEDUPLICATOR
# ══════════════════════════════════════════════════════════════

class MessageDeduplicator:
    """TTL-based seen-set for deduplicating inbound and outbound messages."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 5000):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._seen: dict[str, float] = {}

    def is_duplicate(self, key: str) -> bool:
        self._prune()
        if key in self._seen:
            return True
        self._seen[key] = time.monotonic()
        return False

    def check_content(self, content: str, recipient: str) -> bool:
        h = hashlib.md5(f"{recipient}:{content}".encode()).hexdigest()
        return self.is_duplicate(f"content_{h}")

    def _prune(self):
        cutoff = time.monotonic() - self.ttl
        expired = [k for k, t in self._seen.items() if t < cutoff]
        for k in expired:
            del self._seen[k]


# ══════════════════════════════════════════════════════════════
#  INPUT SANITIZER
# ══════════════════════════════════════════════════════════════

class InputSanitizer:
    def __init__(self, max_length: int = 10000):
        self.max_length = max_length

    def sanitize(self, content: str) -> str:
        if not content:
            return ""
        content = "".join(
            c for c in content if c in ("\n", "\t", "\r") or (ord(c) >= 32)
        )
        if len(content) > self.max_length:
            content = content[: self.max_length] + "... [truncated]"
        return content.strip()

    def sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        safe = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif isinstance(v, dict):
                safe[k] = self.sanitize_metadata(v)
            elif isinstance(v, list):
                safe[k] = [
                    self.sanitize_metadata(i) if isinstance(i, dict) else i
                    for i in v[:50]
                ]
            else:
                safe[k] = str(v)[:500]
        return safe


# ══════════════════════════════════════════════════════════════
#  CHANNEL ADAPTER — Abstract Base
# ══════════════════════════════════════════════════════════════

class ChannelAdapter(abc.ABC):
    """
    Base class for all channel adapters.

    Subclasses implement _do_send and _do_send_template. The base class
    wraps every send with rate limiting, circuit breaker, retry, and metrics.
    """

    channel_type: ChannelType

    def __init__(self):
        self._initialized = False
        self._config: dict[str, Any] = {}
        self._breaker = CircuitBreaker()
        self._rate_limiter: Optional[TokenBucketRateLimiter] = None
        self._metrics: Optional[ChannelMetrics] = None
        self._deduplicator = MessageDeduplicator()
        self._sanitizer = InputSanitizer()

    def _ensure_metrics(self):
        if self._metrics is None:
            self._metrics = ChannelMetrics(self.channel_type)

    # ── Abstract hooks ────────────────────────────────────────

    @abc.abstractmethod
    async def _do_send(self, contact: Contact, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def _do_send_template(self, contact: Contact, template_name: str, template_data: dict[str, Any]) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def initialize(self, config: dict[str, Any]) -> None:
        ...

    # ── Public send ───────────────────────────────────────────

    async def send_message(self, contact: Contact, content: str, metadata: dict[str, Any] = None) -> dict[str, Any]:
        self._ensure_metrics()
        metadata = metadata or {}
        message_id = metadata.get("message_id", str(uuid.uuid4()))
        start = time.monotonic()

        if self._rate_limiter:
            if not await self._rate_limiter.acquire(timeout=10.0):
                self._metrics.record_failure("rate_limited")
                return {"status": "rate_limited", "message_id": message_id}

        if self._breaker.is_open:
            self._metrics.record_failure("circuit_open")
            return {"status": "circuit_open", "message_id": message_id}

        max_retries = 3
        last_error = ""
        for attempt in range(max_retries):
            try:
                result = await self._do_send(contact, content, metadata)
                latency = (time.monotonic() - start) * 1000

                if result.get("status") not in ("failed",):
                    self._breaker.record_success()
                    self._metrics.record_send(latency)
                    result.setdefault("message_id", message_id)
                    result["latency_ms"] = round(latency, 1)
                    result["attempts"] = attempt + 1
                    return result

                # Definitive failure
                self._breaker.record_failure()
                self._metrics.record_failure(result.get("error", ""))
                result.setdefault("message_id", message_id)
                result["attempts"] = attempt + 1
                return result

            except Exception as e:
                last_error = str(e)
                self._breaker.record_failure()
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(1.0 * (2 ** attempt), 10.0))

        self._metrics.record_failure(last_error)
        return {"status": "failed", "message_id": message_id, "error": last_error}

    async def send_template(self, contact: Contact, template_name: str, template_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_metrics()
        start = time.monotonic()
        try:
            result = await self._do_send_template(contact, template_name, template_data)
            latency = (time.monotonic() - start) * 1000
            if result.get("status") in ("sent", "mock_sent"):
                self._breaker.record_success()
                self._metrics.record_send(latency)
            result["latency_ms"] = round(latency, 1)
            return result
        except Exception as e:
            self._breaker.record_failure()
            self._metrics.record_failure(str(e))
            return {"status": "failed", "error": str(e)}

    # ── Inbound ───────────────────────────────────────────────

    async def handle_inbound(self, raw_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        parsed = await self._parse_inbound(raw_payload)
        if not parsed:
            return None

        msg_id = (
            parsed.get("metadata", {}).get("channel_message_id")
            or parsed.get("metadata", {}).get("message_id")
            or ""
        )
        if msg_id and self._deduplicator.is_duplicate(msg_id):
            return None

        if "content" in parsed:
            parsed["content"] = self._sanitizer.sanitize(parsed.get("content", ""))
        if "metadata" in parsed:
            parsed["metadata"] = self._sanitizer.sanitize_metadata(parsed["metadata"])
        return parsed

    async def _parse_inbound(self, raw_payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        return None

    # ── Health ────────────────────────────────────────────────

    async def health_check(self) -> dict[str, Any]:
        self._ensure_metrics()
        return {
            "channel": self.channel_type.value,
            "initialized": self._initialized,
            "circuit_breaker": self._breaker.stats,
            "metrics": self._metrics.to_dict(),
        }

    def get_address(self, contact: Contact) -> Optional[str]:
        ch = contact.get_channel(self.channel_type)
        return ch.address if ch else None

    async def shutdown(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════
#  CHANNEL REGISTRY
# ══════════════════════════════════════════════════════════════

class ChannelRegistry:
    def __init__(self):
        self._adapters: dict[ChannelType, ChannelAdapter] = {}

    def register(self, adapter: ChannelAdapter):
        self._adapters[adapter.channel_type] = adapter

    def get(self, channel_type: ChannelType) -> Optional[ChannelAdapter]:
        return self._adapters.get(channel_type)

    def get_available(self) -> list[ChannelType]:
        return list(self._adapters.keys())

    def get_healthy_channels(self) -> list[ChannelType]:
        return [ch for ch in self._adapters if not self._adapters[ch]._breaker.is_open]

    async def health_check_all(self) -> dict[str, Any]:
        return {ch.value: await a.health_check() for ch, a in self._adapters.items()}

    async def initialize_all(self, configs: dict[str, Any]):
        for ch, adapter in self._adapters.items():
            try:
                ch_cfg = configs.get(ch.value, {})
                # ChannelConfig dataclass → dict so adapters can call .get()
                if hasattr(ch_cfg, "credentials"):
                    ch_cfg = ch_cfg.credentials
                await adapter.initialize(ch_cfg)
            except Exception as e:
                logger.error("channel_init_failed", channel=ch.value, error=str(e))

    async def shutdown_all(self):
        for a in self._adapters.values():
            try:
                await a.shutdown()
            except Exception:
                pass
