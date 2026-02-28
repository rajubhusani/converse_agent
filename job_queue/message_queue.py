"""
Message Queue — Abstract interface with Redis Streams and in-memory backends.

Queue Topology:
  followup:dispatch    — New follow-ups ready for immediate dispatch
  followup:delayed     — Follow-ups with a future execution time (sorted set in Redis)
  followup:retry       — Failed dispatches queued for retry with backoff
  followup:dlq         — Dead-letter queue for permanently failed jobs

Message Schema:
  {
      "job_id":          unique job identifier,
      "followup_id":     FollowUp model ID,
      "rule_id":         originating rule,
      "contact_id":      target contact,
      "priority":        urgent|high|medium|low,
      "attempt":         current attempt number (for retries),
      "max_attempts":    ceiling before DLQ,
      "scheduled_at":    ISO timestamp when the job should execute,
      "created_at":      ISO timestamp when the job was enqueued,
      "business_context": dict with template, context_key, etc.,
      "metadata":        arbitrary extra data,
  }
"""
from __future__ import annotations

import asyncio
import json
import uuid
import structlog
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator, Callable, Optional

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────
#  Job Model
# ──────────────────────────────────────────────────────────────

@dataclass
class QueueJob:
    """A unit of work on the queue."""
    followup_id: str
    rule_id: str
    contact_id: str
    priority: str = "medium"
    attempt: int = 0
    max_attempts: int = 3
    scheduled_at: str = ""
    created_at: str = ""
    business_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    job_id: str = ""

    def __post_init__(self):
        if not self.job_id:
            self.job_id = f"job_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.scheduled_at:
            self.scheduled_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["business_context"] = json.dumps(d["business_context"])
        d["metadata"] = json.dumps(d["metadata"])
        d["attempt"] = str(d["attempt"])
        d["max_attempts"] = str(d["max_attempts"])
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueJob:
        data = dict(data)  # copy
        if isinstance(data.get("business_context"), str):
            data["business_context"] = json.loads(data["business_context"])
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        data["attempt"] = int(data.get("attempt", 0))
        data["max_attempts"] = int(data.get("max_attempts", 3))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def is_scheduled_now(self) -> bool:
        if not self.scheduled_at:
            return True
        try:
            target = datetime.fromisoformat(self.scheduled_at)
            return datetime.utcnow() >= target
        except ValueError:
            return True

    def next_retry_job(self, backoff_seconds: int = 60) -> QueueJob:
        """Create a copy with incremented attempt and backoff delay."""
        retry_at = datetime.utcnow() + timedelta(
            seconds=backoff_seconds * (2 ** self.attempt)  # exponential backoff
        )
        return QueueJob(
            followup_id=self.followup_id,
            rule_id=self.rule_id,
            contact_id=self.contact_id,
            priority=self.priority,
            attempt=self.attempt + 1,
            max_attempts=self.max_attempts,
            scheduled_at=retry_at.isoformat(),
            created_at=self.created_at,
            business_context=self.business_context,
            metadata={**self.metadata, "last_failure_at": datetime.utcnow().isoformat()},
            job_id=self.job_id,  # same job_id across retries for tracing
        )


# ──────────────────────────────────────────────────────────────
#  Queue Names
# ──────────────────────────────────────────────────────────────

class Queues:
    DISPATCH = "followup:dispatch"
    DELAYED = "followup:delayed"
    RETRY = "followup:retry"
    DLQ = "followup:dlq"


# ──────────────────────────────────────────────────────────────
#  Abstract Interface
# ──────────────────────────────────────────────────────────────

class MessageQueue(ABC):
    """Abstract message queue interface."""

    @abstractmethod
    async def connect(self):
        """Establish connection to the queue backend."""
        ...

    @abstractmethod
    async def close(self):
        """Gracefully shut down."""
        ...

    @abstractmethod
    async def publish(self, queue: str, job: QueueJob):
        """Publish a job to a queue."""
        ...

    @abstractmethod
    async def publish_delayed(self, job: QueueJob):
        """Publish a job that should execute at job.scheduled_at."""
        ...

    @abstractmethod
    async def consume(
        self,
        queue: str,
        handler: Callable[[QueueJob], Any],
        consumer_group: str = "default",
        consumer_name: str = "",
        batch_size: int = 10,
    ):
        """
        Start consuming from a queue. Blocks and calls handler for each job.
        Supports consumer groups for horizontal scaling.
        """
        ...

    @abstractmethod
    async def ack(self, queue: str, job_id: str, consumer_group: str = "default"):
        """Acknowledge successful processing of a job."""
        ...

    @abstractmethod
    async def nack(self, queue: str, job: QueueJob, consumer_group: str = "default"):
        """Negative-acknowledge — route to retry or DLQ."""
        ...

    @abstractmethod
    async def queue_length(self, queue: str) -> int:
        """Return the number of pending jobs in a queue."""
        ...

    @abstractmethod
    async def peek(self, queue: str, count: int = 10) -> list[QueueJob]:
        """Peek at jobs without consuming them."""
        ...

    @abstractmethod
    async def promote_delayed(self):
        """Move delayed jobs whose scheduled_at has arrived to the dispatch queue."""
        ...


# ──────────────────────────────────────────────────────────────
#  Redis Streams Implementation
# ──────────────────────────────────────────────────────────────

class RedisMessageQueue(MessageQueue):
    """
    Production queue backed by Redis Streams + Sorted Sets.

    - Dispatch/Retry queues use Redis Streams with consumer groups
    - Delayed queue uses a Redis Sorted Set (ZRANGEBYSCORE for promotion)
    - DLQ uses a Redis Stream for inspection
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._redis = None
        self._running = False

    async def connect(self):
        import redis.asyncio as aioredis
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            max_connections=20,
        )
        await self._redis.ping()
        logger.info("redis_queue_connected", url=self._redis_url)

    async def close(self):
        self._running = False
        if self._redis:
            await self._redis.close()

    async def _ensure_group(self, queue: str, group: str):
        """Create consumer group if it doesn't exist."""
        try:
            await self._redis.xgroup_create(queue, group, id="0", mkstream=True)
        except Exception:
            pass  # group already exists

    async def publish(self, queue: str, job: QueueJob):
        await self._redis.xadd(queue, job.to_dict())
        logger.info("job_published",
                     queue=queue,
                     job_id=job.job_id,
                     followup_id=job.followup_id,
                     priority=job.priority)

    async def publish_delayed(self, job: QueueJob):
        score = datetime.fromisoformat(job.scheduled_at).timestamp()
        payload = json.dumps(job.to_dict())
        await self._redis.zadd(Queues.DELAYED, {payload: score})
        logger.info("delayed_job_published",
                     job_id=job.job_id,
                     scheduled_at=job.scheduled_at)

    async def consume(
        self,
        queue: str,
        handler: Callable[[QueueJob], Any],
        consumer_group: str = "default",
        consumer_name: str = "",
        batch_size: int = 10,
    ):
        if not consumer_name:
            consumer_name = f"worker_{uuid.uuid4().hex[:8]}"

        await self._ensure_group(queue, consumer_group)
        self._running = True
        logger.info("consumer_started",
                     queue=queue,
                     group=consumer_group,
                     consumer=consumer_name)

        while self._running:
            try:
                # Read new messages assigned to this consumer
                messages = await self._redis.xreadgroup(
                    groupname=consumer_group,
                    consumername=consumer_name,
                    streams={queue: ">"},
                    count=batch_size,
                    block=2000,  # block 2s waiting for messages
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        job = QueueJob.from_dict(fields)
                        try:
                            await handler(job)
                            await self._redis.xack(queue, consumer_group, message_id)
                            logger.debug("job_acked",
                                         job_id=job.job_id,
                                         message_id=message_id)
                        except Exception as e:
                            logger.error("job_handler_error",
                                         job_id=job.job_id,
                                         error=str(e))
                            await self.nack(queue, job, consumer_group)
                            await self._redis.xack(queue, consumer_group, message_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("consumer_error", queue=queue, error=str(e))
                await asyncio.sleep(1)

    async def ack(self, queue: str, job_id: str, consumer_group: str = "default"):
        # Ack is handled inline in consume(), exposed for external use
        pass

    async def nack(self, queue: str, job: QueueJob, consumer_group: str = "default"):
        if job.attempt + 1 >= job.max_attempts:
            # Exhausted retries → DLQ
            job.metadata["dlq_reason"] = f"Exceeded {job.max_attempts} attempts"
            await self.publish(Queues.DLQ, job)
            logger.warning("job_moved_to_dlq",
                           job_id=job.job_id,
                           attempts=job.attempt + 1)
        else:
            # Retry with exponential backoff
            retry_job = job.next_retry_job()
            await self.publish_delayed(retry_job)
            logger.info("job_scheduled_for_retry",
                        job_id=job.job_id,
                        attempt=retry_job.attempt,
                        scheduled_at=retry_job.scheduled_at)

    async def queue_length(self, queue: str) -> int:
        return await self._redis.xlen(queue)

    async def peek(self, queue: str, count: int = 10) -> list[QueueJob]:
        messages = await self._redis.xrange(queue, count=count)
        return [QueueJob.from_dict(fields) for _, fields in messages]

    async def promote_delayed(self):
        """Move jobs whose scheduled_at <= now from sorted set to dispatch stream."""
        now = datetime.utcnow().timestamp()
        ready = await self._redis.zrangebyscore(Queues.DELAYED, "-inf", now)

        if not ready:
            return

        pipe = self._redis.pipeline()
        for payload in ready:
            job_data = json.loads(payload)
            job = QueueJob.from_dict(job_data)
            pipe.xadd(Queues.DISPATCH, job.to_dict())
            pipe.zrem(Queues.DELAYED, payload)
        await pipe.execute()

        logger.info("delayed_jobs_promoted", count=len(ready))


# ──────────────────────────────────────────────────────────────
#  In-Memory Implementation (Development)
# ──────────────────────────────────────────────────────────────

class InMemoryMessageQueue(MessageQueue):
    """
    Development/test queue backed by asyncio primitives.
    Single-process only — no consumer groups or persistence.
    """

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._delayed: list[tuple[float, QueueJob]] = []  # (timestamp, job)
        self._dlq: list[QueueJob] = []
        self._running = False
        self._delayed_promoter_task: Optional[asyncio.Task] = None

    def _get_queue(self, name: str) -> asyncio.Queue:
        if name not in self._queues:
            self._queues[name] = asyncio.Queue()
        return self._queues[name]

    async def connect(self):
        self._running = True
        self._delayed_promoter_task = asyncio.create_task(self._promote_loop())
        logger.info("inmemory_queue_connected")

    async def close(self):
        self._running = False
        if self._delayed_promoter_task:
            self._delayed_promoter_task.cancel()
            try:
                await self._delayed_promoter_task
            except asyncio.CancelledError:
                pass

    async def publish(self, queue: str, job: QueueJob):
        q = self._get_queue(queue)
        await q.put(job)
        logger.info("job_published",
                     queue=queue,
                     job_id=job.job_id,
                     followup_id=job.followup_id)

    async def publish_delayed(self, job: QueueJob):
        score = datetime.fromisoformat(job.scheduled_at).timestamp()
        self._delayed.append((score, job))
        self._delayed.sort(key=lambda x: x[0])
        logger.info("delayed_job_published",
                     job_id=job.job_id,
                     scheduled_at=job.scheduled_at)

    async def consume(
        self,
        queue: str,
        handler: Callable[[QueueJob], Any],
        consumer_group: str = "default",
        consumer_name: str = "",
        batch_size: int = 10,
    ):
        q = self._get_queue(queue)
        self._running = True
        logger.info("consumer_started", queue=queue)

        while self._running:
            try:
                job = await asyncio.wait_for(q.get(), timeout=2.0)
                try:
                    await handler(job)
                except Exception as e:
                    logger.error("job_handler_error",
                                 job_id=job.job_id,
                                 error=str(e))
                    await self.nack(queue, job)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def ack(self, queue: str, job_id: str, consumer_group: str = "default"):
        pass  # no-op for in-memory

    async def nack(self, queue: str, job: QueueJob, consumer_group: str = "default"):
        if job.attempt + 1 >= job.max_attempts:
            self._dlq.append(job)
            job.metadata["dlq_reason"] = f"Exceeded {job.max_attempts} attempts"
            logger.warning("job_moved_to_dlq",
                           job_id=job.job_id,
                           attempts=job.attempt + 1)
        else:
            retry_job = job.next_retry_job()
            await self.publish_delayed(retry_job)

    async def queue_length(self, queue: str) -> int:
        return self._get_queue(queue).qsize()

    async def peek(self, queue: str, count: int = 10) -> list[QueueJob]:
        q = self._get_queue(queue)
        items = []
        # asyncio.Queue doesn't support peek natively — drain and re-add
        while not q.empty() and len(items) < count:
            items.append(q.get_nowait())
        for item in items:
            await q.put(item)
        return items

    async def promote_delayed(self):
        now = datetime.utcnow().timestamp()
        ready = [(ts, job) for ts, job in self._delayed if ts <= now]
        self._delayed = [(ts, job) for ts, job in self._delayed if ts > now]

        for _, job in ready:
            await self.publish(Queues.DISPATCH, job)

        if ready:
            logger.info("delayed_jobs_promoted", count=len(ready))

    async def _promote_loop(self):
        """Background loop to promote delayed jobs."""
        while self._running:
            try:
                await self.promote_delayed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("delayed_promote_error", error=str(e))
            await asyncio.sleep(5)


# ──────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────

_instance: Optional[MessageQueue] = None


def create_message_queue(queue_config: dict[str, Any] = None) -> MessageQueue:
    """Factory: create the appropriate queue backend."""
    global _instance
    if _instance:
        return _instance

    config = queue_config or {}
    backend = config.get("backend", "memory")

    if backend == "redis":
        url = config.get("redis_url", "redis://localhost:6379")
        _instance = RedisMessageQueue(redis_url=url)
    else:
        _instance = InMemoryMessageQueue()

    return _instance


def get_message_queue() -> MessageQueue:
    """Return the singleton queue instance."""
    global _instance
    if _instance is None:
        _instance = create_message_queue()
    return _instance
