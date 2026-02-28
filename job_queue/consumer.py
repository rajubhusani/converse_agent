"""
Queue Consumer — Pulls follow-up jobs from the queue and drives dispatch.

Runs as one or more async tasks inside the application process.
For horizontal scaling, deploy multiple processes with the same consumer_group;
Redis Streams guarantees each job is delivered to exactly one consumer.

Topology:
  ┌──────────────┐       ┌─────────────────┐       ┌────────────┐
  │ Rules Engine │──pub──▶│ dispatch queue   │──────▶│  Consumer  │
  │ Backend API  │       │ (Redis Stream)   │       │  Worker(s) │
  └──────────────┘       └─────────────────┘       └─────┬──────┘
                                                          │
                         ┌─────────────────┐              │
                         │ delayed (sorted  │◀── retry ───┘
                         │  set / promoter) │              │
                         └────────┬────────┘              │
                                  │ promote               │
                                  ▼                       │
                         ┌─────────────────┐              │
                         │ dispatch queue   │──────────────┘
                         └─────────────────┘
                                                          │
                         ┌─────────────────┐              │
                         │  DLQ            │◀── exhaust ──┘
                         └─────────────────┘
"""
from __future__ import annotations

import asyncio
import structlog
from typing import Any, Optional

from job_queue.message_queue import (
    MessageQueue, QueueJob, Queues,
    get_message_queue,
)

logger = structlog.get_logger()


class FollowUpConsumer:
    """
    Consumes jobs from the dispatch queue and invokes the orchestrator.

    Usage:
        consumer = FollowUpConsumer(orchestrator, queue)
        await consumer.start()       # blocks, runs forever
        await consumer.start_background()  # returns immediately, runs as task
        await consumer.stop()
    """

    def __init__(
        self,
        orchestrator,  # type: core.orchestrator.Orchestrator — avoid circular import
        queue: MessageQueue = None,
        consumer_group: str = "followup-workers",
        consumer_name: str = "",
        concurrency: int = 5,
    ):
        self.orchestrator = orchestrator
        self.queue = queue or get_message_queue()
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.concurrency = concurrency
        self._tasks: list[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(concurrency)
        self._running = False

    async def start(self):
        """Start consuming — blocks until stop() is called."""
        self._running = True
        logger.info("followup_consumer_starting",
                     group=self.consumer_group,
                     concurrency=self.concurrency)

        await self.queue.consume(
            queue=Queues.DISPATCH,
            handler=self._handle_job,
            consumer_group=self.consumer_group,
            consumer_name=self.consumer_name,
        )

    async def start_background(self) -> asyncio.Task:
        """Start consuming in a background task. Returns the task handle."""
        task = asyncio.create_task(self.start())
        self._tasks.append(task)
        return task

    async def stop(self):
        """Gracefully stop all consumer tasks."""
        self._running = False
        self.queue._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("followup_consumer_stopped")

    async def _handle_job(self, job: QueueJob):
        """
        Process a single follow-up dispatch job.

        Flow:
        1. Validate job
        2. Load the FollowUp model
        3. Check terminal status / max attempts
        4. If schedule_check → fire context change instead of dispatch
        5. Otherwise → delegate to orchestrator.dispatch_followup()
        6. On failure, nack (retry/DLQ logic in queue layer)
        """
        async with self._semaphore:
            logger.info("processing_job",
                        job_id=job.job_id,
                        followup_id=job.followup_id,
                        contact_id=job.contact_id,
                        attempt=job.attempt,
                        priority=job.priority)

            try:
                # Load follow-up from context store
                followup = await self.orchestrator.context.store.get_followup(
                    job.followup_id
                )

                if not followup:
                    logger.error("followup_not_found",
                                 followup_id=job.followup_id,
                                 job_id=job.job_id)
                    return  # Don't retry — data is missing

                # Check if already completed/cancelled
                from models.schemas import FollowUpStatus
                terminal = {FollowUpStatus.COMPLETED, FollowUpStatus.CANCELLED,
                            FollowUpStatus.FAILED}
                if followup.status in terminal:
                    logger.info("followup_already_terminal",
                                followup_id=followup.id,
                                status=followup.status.value)
                    return

                # Check max attempts
                if followup.attempt_count >= followup.max_attempts:
                    logger.warning("followup_max_attempts_reached",
                                   followup_id=followup.id,
                                   attempts=followup.attempt_count,
                                   max=followup.max_attempts)
                    await self.orchestrator._handle_max_attempts(followup)
                    return

                # Route: schedule_check → fire context change (not a message dispatch)
                if followup.metadata.get("is_scheduled_check"):
                    await self._handle_scheduled_check(followup)
                    return

                # Normal dispatch
                result = await self.orchestrator.dispatch_followup(followup)

                status = result.get("status", "")
                if status == "sent":
                    logger.info("job_dispatch_success",
                                job_id=job.job_id,
                                channel=result.get("channel"),
                                conversation_id=result.get("conversation_id"))
                elif status == "failed":
                    reason = result.get("reason", "unknown")
                    logger.warning("job_dispatch_failed",
                                   job_id=job.job_id,
                                   reason=reason)
                    raise DispatchError(f"Dispatch failed: {reason}")
                else:
                    logger.info("job_dispatch_result",
                                job_id=job.job_id,
                                result=result)

            except DispatchError:
                raise  # re-raise so consume() routes to nack
            except Exception as e:
                logger.error("job_processing_error",
                             job_id=job.job_id,
                             error=str(e),
                             exc_info=True)
                raise  # route to nack

    async def _handle_scheduled_check(self, followup):
        """
        A schedule_check follow-up fires a ContextChange instead of
        sending a message. This drives state machine timeout transitions
        like "no payment received after 48h".
        """
        from context.tracker import ContextChange

        trigger_type = followup.business_context.get("check_trigger_type", "timeout")
        trigger_value = followup.business_context.get("check_trigger_value", "scheduled_check")
        conversation_id = followup.conversation_id

        if not conversation_id:
            logger.warning("scheduled_check_no_conversation",
                           followup_id=followup.id)
            followup.status = FollowUpStatus.FAILED
            followup.outcome = "No conversation_id for scheduled check"
            await self.orchestrator.context.store.update_followup(followup)
            return

        logger.info("firing_scheduled_check",
                     followup_id=followup.id,
                     trigger=f"{trigger_type}:{trigger_value}",
                     conversation_id=conversation_id)

        change = await self.orchestrator.context.build_context_change(
            conversation_id=conversation_id,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            extra_data=followup.business_context,
        )
        results = await self.orchestrator.apply_context_change(change)

        # Mark completed
        followup.status = FollowUpStatus.COMPLETED
        followup.completed_at = __import__("datetime").datetime.utcnow()
        followup.outcome = f"Scheduled check fired: {len(results)} transitions"
        await self.orchestrator.context.store.update_followup(followup)

        logger.info("scheduled_check_complete",
                     followup_id=followup.id,
                     transitions=len(results))


class DispatchError(Exception):
    """Raised when follow-up dispatch fails and should be retried."""
    pass


# ──────────────────────────────────────────────────────────────
#  Delayed Job Promoter
# ──────────────────────────────────────────────────────────────

class DelayedJobPromoter:
    """
    Background task that periodically moves delayed/retry jobs
    whose scheduled_at has arrived into the dispatch queue.

    For Redis: runs ZRANGEBYSCORE + XADD pipeline.
    For in-memory: already handled inside InMemoryMessageQueue.
    """

    def __init__(self, queue: MessageQueue = None, interval_seconds: int = 5):
        self.queue = queue or get_message_queue()
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None

    async def start_background(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._run())
        return self._task

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        logger.info("delayed_promoter_started", interval=self.interval)
        while True:
            try:
                await self.queue.promote_delayed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("promoter_error", error=str(e))
            await asyncio.sleep(self.interval)
