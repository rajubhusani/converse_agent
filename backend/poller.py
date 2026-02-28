"""
Backend Poller — periodic sync of follow-ups from your business backend.

Use this when your backend can't push events via webhooks.
Runs as a background task inside the FastAPI lifespan.

Flow:
    Backend (ERP/CRM) → Poller fetches pending follow-ups
    → Deduplicates against existing database records
    → Creates new follow-ups and contacts in PostgreSQL
    → Enqueues to Redis Streams for dispatch
"""
from __future__ import annotations

import asyncio
import structlog
from typing import Any, Optional

from backend.connector import BackendConnector, create_backend_connector
from database.store import PostgresContextStore
from models.schemas import Contact, ContactChannel, ChannelType

logger = structlog.get_logger()


class BackendPoller:
    """
    Polls your backend for new follow-ups and syncs them into the system.

    Configure poll interval and batch size in settings:
        backend:
          poll_interval_seconds: 60
          poll_batch_size: 50
    """

    def __init__(
        self,
        store: PostgresContextStore,
        backend: BackendConnector,
        enqueue_fn=None,
        poll_interval_s: int = 60,
        batch_size: int = 50,
    ):
        self.store = store
        self.backend = backend
        self._enqueue_fn = enqueue_fn  # async fn(followup_dict) → enqueues to Redis
        self.poll_interval_s = poll_interval_s
        self.batch_size = batch_size
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the polling loop as a background task."""
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="backend_poller")
        logger.info("backend_poller_started", interval_s=self.poll_interval_s)

    async def stop(self) -> None:
        """Gracefully stop the poller."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("backend_poller_stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop — runs until stopped."""
        while self._running:
            try:
                await self.poll_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("poll_cycle_error", error=str(e))

            await asyncio.sleep(self.poll_interval_s)

    async def poll_cycle(self) -> dict[str, int]:
        """
        Single poll cycle:
        1. Fetch pending follow-ups from backend
        2. Deduplicate against existing records
        3. Ensure contacts exist in our database
        4. Create follow-ups and enqueue for dispatch

        Returns counts: {"fetched": N, "new": N, "skipped": N, "errors": N}
        """
        stats = {"fetched": 0, "new": 0, "skipped": 0, "errors": 0}

        try:
            raw_followups = await self.backend.get_pending_followups(
                filters={"limit": self.batch_size}
            )
            stats["fetched"] = len(raw_followups)
        except Exception as e:
            logger.error("backend_fetch_failed", error=str(e))
            stats["errors"] = 1
            return stats

        for raw in raw_followups:
            try:
                external_id = raw.get("id", raw.get("external_id", ""))
                if not external_id:
                    logger.warning("followup_missing_id", raw=raw)
                    stats["errors"] += 1
                    continue

                # Check if we already have this follow-up
                existing = await self.store.get_followup_by_external_id(external_id)
                if existing:
                    stats["skipped"] += 1
                    continue

                # Ensure contact exists
                contact = await self._ensure_contact(raw.get("contact_id", ""), raw)

                if not contact:
                    logger.warning("contact_not_found", contact_id=raw.get("contact_id"))
                    stats["errors"] += 1
                    continue

                # Create follow-up in our database
                followup_data = {
                    "contact_id": contact.id,
                    "external_id": external_id,
                    "reason": raw.get("reason", raw.get("description", "")),
                    "priority": raw.get("priority", "medium"),
                    "process_type": raw.get("process_type", ""),
                    "entity_type": raw.get("entity_type", ""),
                    "entity_id": raw.get("entity_id", ""),
                    "business_context": raw.get("business_context", {}),
                    "channel_priority": raw.get("channel_priority", ["voice", "whatsapp", "email"]),
                }

                followup = await self.store.create_followup(followup_data)

                # Enqueue for dispatch
                if self._enqueue_fn:
                    await self._enqueue_fn(followup)

                stats["new"] += 1
                logger.info(
                    "followup_synced",
                    external_id=external_id,
                    contact=contact.name,
                    priority=followup_data["priority"],
                )

            except Exception as e:
                logger.error("followup_sync_error", error=str(e), raw_id=raw.get("id"))
                stats["errors"] += 1

        if stats["new"] > 0:
            logger.info("poll_cycle_complete", **stats)

        return stats

    async def _ensure_contact(self, contact_id: str, raw_followup: dict) -> Optional[Contact]:
        """
        Ensure the contact exists in our database, creating if needed.

        First checks our DB by external_id, then fetches from backend if missing.
        """
        # Check if contact exists by external ID
        contact = await self.store.get_contact_by_external_id(contact_id)
        if contact:
            return contact

        # Fetch from backend
        try:
            raw_contact = await self.backend.get_contact(contact_id)
            if not raw_contact:
                return None
        except Exception as e:
            logger.error("contact_fetch_failed", contact_id=contact_id, error=str(e))
            return None

        # Map backend fields to our Contact model
        # Override normalize_contact() in your BackendConnector subclass
        # if your backend uses different field names
        contact = self.backend.normalize_contact(raw_contact)

        # Persist
        await self.store.upsert_contact(contact)
        return contact


class EventIngestionHandler:
    """
    Handles push events from your backend (alternative to polling).

    Your backend POSTs to /api/v1/events:
    {
        "event_name": "invoice_overdue",
        "data": {
            "invoice_number": "INV-2026-0042",
            "contact_id": "c_123",
            "amount": 52400,
            "days_overdue": 3
        }
    }
    """

    def __init__(self, store: PostgresContextStore, enqueue_fn=None):
        self.store = store
        self._enqueue_fn = enqueue_fn

    async def handle_event(self, event_name: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process a business event and create follow-ups if rules match.

        Returns: {"status": "processed", "followups_created": N}
        """
        logger.info("business_event_received", event=event_name, data_keys=list(data.keys()))

        # The Rules Engine evaluates this event and decides whether
        # to create a follow-up. This is handled by the Orchestrator.
        return {
            "status": "received",
            "event_name": event_name,
            "data": data,
        }
