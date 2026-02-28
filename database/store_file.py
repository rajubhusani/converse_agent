"""
FileContextStore — JSON file-backed store with persistence across restarts.

Data layout:
  {data_dir}/
    contacts.json
    conversations.json
    messages.json
    followups.json
    state_bindings.json
    voice_calls.json

Features:
  - Survives process restarts (unlike InMemoryContextStore)
  - No external dependencies (no database server, no Redis)
  - Writes are batched — flush on every mutation with debounce option
  - Single-process only (no concurrent write safety)

Best for: small deployments, demos, edge devices, air-gapped environments.
"""
from __future__ import annotations

import asyncio
import json
import os
import structlog
from pathlib import Path
from typing import Any, Optional

from database.store_memory import InMemoryContextStore
from database.store_base import BaseContextStore
from models.schemas import Contact

logger = structlog.get_logger()

_COLLECTIONS = [
    "contacts", "conversations", "messages",
    "followups", "state_bindings", "voice_calls",
]


class FileContextStore(InMemoryContextStore):
    """
    Extends InMemoryContextStore with JSON file persistence.

    On init: loads all data from JSON files into memory.
    On every write: flushes the changed collection to disk.

    For higher performance, set flush_interval_s > 0 to batch writes.
    """

    def __init__(self, data_dir: str = "./data", flush_interval_s: float = 0):
        super().__init__()
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._flush_interval = flush_interval_s
        self._dirty: set[str] = set()
        self._flush_task: Optional[asyncio.Task] = None
        self._load_all()
        logger.info("file_store_initialized", data_dir=str(self._data_dir))

    # ── Load / Save ───────────────────────────────────────

    def _file_path(self, collection: str) -> Path:
        return self._data_dir / f"{collection}.json"

    def _load_all(self):
        """Load all collections from disk."""
        for collection in _COLLECTIONS:
            path = self._file_path(collection)
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    self._set_collection(collection, data)
                    logger.debug("file_store_loaded",
                                 collection=collection,
                                 records=len(data) if isinstance(data, dict) else "N/A")
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning("file_store_load_error",
                                   collection=collection, error=str(e))

    def _set_collection(self, collection: str, data: Any):
        """Restore a collection from loaded JSON data."""
        if collection == "contacts":
            self._contacts = data if isinstance(data, dict) else {}
            # Rebuild indexes
            self._address_index.clear()
            self._external_id_index.clear()
            for cid, c in self._contacts.items():
                for ch in (c.get("channels") or []):
                    self._address_index[f"{ch['channel']}:{ch['address']}"] = cid
                if c.get("external_id"):
                    self._external_id_index[c["external_id"]] = cid
        elif collection == "conversations":
            self._conversations = data if isinstance(data, dict) else {}
        elif collection == "messages":
            # messages stored as {conv_id: [msg_list]}
            if isinstance(data, dict):
                from collections import defaultdict
                self._messages = defaultdict(list, data)
            else:
                from collections import defaultdict
                self._messages = defaultdict(list)
        elif collection == "followups":
            self._followups = data if isinstance(data, dict) else {}
            # Rebuild ext index
            self._followup_ext_index.clear()
            for fid, f in self._followups.items():
                if f.get("external_id"):
                    self._followup_ext_index[f["external_id"]] = fid
        elif collection == "state_bindings":
            self._state_bindings = data if isinstance(data, dict) else {}
        elif collection == "voice_calls":
            self._voice_calls = data if isinstance(data, dict) else {}

    def _get_collection_data(self, collection: str) -> Any:
        """Get serializable data for a collection."""
        mapping = {
            "contacts": self._contacts,
            "conversations": self._conversations,
            "messages": dict(self._messages),
            "followups": self._followups,
            "state_bindings": self._state_bindings,
            "voice_calls": self._voice_calls,
        }
        return mapping.get(collection, {})

    def _flush_collection(self, collection: str):
        """Write a single collection to disk."""
        path = self._file_path(collection)
        data = self._get_collection_data(collection)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.rename(path)  # atomic on POSIX

    def _mark_dirty(self, *collections: str):
        """Mark collections as needing a flush."""
        if self._flush_interval <= 0:
            # Immediate flush
            for c in collections:
                self._flush_collection(c)
        else:
            self._dirty.update(collections)
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.get_event_loop().create_task(
                    self._deferred_flush()
                )

    async def _deferred_flush(self):
        """Batch flush after interval."""
        await asyncio.sleep(self._flush_interval)
        dirty = self._dirty.copy()
        self._dirty.clear()
        for c in dirty:
            self._flush_collection(c)

    def flush_all(self):
        """Force flush all collections to disk."""
        for c in _COLLECTIONS:
            self._flush_collection(c)
        logger.info("file_store_flushed_all")

    # ── Override write methods to trigger persistence ──────

    async def upsert_contact(self, contact: Contact) -> Contact:
        result = await super().upsert_contact(contact)
        self._mark_dirty("contacts")
        return result

    async def create_conversation(self, contact_id: str, channel: str = "", **kwargs) -> dict:
        result = await super().create_conversation(contact_id, channel, **kwargs)
        self._mark_dirty("conversations")
        return result

    async def update_conversation(self, conversation_id: str, **kwargs) -> None:
        await super().update_conversation(conversation_id, **kwargs)
        self._mark_dirty("conversations")

    async def add_message(self, conversation_id: str, direction: str, channel: str,
                          content: str, metadata: dict = None, channel_message_id: str = "") -> dict:
        result = await super().add_message(
            conversation_id, direction, channel, content, metadata, channel_message_id
        )
        self._mark_dirty("messages", "conversations")
        return result

    async def create_followup(self, data: dict[str, Any]) -> dict:
        result = await super().create_followup(data)
        self._mark_dirty("followups")
        return result

    async def update_followup(self, followup_id: str, **kwargs) -> None:
        await super().update_followup(followup_id, **kwargs)
        self._mark_dirty("followups")

    async def upsert_state_binding(self, conversation_id: str, process_type: str, **kwargs) -> None:
        await super().upsert_state_binding(conversation_id, process_type, **kwargs)
        self._mark_dirty("state_bindings")

    async def save_voice_call(self, call_data: dict[str, Any]) -> None:
        await super().save_voice_call(call_data)
        self._mark_dirty("voice_calls")
