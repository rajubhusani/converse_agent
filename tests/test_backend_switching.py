"""
Tests — Database Backend Switching & Store Factory

Tests all configurable backends:
  Database:  PostgreSQL, MySQL, SQLite (via URL)
  Store:     SQL, Memory, File
  Queue:     Redis, Memory
  Adapter:   StoreAdapter bridges ContextStore ↔ BaseContextStore

Run:
  pytest tests/test_backend_switching.py -v
"""
import asyncio
import json
import os
import shutil
import tempfile
import pytest
import pytest_asyncio

from unittest.mock import patch, MagicMock, AsyncMock

# ──────────────────────────────────────────────────────────────
#  Config / Settings Tests
# ──────────────────────────────────────────────────────────────


class TestDatabaseConfig:
    """Test that settings correctly parse all database configurations."""

    def test_default_config(self):
        from config.settings import DatabaseConfig
        cfg = DatabaseConfig()
        assert cfg.store_backend == "memory"
        assert cfg.store_file_dir == "./data"
        assert "sqlite" in cfg.url

    def test_postgresql_url(self):
        from config.settings import DatabaseConfig
        cfg = DatabaseConfig(url="postgresql://user:pass@host:5432/db")
        assert cfg.url == "postgresql://user:pass@host:5432/db"

    def test_mysql_url(self):
        from config.settings import DatabaseConfig
        cfg = DatabaseConfig(url="mysql://user:pass@host:3306/db")
        assert cfg.url == "mysql://user:pass@host:3306/db"

    def test_sqlite_url(self):
        from config.settings import DatabaseConfig
        cfg = DatabaseConfig(url="sqlite:///./test.db")
        assert cfg.url == "sqlite:///./test.db"

    def test_store_backend_options(self):
        from config.settings import DatabaseConfig
        for backend in ("sql", "memory", "file"):
            cfg = DatabaseConfig(store_backend=backend)
            assert cfg.store_backend == backend

    def test_queue_config_defaults(self):
        from config.settings import QueueConfig
        cfg = QueueConfig()
        assert cfg.backend == "memory"
        assert cfg.redis_url == "redis://localhost:6379"

    def test_queue_config_redis(self):
        from config.settings import QueueConfig
        cfg = QueueConfig(backend="redis", redis_url="redis://custom:6380")
        assert cfg.backend == "redis"
        assert cfg.redis_url == "redis://custom:6380"


# ──────────────────────────────────────────────────────────────
#  Session URL Mapping Tests
# ──────────────────────────────────────────────────────────────


class TestSessionURLMapping:
    """Test that database URLs are correctly mapped to async drivers."""

    def test_postgresql_url_mapping(self):
        from database.session import _to_async_url
        assert _to_async_url("postgresql://u:p@h/d") == "postgresql+asyncpg://u:p@h/d"

    def test_postgres_url_mapping(self):
        from database.session import _to_async_url
        assert _to_async_url("postgres://u:p@h/d") == "postgresql+asyncpg://u:p@h/d"

    def test_mysql_url_mapping(self):
        from database.session import _to_async_url
        assert _to_async_url("mysql://u:p@h/d") == "mysql+aiomysql://u:p@h/d"

    def test_mysql_pymysql_url_mapping(self):
        from database.session import _to_async_url
        assert _to_async_url("mysql+pymysql://u:p@h/d") == "mysql+aiomysql://u:p@h/d"

    def test_sqlite_url_mapping(self):
        from database.session import _to_async_url
        assert _to_async_url("sqlite:///./test.db") == "sqlite+aiosqlite:///./test.db"

    def test_already_async_url_passthrough(self):
        from database.session import _to_async_url
        url = "postgresql+asyncpg://u:p@h/d"
        assert _to_async_url(url) == url

    def test_engine_kwargs_sqlite(self):
        from database.session import _engine_kwargs
        kwargs = _engine_kwargs("sqlite+aiosqlite:///./test.db")
        assert "connect_args" in kwargs
        assert "pool_size" not in kwargs

    def test_engine_kwargs_postgresql(self):
        from database.session import _engine_kwargs
        kwargs = _engine_kwargs("postgresql+asyncpg://u:p@h/d")
        assert "pool_size" in kwargs
        assert kwargs["pool_size"] == 10

    def test_engine_kwargs_mysql(self):
        from database.session import _engine_kwargs
        kwargs = _engine_kwargs("mysql+aiomysql://u:p@h/d")
        assert "pool_size" in kwargs
        assert "pool_recycle" in kwargs
        assert kwargs["pool_recycle"] == 1800


# ──────────────────────────────────────────────────────────────
#  Store Factory Tests
# ──────────────────────────────────────────────────────────────


class TestStoreFactory:
    """Test that store factory creates the correct backend."""

    def setup_method(self):
        from database.store_factory import reset_store
        reset_store()

    def teardown_method(self):
        from database.store_factory import reset_store
        reset_store()

    def test_create_memory_store(self):
        from database.store_factory import create_store
        from database.store_memory import InMemoryContextStore
        store = create_store({"store_backend": "memory"})
        assert isinstance(store, InMemoryContextStore)

    def test_create_memory_store_default(self):
        from database.store_factory import create_store
        from database.store_memory import InMemoryContextStore
        store = create_store({})
        assert isinstance(store, InMemoryContextStore)

    def test_create_file_store(self):
        from database.store_factory import create_store, reset_store
        from database.store_file import FileContextStore
        tmpdir = tempfile.mkdtemp()
        try:
            store = create_store({"store_backend": "file", "store_file_dir": tmpdir})
            assert isinstance(store, FileContextStore)
        finally:
            reset_store()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_create_sql_store(self):
        from database.store_factory import create_store
        from database.store import SqlContextStore
        store = create_store({"store_backend": "sql"})
        assert isinstance(store, SqlContextStore)

    def test_singleton_behavior(self):
        from database.store_factory import create_store, get_store, reset_store
        s1 = create_store({"store_backend": "memory"})
        s2 = get_store()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        from database.store_factory import create_store, get_store, reset_store
        s1 = create_store({"store_backend": "memory"})
        reset_store()
        s2 = create_store({"store_backend": "memory"})
        assert s1 is not s2


# ──────────────────────────────────────────────────────────────
#  InMemoryContextStore Tests
# ──────────────────────────────────────────────────────────────


class TestInMemoryContextStore:
    """Full CRUD test suite for InMemoryContextStore."""

    @pytest_asyncio.fixture
    async def store(self):
        from database.store_memory import InMemoryContextStore
        return InMemoryContextStore()

    @pytest.fixture
    def sample_contact(self):
        from models.schemas import Contact, ContactChannel, ChannelType
        return Contact(
            id="c_test_001",
            name="Sharma Ji",
            role="dealer",
            organization="Sharma Traders",
            channels=[
                ContactChannel(channel=ChannelType.VOICE, address="+919876543210", preferred=True),
                ContactChannel(channel=ChannelType.WHATSAPP, address="+919876543210"),
            ],
        )

    @pytest.mark.asyncio
    async def test_upsert_and_get_contact(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        retrieved = await store.get_contact("c_test_001")
        assert retrieved is not None
        assert retrieved.name == "Sharma Ji"

    @pytest.mark.asyncio
    async def test_find_contact_by_address(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        found = await store.find_contact_by_address("voice", "+919876543210")
        assert found is not None
        assert found.id == "c_test_001"

    @pytest.mark.asyncio
    async def test_find_contact_by_address_not_found(self, store):
        found = await store.find_contact_by_address("voice", "+910000000000")
        assert found is None

    @pytest.mark.asyncio
    async def test_create_and_get_conversation(self, store):
        conv = await store.create_conversation(
            contact_id="c1", channel="voice",
            process_type="order_followup",
            business_context={"order_id": "O-123"},
        )
        assert conv["id"]
        assert conv["status"] == "active"
        assert conv["business_context"]["order_id"] == "O-123"

        retrieved = await store.get_conversation(conv["id"])
        assert retrieved is not None
        assert retrieved["contact_id"] == "c1"

    @pytest.mark.asyncio
    async def test_find_active_conversation(self, store):
        await store.create_conversation(contact_id="c1", channel="voice")
        active = await store.find_active_conversation("c1")
        assert active is not None
        assert active["status"] == "active"

    @pytest.mark.asyncio
    async def test_update_conversation(self, store):
        conv = await store.create_conversation(contact_id="c1")
        await store.update_conversation(conv["id"], status="completed", outcome="order_placed")
        updated = await store.get_conversation(conv["id"])
        assert updated["status"] == "completed"
        assert updated["outcome"] == "order_placed"

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, store):
        conv = await store.create_conversation(contact_id="c1")
        msg = await store.add_message(
            conversation_id=conv["id"],
            direction="outbound", channel="voice",
            content="Namaste Sharma ji!",
            metadata={"intent": "greeting"},
        )
        assert msg["id"]
        assert msg["content"] == "Namaste Sharma ji!"

        messages = await store.get_conversation_messages(conv["id"])
        assert len(messages) == 1
        assert messages[0]["direction"] == "outbound"

    @pytest.mark.asyncio
    async def test_message_limit(self, store):
        conv = await store.create_conversation(contact_id="c1")
        for i in range(10):
            await store.add_message(conv["id"], "inbound", "voice", f"msg {i}")
        messages = await store.get_conversation_messages(conv["id"], limit=3)
        assert len(messages) == 3
        assert messages[-1]["content"] == "msg 9"

    @pytest.mark.asyncio
    async def test_create_and_get_followup(self, store):
        fup = await store.create_followup({
            "contact_id": "c1",
            "reason": "Order delivery check",
            "priority": "high",
            "process_type": "order_followup",
            "business_context": {"order_id": "O-123"},
        })
        assert fup["id"]
        assert fup["priority"] == "high"
        assert fup["status"] == "pending"

        retrieved = await store.get_followup(fup["id"])
        assert retrieved["reason"] == "Order delivery check"

    @pytest.mark.asyncio
    async def test_get_pending_followups_priority_order(self, store):
        await store.create_followup({"contact_id": "c1", "priority": "low", "reason": "low"})
        await store.create_followup({"contact_id": "c2", "priority": "critical", "reason": "critical"})
        await store.create_followup({"contact_id": "c3", "priority": "high", "reason": "high"})

        pending = await store.get_pending_followups()
        assert len(pending) == 3
        assert pending[0]["priority"] == "critical"
        assert pending[1]["priority"] == "high"
        assert pending[2]["priority"] == "low"

    @pytest.mark.asyncio
    async def test_update_followup(self, store):
        fup = await store.create_followup({"contact_id": "c1", "reason": "test"})
        await store.update_followup(fup["id"], status="completed", outcome="order_placed")
        updated = await store.get_followup(fup["id"])
        assert updated["status"] == "completed"

    @pytest.mark.asyncio
    async def test_upsert_state_binding(self, store):
        await store.upsert_state_binding(
            conversation_id="conv1",
            process_type="payment_collection",
            current_state="reminder_sent",
            state_data={"amount": 25000},
        )
        binding = await store.get_state_binding("conv1", "payment_collection")
        assert binding is not None
        assert binding["current_state"] == "reminder_sent"
        assert binding["state_data"]["amount"] == 25000

    @pytest.mark.asyncio
    async def test_upsert_state_binding_update(self, store):
        await store.upsert_state_binding("conv1", "test", current_state="s1")
        await store.upsert_state_binding("conv1", "test", current_state="s2")
        binding = await store.get_state_binding("conv1", "test")
        assert binding["current_state"] == "s2"

    @pytest.mark.asyncio
    async def test_save_voice_call(self, store):
        await store.save_voice_call({
            "id": "call_001",
            "conversation_id": "conv1",
            "contact_id": "c1",
            "direction": "outbound",
            "status": "completed",
            "duration_seconds": 120.5,
        })
        # No get_voice_call in interface, but data is stored
        assert "call_001" in store._voice_calls

    @pytest.mark.asyncio
    async def test_stats(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        await store.create_conversation(contact_id="c1")
        stats = store.stats()
        assert stats["contacts"] == 1
        assert stats["conversations"] == 1

    @pytest.mark.asyncio
    async def test_external_id_index(self, store):
        from models.schemas import Contact, ContactChannel, ChannelType
        contact = Contact(
            id="c_ext", name="External", channels=[],
        )
        contact.external_id = "EXT-001"
        # store should handle this even if Contact doesn't have external_id by default
        data = {
            "id": "c_ext", "name": "External", "role": "",
            "organization": "", "channels": [],
            "metadata": {}, "external_id": "EXT-001",
        }
        store._contacts["c_ext"] = data
        store._external_id_index["EXT-001"] = "c_ext"
        found = await store.get_contact_by_external_id("EXT-001")
        assert found is not None


# ──────────────────────────────────────────────────────────────
#  FileContextStore Tests
# ──────────────────────────────────────────────────────────────


class TestFileContextStore:
    """Test JSON file persistence."""

    @pytest_asyncio.fixture
    async def store_with_dir(self):
        from database.store_file import FileContextStore
        tmpdir = tempfile.mkdtemp()
        store = FileContextStore(data_dir=tmpdir)
        yield store, tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_write_creates_file(self, store_with_dir):
        store, tmpdir = store_with_dir
        await store.create_conversation(contact_id="c1")
        assert os.path.exists(os.path.join(tmpdir, "conversations.json"))

    @pytest.mark.asyncio
    async def test_persistence_across_reload(self, store_with_dir):
        store, tmpdir = store_with_dir
        from models.schemas import Contact, ContactChannel, ChannelType
        contact = Contact(
            id="persist_test", name="Persistence",
            channels=[ContactChannel(channel=ChannelType.VOICE, address="+911111111111")],
        )
        await store.upsert_contact(contact)
        assert os.path.exists(os.path.join(tmpdir, "contacts.json"))

        # Create new store from same directory — should load data
        from database.store_file import FileContextStore
        store2 = FileContextStore(data_dir=tmpdir)
        loaded = await store2.get_contact("persist_test")
        assert loaded is not None
        assert loaded.name == "Persistence"

    @pytest.mark.asyncio
    async def test_atomic_write(self, store_with_dir):
        store, tmpdir = store_with_dir
        await store.create_conversation(contact_id="c1")
        # No .tmp files should remain
        tmp_files = [f for f in os.listdir(tmpdir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0

    @pytest.mark.asyncio
    async def test_flush_all(self, store_with_dir):
        store, tmpdir = store_with_dir
        await store.create_conversation(contact_id="c1")
        await store.create_followup({"contact_id": "c1", "reason": "test"})
        store.flush_all()
        assert os.path.exists(os.path.join(tmpdir, "conversations.json"))
        assert os.path.exists(os.path.join(tmpdir, "followups.json"))

    @pytest.mark.asyncio
    async def test_followup_persistence(self, store_with_dir):
        store, tmpdir = store_with_dir
        fup = await store.create_followup({
            "contact_id": "c1",
            "reason": "Payment overdue",
            "priority": "high",
            "external_id": "INV-999",
        })

        from database.store_file import FileContextStore
        store2 = FileContextStore(data_dir=tmpdir)
        loaded = await store2.get_followup(fup["id"])
        assert loaded is not None
        assert loaded["reason"] == "Payment overdue"

    @pytest.mark.asyncio
    async def test_messages_persistence(self, store_with_dir):
        store, tmpdir = store_with_dir
        conv = await store.create_conversation(contact_id="c1")
        await store.add_message(conv["id"], "outbound", "voice", "Hello!")
        await store.add_message(conv["id"], "inbound", "voice", "Hi!")

        from database.store_file import FileContextStore
        store2 = FileContextStore(data_dir=tmpdir)
        msgs = await store2.get_conversation_messages(conv["id"])
        assert len(msgs) == 2


# ──────────────────────────────────────────────────────────────
#  StoreAdapter Tests
# ──────────────────────────────────────────────────────────────


class TestStoreAdapter:
    """Test that StoreAdapter correctly bridges ContextStore ↔ BaseContextStore."""

    @pytest_asyncio.fixture
    async def adapter(self):
        from database.store_memory import InMemoryContextStore
        from database.adapter import StoreAdapter
        backend = InMemoryContextStore()
        return StoreAdapter(backend)

    @pytest.fixture
    def sample_contact(self):
        from models.schemas import Contact, ContactChannel, ChannelType
        return Contact(
            id="adapt_c1", name="Adapter Test",
            channels=[ContactChannel(channel=ChannelType.VOICE, address="+919000000001")],
        )

    @pytest.fixture
    def sample_conversation(self):
        from models.schemas import Conversation, ChannelType, ConversationStatus
        return Conversation(
            id="adapt_conv1",
            contact_id="adapt_c1",
            status=ConversationStatus.ACTIVE,
            active_channel=ChannelType.VOICE,
            business_context={"order_id": "O-123"},
        )

    @pytest.fixture
    def sample_followup(self):
        from models.schemas import FollowUp, FollowUpStatus
        return FollowUp(
            id="adapt_f1",
            rule_id="rule_order_followup",
            contact_id="adapt_c1",
            reason="Delivery check",
            priority="high",
            status=FollowUpStatus.SCHEDULED,
            business_context={"order_id": "O-123"},
        )

    @pytest.mark.asyncio
    async def test_upsert_contact_delegates(self, adapter, sample_contact):
        result = await adapter.upsert_contact(sample_contact)
        assert result.name == "Adapter Test"
        # Verify backend has it
        backend_result = await adapter.backend.get_contact("adapt_c1")
        assert backend_result is not None

    @pytest.mark.asyncio
    async def test_get_contact_from_backend(self, adapter, sample_contact):
        await adapter.upsert_contact(sample_contact)
        result = await adapter.get_contact("adapt_c1")
        assert result is not None
        assert result.name == "Adapter Test"

    @pytest.mark.asyncio
    async def test_find_contact_by_address(self, adapter, sample_contact):
        from models.schemas import ChannelType
        await adapter.upsert_contact(sample_contact)
        result = await adapter.find_contact_by_address(ChannelType.VOICE, "+919000000001")
        assert result is not None
        assert result.id == "adapt_c1"

    @pytest.mark.asyncio
    async def test_create_followup_persists_to_backend(self, adapter, sample_followup):
        result = await adapter.create_followup(sample_followup)
        assert result.reason == "Delivery check"
        # Check backend
        backend_fup = await adapter.backend.get_followup(result.id)
        assert backend_fup is not None
        assert backend_fup["priority"] == "high"

    @pytest.mark.asyncio
    async def test_create_conversation_persists(self, adapter, sample_conversation):
        result = await adapter.create_conversation(sample_conversation)
        assert result.contact_id == "adapt_c1"

    @pytest.mark.asyncio
    async def test_stats(self, adapter, sample_contact):
        await adapter.upsert_contact(sample_contact)
        stats = adapter.stats()
        assert stats["backend"] == "InMemoryContextStore"
        assert "backend_stats" in stats

    @pytest.mark.asyncio
    async def test_adapter_has_context_store_interface(self, adapter):
        """Verify adapter is compatible with ContextTracker."""
        from context.tracker import ContextStore
        assert isinstance(adapter, ContextStore)


# ──────────────────────────────────────────────────────────────
#  Message Queue Backend Tests
# ──────────────────────────────────────────────────────────────


class TestMessageQueueFactory:
    """Test queue factory creates correct backend."""

    def setup_method(self):
        import job_queue.message_queue as mq
        mq._instance = None

    def teardown_method(self):
        import job_queue.message_queue as mq
        mq._instance = None

    def test_create_memory_queue(self):
        from job_queue.message_queue import create_message_queue, InMemoryMessageQueue
        q = create_message_queue({"backend": "memory"})
        assert isinstance(q, InMemoryMessageQueue)

    def test_create_memory_queue_default(self):
        from job_queue.message_queue import create_message_queue, InMemoryMessageQueue
        q = create_message_queue({})
        assert isinstance(q, InMemoryMessageQueue)

    def test_create_redis_queue(self):
        from job_queue.message_queue import create_message_queue, RedisMessageQueue
        import job_queue.message_queue as mq
        mq._instance = None
        q = create_message_queue({"backend": "redis", "redis_url": "redis://localhost:6379"})
        assert isinstance(q, RedisMessageQueue)


class TestInMemoryQueue:
    """Test in-memory queue operations."""

    @pytest_asyncio.fixture
    async def queue(self):
        from job_queue.message_queue import InMemoryMessageQueue
        q = InMemoryMessageQueue()
        await q.connect()
        yield q
        await q.close()

    @pytest.fixture
    def sample_job(self):
        from job_queue.message_queue import QueueJob
        return QueueJob(
            followup_id="f1",
            rule_id="r1",
            contact_id="c1",
            priority="high",
        )

    @pytest.mark.asyncio
    async def test_publish_and_peek(self, queue, sample_job):
        from job_queue.message_queue import Queues
        await queue.publish(Queues.DISPATCH, sample_job)
        jobs = await queue.peek(Queues.DISPATCH)
        assert len(jobs) == 1
        assert jobs[0].followup_id == "f1"

    @pytest.mark.asyncio
    async def test_queue_length(self, queue, sample_job):
        from job_queue.message_queue import Queues
        await queue.publish(Queues.DISPATCH, sample_job)
        length = await queue.queue_length(Queues.DISPATCH)
        assert length == 1

    @pytest.mark.asyncio
    async def test_nack_to_dlq(self, queue):
        from job_queue.message_queue import QueueJob, Queues
        job = QueueJob(followup_id="f1", rule_id="r1", contact_id="c1",
                       attempt=2, max_attempts=3)
        await queue.nack(Queues.DISPATCH, job)
        assert len(queue._dlq) == 1

    @pytest.mark.asyncio
    async def test_nack_retry(self, queue):
        from job_queue.message_queue import QueueJob, Queues
        job = QueueJob(followup_id="f1", rule_id="r1", contact_id="c1",
                       attempt=0, max_attempts=3)
        await queue.nack(Queues.DISPATCH, job)
        assert len(queue._delayed) == 1


# ──────────────────────────────────────────────────────────────
#  ORM Model Cross-Database Compatibility Tests
# ──────────────────────────────────────────────────────────────


class TestModelCompatibility:
    """Verify ORM models use cross-database-safe types only."""

    def test_no_jsonb_import(self):
        """Models should use JSON, not PostgreSQL-specific JSONB."""
        import inspect
        from database import models
        source = inspect.getsource(models)
        # JSONB should not be imported from pg dialect
        assert "from sqlalchemy.dialects.postgresql import JSONB" not in source
        # JSONB should not be used as a column type (comments mentioning JSONB are fine)
        assert "= mapped_column(JSONB" not in source
        assert "Column(JSONB" not in source

    def test_json_columns_used(self):
        from database.models import ContactRow
        from sqlalchemy import JSON
        channels_col = ContactRow.__table__.columns["channels"]
        # Type should be JSON (portable), not JSONB (PG-only)
        assert isinstance(channels_col.type, JSON)

    def test_all_pks_are_string(self):
        """String PKs work across all databases (no auto-increment issues)."""
        from database.models import (
            ContactRow, ConversationRow, MessageRow,
            FollowUpRow, StateBindingRow, VoiceCallRow,
        )
        for model in [ContactRow, ConversationRow, MessageRow,
                      FollowUpRow, StateBindingRow, VoiceCallRow]:
            pk_col = model.__table__.columns["id"]
            from sqlalchemy import String
            assert isinstance(pk_col.type, String), f"{model.__tablename__}.id should be String"

    def test_datetime_has_timezone(self):
        """DateTime columns should use timezone=True for cross-db consistency."""
        from database.models import ContactRow
        from sqlalchemy import DateTime
        created_col = ContactRow.__table__.columns["created_at"]
        assert isinstance(created_col.type, DateTime)
        assert created_col.type.timezone is True


# ──────────────────────────────────────────────────────────────
#  Integration: Configured Store Creation
# ──────────────────────────────────────────────────────────────


class TestConfiguredStoreCreation:
    """Test create_configured_store with different settings."""

    def setup_method(self):
        from database.store_factory import reset_store
        reset_store()

    def teardown_method(self):
        from database.store_factory import reset_store
        reset_store()

    def test_memory_backend(self):
        from config.settings import Settings, DatabaseConfig
        settings = Settings(database=DatabaseConfig(store_backend="memory"))
        with patch("database.adapter.get_settings", return_value=settings):
            with patch("database.store_factory._instance", None):
                from database.adapter import create_configured_store, StoreAdapter
                store = create_configured_store()
                assert isinstance(store, StoreAdapter)
                from database.store_memory import InMemoryContextStore
                assert isinstance(store.backend, InMemoryContextStore)

    def test_file_backend(self):
        tmpdir = tempfile.mkdtemp()
        try:
            from config.settings import Settings, DatabaseConfig
            settings = Settings(database=DatabaseConfig(
                store_backend="file", store_file_dir=tmpdir
            ))
            with patch("database.adapter.get_settings", return_value=settings):
                with patch("database.store_factory._instance", None):
                    from database.adapter import create_configured_store, StoreAdapter
                    store = create_configured_store()
                    assert isinstance(store, StoreAdapter)
                    from database.store_file import FileContextStore
                    assert isinstance(store.backend, FileContextStore)
        finally:
            from database.store_factory import reset_store
            reset_store()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_plain_store_returns_original(self):
        from database.adapter import create_plain_store
        from context.tracker import ContextStore
        store = create_plain_store()
        assert type(store) is ContextStore  # exact type, not subclass


# ──────────────────────────────────────────────────────────────
#  QueueJob Serialization Tests
# ──────────────────────────────────────────────────────────────


class TestQueueJobSerialization:
    """Test QueueJob round-trips through dict serialization."""

    def test_to_dict_and_back(self):
        from job_queue.message_queue import QueueJob
        job = QueueJob(
            followup_id="f1", rule_id="r1", contact_id="c1",
            priority="high", attempt=2, max_attempts=5,
            business_context={"order_id": "O-123", "amount": 5000},
            metadata={"retry_reason": "timeout"},
        )
        d = job.to_dict()
        # All values should be strings for Redis compatibility
        assert isinstance(d["attempt"], str)
        assert isinstance(d["business_context"], str)

        restored = QueueJob.from_dict(d)
        assert restored.followup_id == "f1"
        assert restored.attempt == 2
        assert restored.business_context["order_id"] == "O-123"

    def test_next_retry_job(self):
        from job_queue.message_queue import QueueJob
        job = QueueJob(followup_id="f1", rule_id="r1", contact_id="c1", attempt=1)
        retry = job.next_retry_job(backoff_seconds=30)
        assert retry.attempt == 2
        assert retry.job_id == job.job_id
        assert retry.scheduled_at != job.scheduled_at

    def test_is_scheduled_now(self):
        from job_queue.message_queue import QueueJob
        job = QueueJob(followup_id="f1", rule_id="r1", contact_id="c1")
        assert job.is_scheduled_now is True
