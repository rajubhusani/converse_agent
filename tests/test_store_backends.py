"""
Tests for all context store backends.

Covers:
  - InMemoryContextStore
  - FileContextStore (JSON file persistence)
  - SqlContextStore (via SQLite for test portability)
  - Store factory
  - Queue factory (memory vs redis selection)
"""
import asyncio
import json
import os
import shutil
import tempfile
import pytest

from models.schemas import Contact, ContactChannel, ChannelType


# ──────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_contact():
    return Contact(
        id="c_test_001",
        name="Sharma Ji",
        role="dealer",
        organization="Sharma Traders",
        channels=[
            ContactChannel(channel=ChannelType.VOICE, address="+919876543210", preferred=True, verified=True),
            ContactChannel(channel=ChannelType.WHATSAPP, address="+919876543210", preferred=False, verified=True),
        ],
    )


@pytest.fixture
def sample_followup_data():
    return {
        "contact_id": "c_test_001",
        "external_id": "ext_fu_001",
        "reason": "Payment overdue",
        "priority": "high",
        "process_type": "payment_collection",
        "entity_type": "invoice",
        "entity_id": "INV-301",
        "business_context": {"amount": 250000, "days_overdue": 15},
        "channel_priority": ["voice", "whatsapp"],
    }


# ──────────────────────────────────────────────────────────────
#  InMemoryContextStore
# ──────────────────────────────────────────────────────────────

class TestInMemoryContextStore:
    @pytest.fixture
    def store(self):
        from database.store_memory import InMemoryContextStore
        return InMemoryContextStore()

    @pytest.mark.asyncio
    async def test_upsert_and_get_contact(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        result = await store.get_contact("c_test_001")
        assert result is not None
        assert result.name == "Sharma Ji"
        assert result.organization == "Sharma Traders"
        assert len(result.channels) == 2

    @pytest.mark.asyncio
    async def test_find_contact_by_address(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        result = await store.find_contact_by_address("voice", "+919876543210")
        assert result is not None
        assert result.id == "c_test_001"

    @pytest.mark.asyncio
    async def test_find_contact_by_address_not_found(self, store):
        result = await store.find_contact_by_address("voice", "+910000000000")
        assert result is None

    @pytest.mark.asyncio
    async def test_contact_external_id(self, store, sample_contact):
        sample_contact.external_id = "EXT_001"
        await store.upsert_contact(sample_contact)
        result = await store.get_contact_by_external_id("EXT_001")
        assert result is not None
        assert result.id == "c_test_001"

    @pytest.mark.asyncio
    async def test_create_conversation(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        conv = await store.create_conversation(
            contact_id="c_test_001",
            channel="voice",
            process_type="payment_collection",
        )
        assert conv["id"]
        assert conv["status"] == "active"
        assert conv["contact_id"] == "c_test_001"

    @pytest.mark.asyncio
    async def test_find_active_conversation(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        await store.create_conversation(contact_id="c_test_001", channel="voice")
        result = await store.find_active_conversation("c_test_001")
        assert result is not None
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_update_conversation(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        conv = await store.create_conversation(contact_id="c_test_001")
        await store.update_conversation(conv["id"], status="resolved", outcome="paid")
        updated = await store.get_conversation(conv["id"])
        assert updated["status"] == "resolved"
        assert updated["outcome"] == "paid"

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        conv = await store.create_conversation(contact_id="c_test_001")
        await store.add_message(conv["id"], "outbound", "voice", "Hello Sharma ji")
        await store.add_message(conv["id"], "inbound", "voice", "Haan bolo")
        msgs = await store.get_conversation_messages(conv["id"])
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Hello Sharma ji"
        assert msgs[1]["direction"] == "inbound"

    @pytest.mark.asyncio
    async def test_message_limit(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        conv = await store.create_conversation(contact_id="c_test_001")
        for i in range(20):
            await store.add_message(conv["id"], "outbound", "voice", f"msg {i}")
        msgs = await store.get_conversation_messages(conv["id"], limit=5)
        assert len(msgs) == 5
        assert msgs[0]["content"] == "msg 15"

    @pytest.mark.asyncio
    async def test_create_followup(self, store, sample_followup_data):
        fup = await store.create_followup(sample_followup_data)
        assert fup["id"]
        assert fup["priority"] == "high"
        assert fup["status"] == "pending"

    @pytest.mark.asyncio
    async def test_followup_by_external_id(self, store, sample_followup_data):
        fup = await store.create_followup(sample_followup_data)
        result = await store.get_followup_by_external_id("ext_fu_001")
        assert result is not None
        assert result["id"] == fup["id"]

    @pytest.mark.asyncio
    async def test_pending_followups_priority_order(self, store):
        await store.create_followup({"contact_id": "c1", "priority": "low"})
        await store.create_followup({"contact_id": "c2", "priority": "critical"})
        await store.create_followup({"contact_id": "c3", "priority": "medium"})
        pending = await store.get_pending_followups()
        priorities = [f["priority"] for f in pending]
        assert priorities == ["critical", "medium", "low"]

    @pytest.mark.asyncio
    async def test_update_followup(self, store, sample_followup_data):
        fup = await store.create_followup(sample_followup_data)
        await store.update_followup(fup["id"], status="resolved", outcome="paid_in_full")
        updated = await store.get_followup(fup["id"])
        assert updated["status"] == "resolved"
        assert updated["outcome"] == "paid_in_full"

    @pytest.mark.asyncio
    async def test_state_binding_upsert(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        conv = await store.create_conversation(contact_id="c_test_001")
        await store.upsert_state_binding(
            conv["id"], "payment_collection",
            current_state="awaiting_payment",
            entity_type="invoice", entity_id="INV-301",
        )
        binding = await store.get_state_binding(conv["id"], "payment_collection")
        assert binding is not None
        assert binding["current_state"] == "awaiting_payment"

    @pytest.mark.asyncio
    async def test_state_binding_update(self, store, sample_contact):
        await store.upsert_contact(sample_contact)
        conv = await store.create_conversation(contact_id="c_test_001")
        await store.upsert_state_binding(
            conv["id"], "payment_collection",
            current_state="initial",
        )
        await store.upsert_state_binding(
            conv["id"], "payment_collection",
            current_state="promised_to_pay",
        )
        binding = await store.get_state_binding(conv["id"], "payment_collection")
        assert binding["current_state"] == "promised_to_pay"

    @pytest.mark.asyncio
    async def test_voice_call(self, store):
        await store.save_voice_call({
            "id": "call_001",
            "conversation_id": "conv_001",
            "contact_id": "c_001",
            "direction": "outbound",
            "status": "completed",
            "duration_seconds": 180.5,
        })
        # save_voice_call doesn't have a getter in the interface
        # but we can verify through internal state
        assert "call_001" in store._voice_calls

    @pytest.mark.asyncio
    async def test_stats(self, store, sample_contact, sample_followup_data):
        await store.upsert_contact(sample_contact)
        await store.create_conversation(contact_id="c_test_001")
        await store.create_followup(sample_followup_data)
        s = store.stats()
        assert s["contacts"] == 1
        assert s["conversations"] == 1
        assert s["followups"] == 1


# ──────────────────────────────────────────────────────────────
#  FileContextStore
# ──────────────────────────────────────────────────────────────

class TestFileContextStore:
    @pytest.fixture
    def data_dir(self):
        d = tempfile.mkdtemp(prefix="converse_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def store(self, data_dir):
        from database.store_file import FileContextStore
        return FileContextStore(data_dir=data_dir, flush_interval_s=0)

    @pytest.mark.asyncio
    async def test_persistence_round_trip(self, data_dir, sample_contact):
        from database.store_file import FileContextStore

        # Write
        store1 = FileContextStore(data_dir=data_dir, flush_interval_s=0)
        await store1.upsert_contact(sample_contact)
        conv = await store1.create_conversation(contact_id="c_test_001", channel="voice")
        await store1.add_message(conv["id"], "outbound", "voice", "Hello")

        # Verify files exist
        assert os.path.exists(os.path.join(data_dir, "contacts.json"))
        assert os.path.exists(os.path.join(data_dir, "conversations.json"))
        assert os.path.exists(os.path.join(data_dir, "messages.json"))

        # Read with fresh instance
        store2 = FileContextStore(data_dir=data_dir, flush_interval_s=0)
        contact = await store2.get_contact("c_test_001")
        assert contact is not None
        assert contact.name == "Sharma Ji"

        loaded_conv = await store2.get_conversation(conv["id"])
        assert loaded_conv is not None
        assert loaded_conv["active_channel"] == "voice"

        msgs = await store2.get_conversation_messages(conv["id"])
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_address_index_rebuilt(self, data_dir, sample_contact):
        from database.store_file import FileContextStore

        store1 = FileContextStore(data_dir=data_dir, flush_interval_s=0)
        await store1.upsert_contact(sample_contact)

        store2 = FileContextStore(data_dir=data_dir, flush_interval_s=0)
        found = await store2.find_contact_by_address("voice", "+919876543210")
        assert found is not None
        assert found.id == "c_test_001"

    @pytest.mark.asyncio
    async def test_followup_persistence(self, data_dir, sample_followup_data):
        from database.store_file import FileContextStore

        store1 = FileContextStore(data_dir=data_dir, flush_interval_s=0)
        fup = await store1.create_followup(sample_followup_data)

        store2 = FileContextStore(data_dir=data_dir, flush_interval_s=0)
        loaded = await store2.get_followup(fup["id"])
        assert loaded is not None
        assert loaded["priority"] == "high"

        by_ext = await store2.get_followup_by_external_id("ext_fu_001")
        assert by_ext is not None
        assert by_ext["id"] == fup["id"]

    @pytest.mark.asyncio
    async def test_corrupt_file_handled(self, data_dir):
        from database.store_file import FileContextStore

        # Write corrupt JSON
        with open(os.path.join(data_dir, "contacts.json"), "w") as f:
            f.write("{invalid json!!!")

        # Should not crash — logs warning and starts empty
        store = FileContextStore(data_dir=data_dir, flush_interval_s=0)
        result = await store.get_contact("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_flush_all(self, store, data_dir, sample_contact, sample_followup_data):
        await store.upsert_contact(sample_contact)
        await store.create_followup(sample_followup_data)
        store.flush_all()
        # All JSON files should exist
        for name in ["contacts", "followups"]:
            path = os.path.join(data_dir, f"{name}.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)


# ──────────────────────────────────────────────────────────────
#  Store Factory
# ──────────────────────────────────────────────────────────────

class TestStoreFactory:
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

    def test_create_file_store(self):
        from database.store_factory import create_store, reset_store
        from database.store_file import FileContextStore
        d = tempfile.mkdtemp(prefix="converse_factory_")
        try:
            store = create_store({"store_backend": "file", "store_file_dir": d})
            assert isinstance(store, FileContextStore)
        finally:
            reset_store()
            shutil.rmtree(d, ignore_errors=True)

    def test_create_sql_store(self):
        from database.store_factory import create_store
        from database.store import SqlContextStore
        store = create_store({"store_backend": "sql"})
        assert isinstance(store, SqlContextStore)

    def test_default_is_memory(self):
        from database.store_factory import create_store
        from database.store_memory import InMemoryContextStore
        store = create_store({})
        assert isinstance(store, InMemoryContextStore)

    def test_singleton(self):
        from database.store_factory import create_store, get_store
        s1 = create_store({"store_backend": "memory"})
        s2 = get_store()
        assert s1 is s2


# ──────────────────────────────────────────────────────────────
#  Database Session — URL Translation
# ──────────────────────────────────────────────────────────────

class TestSessionUrlTranslation:
    def test_postgresql_url(self):
        from database.session import _to_async_url
        assert _to_async_url("postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"

    def test_postgres_url(self):
        from database.session import _to_async_url
        assert _to_async_url("postgres://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"

    def test_mysql_url(self):
        from database.session import _to_async_url
        assert _to_async_url("mysql://u:p@h/db") == "mysql+aiomysql://u:p@h/db"

    def test_mysql_pymysql_url(self):
        from database.session import _to_async_url
        assert _to_async_url("mysql+pymysql://u:p@h/db") == "mysql+aiomysql://u:p@h/db"

    def test_sqlite_url(self):
        from database.session import _to_async_url
        assert _to_async_url("sqlite:///./test.db") == "sqlite+aiosqlite:///./test.db"

    def test_already_async_url(self):
        from database.session import _to_async_url
        url = "postgresql+asyncpg://u:p@h/db"
        assert _to_async_url(url) == url

    def test_mysql_aiomysql_url(self):
        from database.session import _to_async_url
        url = "mysql+aiomysql://u:p@h/db"
        assert _to_async_url(url) == url


# ──────────────────────────────────────────────────────────────
#  Queue Factory
# ──────────────────────────────────────────────────────────────

class TestQueueFactory:
    def test_memory_queue_default(self):
        from job_queue.message_queue import create_message_queue, InMemoryMessageQueue
        # Reset singleton
        import job_queue.message_queue as m
        m._instance = None
        q = create_message_queue({"backend": "memory"})
        assert isinstance(q, InMemoryMessageQueue)
        m._instance = None

    def test_redis_queue_selected(self):
        from job_queue.message_queue import create_message_queue, RedisMessageQueue
        import job_queue.message_queue as m
        m._instance = None
        q = create_message_queue({"backend": "redis", "redis_url": "redis://localhost:6379"})
        assert isinstance(q, RedisMessageQueue)
        m._instance = None


# ──────────────────────────────────────────────────────────────
#  Cross-DB Models Portability
# ──────────────────────────────────────────────────────────────

class TestModelsPortability:
    """Verify models use portable JSON type instead of JSONB."""

    def test_contact_channels_is_json(self):
        from sqlalchemy import JSON
        from database.models import ContactRow
        col = ContactRow.__table__.columns["channels"]
        assert isinstance(col.type, JSON)

    def test_contact_metadata_is_json(self):
        from sqlalchemy import JSON
        from database.models import ContactRow
        col = ContactRow.__table__.columns["metadata"]
        assert isinstance(col.type, JSON)

    def test_conversation_business_context_is_json(self):
        from sqlalchemy import JSON
        from database.models import ConversationRow
        col = ConversationRow.__table__.columns["business_context"]
        assert isinstance(col.type, JSON)

    def test_followup_channel_priority_is_json(self):
        from sqlalchemy import JSON
        from database.models import FollowUpRow
        col = FollowUpRow.__table__.columns["channel_priority"]
        assert isinstance(col.type, JSON)

    def test_state_binding_state_data_is_json(self):
        from sqlalchemy import JSON
        from database.models import StateBindingRow
        col = StateBindingRow.__table__.columns["state_data"]
        assert isinstance(col.type, JSON)

    def test_voice_call_transcript_is_json(self):
        from sqlalchemy import JSON
        from database.models import VoiceCallRow
        col = VoiceCallRow.__table__.columns["transcript"]
        assert isinstance(col.type, JSON)

    def test_no_jsonb_anywhere(self):
        """Ensure JSONB (PostgreSQL-specific) is not used in any model."""
        from database.models import Base
        for table in Base.metadata.tables.values():
            for col in table.columns:
                col_type_name = type(col.type).__name__
                assert col_type_name != "JSONB", (
                    f"Column {table.name}.{col.name} uses JSONB — "
                    f"use JSON for cross-DB compatibility"
                )

    def test_all_tables_defined(self):
        from database.models import Base
        expected = {"contacts", "conversations", "messages", "followups",
                    "state_bindings", "voice_calls"}
        assert set(Base.metadata.tables.keys()) == expected


# ──────────────────────────────────────────────────────────────
#  Config — Database Settings
# ──────────────────────────────────────────────────────────────

class TestDatabaseConfig:
    def test_default_store_backend(self):
        from config.settings import DatabaseConfig
        cfg = DatabaseConfig()
        assert cfg.store_backend == "memory"

    def test_default_url_is_sqlite(self):
        from config.settings import DatabaseConfig
        cfg = DatabaseConfig()
        assert "sqlite" in cfg.url

    def test_store_file_dir_default(self):
        from config.settings import DatabaseConfig
        cfg = DatabaseConfig()
        assert cfg.store_file_dir == "./data"

    def test_backwards_compatible_alias(self):
        from database.store import PostgresContextStore, SqlContextStore
        assert PostgresContextStore is SqlContextStore
