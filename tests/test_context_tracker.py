"""Tests for ContextTracker — cross-channel state management."""
import pytest
from models.schemas import (
    ChannelType, Contact, ContactChannel, Conversation, ConversationStatus,
    FollowUp, FollowUpStatus, StateBinding,
)
from context.tracker import ContextStore, ContextTracker, ContextChange


@pytest.mark.asyncio
class TestContextStore:
    async def test_upsert_and_get_contact(self, context_store, sample_contact):
        await context_store.upsert_contact(sample_contact)
        retrieved = await context_store.get_contact(sample_contact.id)
        assert retrieved is not None
        assert retrieved.name == "Rajesh Kumar"

    async def test_find_contact_by_address(self, context_store, sample_contact):
        await context_store.upsert_contact(sample_contact)
        found = await context_store.find_contact_by_address(
            ChannelType.WHATSAPP, "+919876543210",
        )
        assert found is not None
        assert found.id == sample_contact.id

    async def test_find_contact_not_found(self, context_store):
        found = await context_store.find_contact_by_address(
            ChannelType.EMAIL, "nobody@nowhere.com",
        )
        assert found is None

    async def test_create_and_get_conversation(self, context_store, sample_conversation):
        await context_store.create_conversation(sample_conversation)
        retrieved = await context_store.get_conversation(sample_conversation.id)
        assert retrieved is not None
        assert retrieved.contact_id == sample_conversation.contact_id

    async def test_list_conversations_filter_status(self, context_store, sample_conversation):
        await context_store.create_conversation(sample_conversation)
        active = await context_store.list_conversations(status=ConversationStatus.ACTIVE)
        assert len(active) == 1
        resolved = await context_store.list_conversations(status=ConversationStatus.RESOLVED)
        assert len(resolved) == 0

    async def test_state_binding_crud(self, context_store, sample_binding):
        await context_store.create_state_binding(sample_binding)
        retrieved = await context_store.get_state_binding(sample_binding.id)
        assert retrieved is not None
        assert retrieved.process_type == "payment_collection"

    async def test_bindings_for_conversation(self, context_store, sample_binding):
        await context_store.create_state_binding(sample_binding)
        bindings = await context_store.get_bindings_for_conversation(
            sample_binding.conversation_id,
        )
        assert len(bindings) == 1

    async def test_bindings_for_entity(self, context_store, sample_binding):
        await context_store.create_state_binding(sample_binding)
        bindings = await context_store.get_bindings_for_entity("invoice", "INV-2024-1234")
        assert len(bindings) == 1

    async def test_active_bindings_for_contact(self, context_store, sample_binding):
        await context_store.create_state_binding(sample_binding)
        active = await context_store.get_active_bindings_for_contact(sample_binding.contact_id)
        assert len(active) == 1

        # Mark terminal
        sample_binding.metadata["is_terminal"] = True
        await context_store.update_state_binding(sample_binding)
        active = await context_store.get_active_bindings_for_contact(sample_binding.contact_id)
        assert len(active) == 0

    async def test_list_state_bindings_filter(self, context_store, sample_binding):
        await context_store.create_state_binding(sample_binding)
        found = await context_store.list_state_bindings(process_type="payment_collection")
        assert len(found) == 1
        not_found = await context_store.list_state_bindings(process_type="nonexistent")
        assert len(not_found) == 0


@pytest.mark.asyncio
class TestContextTracker:
    async def test_route_inbound_creates_contact(self, context_tracker):
        contact, conv, msg = await context_tracker.route_inbound_message(
            channel=ChannelType.WHATSAPP,
            sender_address="+919999999999",
            content="Hello",
        )
        assert contact is not None
        assert conv is not None
        assert msg.content == "Hello"
        assert conv.active_channel == ChannelType.WHATSAPP

    async def test_route_inbound_reuses_contact(self, context_tracker, sample_contact):
        await context_tracker.store.upsert_contact(sample_contact)
        contact, _, _ = await context_tracker.route_inbound_message(
            channel=ChannelType.WHATSAPP,
            sender_address="+919876543210",
            content="Hi again",
        )
        assert contact.id == sample_contact.id

    async def test_cross_channel_continuity(self, context_tracker, sample_contact):
        await context_tracker.store.upsert_contact(sample_contact)

        # First message on WhatsApp
        _, conv1, _ = await context_tracker.route_inbound_message(
            ChannelType.WHATSAPP, "+919876543210", "Hello from WhatsApp",
        )

        # Second message on Email — should join same conversation
        _, conv2, _ = await context_tracker.route_inbound_message(
            ChannelType.EMAIL, "rajesh@kumarmotors.com", "Hello from Email",
        )
        # They share the same contact, so should be same conversation
        assert conv1.id == conv2.id
        assert ChannelType.WHATSAPP in conv2.channels_used
        assert ChannelType.EMAIL in conv2.channels_used

    async def test_record_outbound_message(self, context_tracker, sample_contact, sample_conversation):
        await context_tracker.store.upsert_contact(sample_contact)
        await context_tracker.store.create_conversation(sample_conversation)

        msg = await context_tracker.record_outbound_message(
            sample_conversation.id, ChannelType.WHATSAPP, "Follow-up message",
        )
        assert msg.direction.value == "outbound"
        conv = await context_tracker.store.get_conversation(sample_conversation.id)
        assert conv.status == ConversationStatus.WAITING_RESPONSE
        assert conv.attempt_count == 1

    async def test_bind_state(self, context_tracker, sample_binding, sample_conversation):
        await context_tracker.store.create_conversation(sample_conversation)
        binding = await context_tracker.bind_state(sample_binding)
        assert binding.id in (await context_tracker.store.get_conversation(sample_conversation.id)).state_bindings

    async def test_build_context_change(self, context_tracker, sample_contact, sample_conversation):
        await context_tracker.store.upsert_contact(sample_contact)
        await context_tracker.store.create_conversation(sample_conversation)

        change = await context_tracker.build_context_change(
            conversation_id=sample_conversation.id,
            trigger_type="intent",
            trigger_value="payment_promised",
            extra_data={"extracted_amount": 50000},
        )
        assert change.trigger_type == "intent"
        assert change.trigger_value == "payment_promised"
        assert change.context_data["contact_name"] == "Rajesh Kumar"
        assert change.context_data["extracted_amount"] == 50000
        assert change.context_data["invoice_number"] == "INV-2024-1234"

    async def test_get_full_context(self, context_tracker, sample_contact, sample_conversation, sample_binding):
        await context_tracker.store.upsert_contact(sample_contact)
        await context_tracker.store.create_conversation(sample_conversation)
        # Use bind_state which handles both the store and the conversation link
        await context_tracker.bind_state(sample_binding)

        ctx = await context_tracker.get_full_context(sample_conversation.id)
        assert ctx["conversation_id"] == sample_conversation.id
        assert ctx["contact"]["name"] == "Rajesh Kumar"
        assert len(ctx["state_bindings"]) == 1
        assert ctx["state_bindings"][0]["process_type"] == "payment_collection"

    async def test_resolve_conversation(self, context_tracker, sample_conversation):
        await context_tracker.store.create_conversation(sample_conversation)
        conv = await context_tracker.resolve_conversation(sample_conversation.id, "Payment received")
        assert conv.status == ConversationStatus.RESOLVED
        assert conv.resolved_at is not None

    async def test_escalate_conversation(self, context_tracker, sample_conversation):
        await context_tracker.store.create_conversation(sample_conversation)
        conv = await context_tracker.escalate_conversation(sample_conversation.id, "Customer angry")
        assert conv.status == ConversationStatus.ESCALATED
        assert conv.metadata["escalation_reason"] == "Customer angry"

    async def test_get_stats(self, context_tracker, sample_contact, sample_conversation, sample_binding):
        await context_tracker.store.upsert_contact(sample_contact)
        await context_tracker.store.create_conversation(sample_conversation)
        await context_tracker.store.create_state_binding(sample_binding)

        stats = await context_tracker.get_stats()
        assert stats["total_conversations"] == 1
        assert stats["active_conversations"] == 1
        assert stats["total_contacts"] == 1
        assert stats["total_state_bindings"] == 1
        assert "payment_collection" in stats["state_distribution"]
