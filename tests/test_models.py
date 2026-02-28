"""Tests for data models and trigger matching logic."""
import pytest
from models.schemas import (
    TransitionTrigger, RuleCondition, StateBinding, StateTransitionRecord,
    Contact, ContactChannel, ChannelType, Conversation, ConversationStatus,
)


class TestTransitionTrigger:
    def test_exact_value_match(self):
        trigger = TransitionTrigger(type="intent", value="payment_confirmed")
        assert trigger.matches("intent", "payment_confirmed")
        assert not trigger.matches("intent", "payment_promised")
        assert not trigger.matches("event", "payment_confirmed")

    def test_values_or_match(self):
        trigger = TransitionTrigger(type="intent", values=["payment_confirmed", "payment_received"])
        assert trigger.matches("intent", "payment_confirmed")
        assert trigger.matches("intent", "payment_received")
        assert not trigger.matches("intent", "payment_promised")

    def test_type_only_match(self):
        trigger = TransitionTrigger(type="intent")
        assert trigger.matches("intent", "anything")
        assert trigger.matches("intent", "something_else")
        assert not trigger.matches("event", "anything")

    def test_type_mismatch(self):
        trigger = TransitionTrigger(type="event", value="payment_received")
        assert not trigger.matches("intent", "payment_received")


class TestStateBinding:
    def test_is_terminal_default(self):
        binding = StateBinding(
            conversation_id="c1", contact_id="ct1",
            process_type="test", current_state="active",
        )
        assert not binding.is_terminal

    def test_is_terminal_from_metadata(self):
        binding = StateBinding(
            conversation_id="c1", contact_id="ct1",
            process_type="test", current_state="done",
            metadata={"is_terminal": True},
        )
        assert binding.is_terminal

    def test_last_transition_empty(self):
        binding = StateBinding(
            conversation_id="c1", contact_id="ct1",
            process_type="test", current_state="active",
        )
        assert binding.last_transition is None

    def test_last_transition(self):
        binding = StateBinding(
            conversation_id="c1", contact_id="ct1",
            process_type="test", current_state="b",
            history=[
                StateTransitionRecord(from_state="a", to_state="b",
                                      trigger_type="intent", trigger_value="go"),
            ],
        )
        assert binding.last_transition.to_state == "b"

    def test_model_dump_roundtrip(self):
        binding = StateBinding(
            conversation_id="c1", contact_id="ct1",
            process_type="payment", current_state="pending",
            entity_id="INV-001", entity_type="invoice",
            business_data={"amount": 500},
        )
        data = binding.model_dump()
        restored = StateBinding(**data)
        assert restored.entity_id == "INV-001"
        assert restored.business_data["amount"] == 500


class TestContact:
    def test_preferred_channel(self):
        contact = Contact(
            name="Test",
            channels=[
                ContactChannel(channel=ChannelType.EMAIL, address="a@b.com"),
                ContactChannel(channel=ChannelType.WHATSAPP, address="+1234", preferred=True),
            ],
        )
        assert contact.preferred_channel.channel == ChannelType.WHATSAPP

    def test_preferred_channel_fallback(self):
        contact = Contact(
            name="Test",
            channels=[
                ContactChannel(channel=ChannelType.EMAIL, address="a@b.com"),
            ],
        )
        assert contact.preferred_channel.channel == ChannelType.EMAIL

    def test_get_channel(self):
        contact = Contact(
            name="Test",
            channels=[
                ContactChannel(channel=ChannelType.EMAIL, address="a@b.com"),
                ContactChannel(channel=ChannelType.WHATSAPP, address="+1234"),
            ],
        )
        assert contact.get_channel(ChannelType.WHATSAPP).address == "+1234"
        assert contact.get_channel(ChannelType.VOICE) is None


class TestConversation:
    def test_default_status(self):
        conv = Conversation(contact_id="c1")
        assert conv.status == ConversationStatus.PENDING

    def test_state_bindings_list(self):
        conv = Conversation(contact_id="c1", state_bindings=["b1", "b2"])
        assert len(conv.state_bindings) == 2
