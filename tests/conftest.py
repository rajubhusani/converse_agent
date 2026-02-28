"""Shared test fixtures for ConverseAgent."""
import pytest
import asyncio
from typing import Any

from models.schemas import (
    ChannelType, Contact, ContactChannel, Conversation, ConversationStatus,
    FollowUp, FollowUpStatus, FollowUpPriority,
    StateMapDef, StateTransitionDef, TransitionTrigger, TransitionAction,
    StateBinding, RuleCondition,
)
from context.tracker import ContextStore, ContextTracker
from context.state_machine import BusinessStateMachine


@pytest.fixture
def payment_state_map() -> StateMapDef:
    """A realistic payment collection state map for testing."""
    return StateMapDef(
        process_type="payment_collection",
        states=["pending", "reminded", "acknowledged", "promised", "confirmed", "overdue", "escalated", "closed"],
        initial_state="pending",
        terminal_states=["confirmed", "closed"],
        transitions=[
            StateTransitionDef(
                from_states=["pending"],
                to_state="reminded",
                trigger=TransitionTrigger(type="action", value="outreach_sent"),
                description="First outreach sent",
            ),
            StateTransitionDef(
                from_states=["reminded"],
                to_state="acknowledged",
                trigger=TransitionTrigger(type="intent", value="acknowledged"),
                actions=[
                    TransitionAction(type="schedule_check", delay_minutes=2880,
                                     metadata={"check_trigger_type": "timeout",
                                               "check_trigger_value": "no_response_48h"}),
                ],
                description="Contact acknowledged the message",
            ),
            StateTransitionDef(
                from_states=["reminded", "acknowledged"],
                to_state="promised",
                trigger=TransitionTrigger(type="intent", value="payment_promised"),
                actions=[
                    TransitionAction(type="update_backend", backend_endpoint="update_payment_status"),
                    TransitionAction(type="schedule_check", delay_minutes=4320,
                                     metadata={"check_trigger_type": "timeout",
                                               "check_trigger_value": "promise_check"}),
                ],
                description="Contact promised to pay",
            ),
            StateTransitionDef(
                from_states=["promised"],
                to_state="overdue",
                trigger=TransitionTrigger(type="timeout", value="promise_check"),
                actions=[
                    TransitionAction(type="enqueue_followup", template="payment_overdue_reminder",
                                     priority=FollowUpPriority.HIGH),
                ],
                description="Promise period expired",
            ),
            StateTransitionDef(
                from_states=["*"],
                to_state="confirmed",
                trigger=TransitionTrigger(type="intent", values=["payment_confirmed"]),
                actions=[
                    TransitionAction(type="update_backend", backend_endpoint="mark_paid"),
                    TransitionAction(type="resolve"),
                ],
                description="Payment confirmed by contact",
            ),
            StateTransitionDef(
                from_states=["*"],
                to_state="confirmed",
                trigger=TransitionTrigger(type="event", value="payment_received"),
                actions=[
                    TransitionAction(type="enqueue_followup", template="payment_thank_you"),
                    TransitionAction(type="resolve"),
                ],
                description="Payment received via backend event",
            ),
            StateTransitionDef(
                from_states=["*"],
                to_state="escalated",
                trigger=TransitionTrigger(type="intent", value="escalation_needed"),
                actions=[
                    TransitionAction(type="escalate", metadata={"reason": "Contact requested escalation"}),
                    TransitionAction(type="notify", metadata={"target": "manager"}),
                ],
                description="Contact requested escalation",
            ),
            # Conditional transition: only escalate if days_overdue > 14
            StateTransitionDef(
                from_states=["overdue"],
                to_state="escalated",
                trigger=TransitionTrigger(
                    type="timeout", value="max_attempts_reached",
                    conditions=[RuleCondition(field="days_overdue", operator="gt", value=14)],
                ),
                actions=[TransitionAction(type="escalate")],
                description="Auto-escalate severely overdue",
            ),
        ],
    )


@pytest.fixture
def state_machine(payment_state_map) -> BusinessStateMachine:
    sm = BusinessStateMachine()
    sm.register_map(payment_state_map)
    return sm


@pytest.fixture
def context_store() -> ContextStore:
    return ContextStore()


@pytest.fixture
def context_tracker(context_store) -> ContextTracker:
    return ContextTracker(context_store)


@pytest.fixture
def sample_contact() -> Contact:
    return Contact(
        id="contact-001",
        external_id="D001",
        name="Rajesh Kumar",
        role="dealer",
        organization="Kumar Motors",
        channels=[
            ContactChannel(channel=ChannelType.WHATSAPP, address="+919876543210", preferred=True),
            ContactChannel(channel=ChannelType.EMAIL, address="rajesh@kumarmotors.com"),
        ],
    )


@pytest.fixture
def sample_conversation(sample_contact) -> Conversation:
    return Conversation(
        id="conv-001",
        contact_id=sample_contact.id,
        status=ConversationStatus.ACTIVE,
        active_channel=ChannelType.WHATSAPP,
        channels_used=[ChannelType.WHATSAPP],
        business_context={
            "invoice_number": "INV-2024-1234",
            "amount": 250000,
            "days_overdue": 7,
        },
    )


@pytest.fixture
def sample_binding(sample_contact, sample_conversation) -> StateBinding:
    return StateBinding(
        id="binding-001",
        conversation_id=sample_conversation.id,
        contact_id=sample_contact.id,
        process_type="payment_collection",
        current_state="pending",
        entity_id="INV-2024-1234",
        entity_type="invoice",
        business_data={"amount": 250000, "days_overdue": 7},
    )
