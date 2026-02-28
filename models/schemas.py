"""
Core data models for the ConverseAgent system.
These are the universal types shared across all modules.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────────

class ChannelType(str, Enum):
    CHAT = "chat"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    VOICE = "voice"
    SMS = "sms"


class MessageDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class ConversationStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    WAITING_RESPONSE = "waiting_response"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class FollowUpStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    AWAITING_REPLY = "awaiting_reply"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FollowUpPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# ──────────────────────────────────────────────────────────────
#  Contact — the person/entity being communicated with
# ──────────────────────────────────────────────────────────────

class ContactChannel(BaseModel):
    """A single reachable channel for a contact."""
    channel: ChannelType
    address: str                              # phone, email, user_id, etc.
    preferred: bool = False
    verified: bool = True


class Contact(BaseModel):
    """A business contact (dealer, vendor, customer, employee, …)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    external_id: str = ""                     # ID in the backend system
    name: str
    role: str = ""                            # e.g. "dealer", "vendor", "customer"
    organization: str = ""
    channels: list[ContactChannel] = []
    metadata: dict[str, Any] = {}             # arbitrary backend data
    tags: list[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def preferred_channel(self) -> Optional[ContactChannel]:
        return next((c for c in self.channels if c.preferred), self.channels[0] if self.channels else None)

    def get_channel(self, channel_type: ChannelType) -> Optional[ContactChannel]:
        return next((c for c in self.channels if c.channel == channel_type), None)


# ──────────────────────────────────────────────────────────────
#  Message — a single message in a conversation
# ──────────────────────────────────────────────────────────────

class Attachment(BaseModel):
    filename: str
    content_type: str
    url: str = ""
    size_bytes: int = 0


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    channel: ChannelType
    direction: MessageDirection
    sender_id: str                            # contact_id or "agent"
    content: str
    content_type: str = "text"                # text | html | template | media
    attachments: list[Attachment] = []
    metadata: dict[str, Any] = {}             # channel-specific metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    delivered: bool = False
    read: bool = False


# ──────────────────────────────────────────────────────────────
#  Business State — tracks where a business process stands
# ──────────────────────────────────────────────────────────────

class StateTransitionRecord(BaseModel):
    """Immutable log of a single state transition."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_state: str
    to_state: str
    trigger_type: str               # intent | event | timeout | manual | action
    trigger_value: str              # the specific intent/event name
    actions_fired: list[str] = []   # action IDs that executed
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = {}


class StateBinding(BaseModel):
    """
    Binds a conversation to ONE business process and tracks its state.

    A single conversation can have MULTIPLE StateBindings — e.g. one for
    'payment_collection' and another for 'delivery_tracking' on the same order.

    This is the bridge between conversation context and business reality.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    contact_id: str
    process_type: str                           # e.g. "payment_collection", "order_fulfillment"
    current_state: str                          # e.g. "reminded", "promised", "confirmed"
    entity_id: str = ""                         # external business entity ID (invoice, order, ticket)
    entity_type: str = ""                       # "invoice", "purchase_order", "support_ticket"
    business_data: dict[str, Any] = {}          # snapshot of relevant business fields
    history: list[StateTransitionRecord] = []   # full transition log
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: dict[str, Any] = {}

    @property
    def is_terminal(self) -> bool:
        """Check if current state is a terminal state (defined in state map config)."""
        return self.metadata.get("is_terminal", False)

    @property
    def last_transition(self) -> Optional[StateTransitionRecord]:
        return self.history[-1] if self.history else None


# ──────────────────────────────────────────────────────────────
#  Rule Condition — shared by rules engine AND state machine
#  (defined early because TransitionTrigger references it)
# ──────────────────────────────────────────────────────────────

class RuleCondition(BaseModel):
    field: str
    operator: str           # eq | neq | gt | gte | lt | lte | in | contains | regex
    value: Any


# ──────────────────────────────────────────────────────────────
#  State Map Configuration — defines a business process FSM
# ──────────────────────────────────────────────────────────────

class TransitionAction(BaseModel):
    """An action to execute when a state transition fires."""
    type: str                                   # enqueue_followup | update_backend | resolve | escalate | notify | schedule_check
    template: str = ""
    channel_priority: list[ChannelType] = []
    delay_minutes: int = 0
    backend_endpoint: str = ""
    backend_payload: dict[str, Any] = {}
    priority: FollowUpPriority = FollowUpPriority.MEDIUM
    metadata: dict[str, Any] = {}


class TransitionTrigger(BaseModel):
    """What causes a state transition."""
    type: str                                   # intent | event | timeout | manual | action
    value: str = ""                             # specific intent name, event name, etc.
    values: list[str] = []                      # OR-match — any of these trigger it
    conditions: list[RuleCondition] = []        # additional conditions on business_data

    def matches(self, trigger_type: str, trigger_value: str) -> bool:
        if self.type != trigger_type:
            return False
        if self.values:
            return trigger_value in self.values
        if self.value:
            return trigger_value == self.value
        return True                             # type-only match (e.g. any intent)


class StateTransitionDef(BaseModel):
    """One possible transition in the state map."""
    from_states: list[str]                      # states this transition applies from
    to_state: str                               # destination state
    trigger: TransitionTrigger
    actions: list[TransitionAction] = []
    description: str = ""


class StateMapDef(BaseModel):
    """
    Complete definition of a business process as a state machine.

    Example:
      process_type: payment_collection
      states: [pending, reminded, acknowledged, promised, confirmed, overdue, escalated, closed]
      initial_state: pending
      terminal_states: [confirmed, closed]
      transitions:
        - from_states: [pending]
          to_state: reminded
          trigger: { type: action, value: send_reminder }
        - from_states: [reminded, acknowledged]
          to_state: confirmed
          trigger: { type: intent, value: payment_confirmed }
          actions:
            - { type: update_backend, backend_endpoint: mark_paid }
            - { type: resolve }
    """
    process_type: str
    states: list[str]
    initial_state: str
    terminal_states: list[str] = []
    transitions: list[StateTransitionDef] = []
    description: str = ""
    metadata: dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────
#  Conversation — tracks a thread across channels
# ──────────────────────────────────────────────────────────────

class Conversation(BaseModel):
    """
    A conversation thread. Key design: one conversation can span
    multiple channels (e.g. started on WhatsApp, continued on email).
    The business_context links it to the backend system.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contact_id: str
    followup_id: Optional[str] = None         # links to FollowUp
    status: ConversationStatus = ConversationStatus.PENDING
    channels_used: list[ChannelType] = []
    active_channel: Optional[ChannelType] = None
    business_context: dict[str, Any] = {}     # data from backend system
    summary: str = ""                         # LLM-generated summary
    messages: list[Message] = []
    state_bindings: list[str] = []            # StateBinding IDs bound to this conversation
    attempt_count: int = 0
    max_attempts: int = 5
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    metadata: dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────
#  FollowUp — a scheduled/triggered follow-up action
# ──────────────────────────────────────────────────────────────

class FollowUp(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str                              # which rule triggered this
    contact_id: str
    conversation_id: Optional[str] = None
    status: FollowUpStatus = FollowUpStatus.SCHEDULED
    priority: FollowUpPriority = FollowUpPriority.MEDIUM
    business_context: dict[str, Any] = {}
    reason: str = ""                          # human-readable reason
    channel_priority: list[ChannelType] = []  # ordered preference
    current_channel: Optional[ChannelType] = None
    attempt_count: int = 0
    max_attempts: int = 3
    next_attempt_at: Optional[datetime] = None
    scheduled_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    outcome: str = ""                         # result description
    metadata: dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────
#  Rule — defines when and how to follow up
# ──────────────────────────────────────────────────────────────

class RuleTrigger(BaseModel):
    type: str               # schedule | event | condition
    cron: Optional[str] = None
    event_name: Optional[str] = None


class RuleAction(BaseModel):
    type: str               # start_conversation | send_notification | escalate | update_backend
    channel_priority: list[ChannelType] = []
    template: str = ""
    delay_minutes: int = 0
    escalation: Optional[dict[str, Any]] = None
    process_type: str = ""          # state machine to bind (e.g. "payment_collection")
    entity_type: str = ""           # business entity type (e.g. "invoice")
    entity_id_field: str = ""       # field name in data that holds the entity ID (e.g. "invoice_number")
    initial_state: str = ""         # override initial state (default: state map's initial_state)


class Rule(BaseModel):
    id: str
    name: str
    description: str = ""
    enabled: bool = True
    trigger: RuleTrigger
    conditions: list[RuleCondition] = []
    actions: list[RuleAction] = []
    metadata: dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────
#  Events — internal event bus messages
# ──────────────────────────────────────────────────────────────

class AgentEvent(BaseModel):
    """Internal event passed through the event bus."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str                           # message_received | followup_due | backend_event | ...
    source: str                               # channel name or "backend" or "scheduler"
    payload: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
