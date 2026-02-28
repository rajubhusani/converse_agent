"""
Context Tracker — Maintains unified conversation state across ALL channels.

This is the heart of cross-channel continuity. When a contact is reached on
WhatsApp, then replies by email, the context tracker ensures the agent has
full history and business context regardless of channel.

Context changes (new messages, intent detection, backend events) flow through
the state machine to drive business state transitions and trigger follow-up
actions.

Architecture:
  Context Change → ContextTracker.build_context_change()
    → Orchestrator feeds to BusinessStateMachine.apply_trigger_to_all(bindings)
    → TransitionResult[actions]
    → Orchestrator executes actions (enqueue, resolve, update_backend, escalate)
"""
from __future__ import annotations

import structlog
from datetime import datetime
from typing import Any, Optional

from models.schemas import (
    ChannelType, Contact, ContactChannel, Conversation, ConversationStatus,
    FollowUp, FollowUpStatus, Message, MessageDirection,
    StateBinding, StateTransitionRecord,
)

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────
#  Context Change — the input event model
# ──────────────────────────────────────────────────────────────

class ContextChange:
    """
    Represents any change in conversation context that could
    trigger a business state transition.

    trigger_type:
      "intent"   — LLM-detected intent from a contact message
      "event"    — Backend system event (order_shipped, payment_received)
      "timeout"  — Timer expired (no response after N hours)
      "manual"   — Operator/dashboard-driven state change
      "action"   — System action (first outreach sent, reminder sent)

    trigger_value: the specific intent/event name
    context_data:  merged business + conversation data for condition evaluation
    """

    def __init__(
        self,
        trigger_type: str,
        trigger_value: str,
        conversation_id: str = "",
        contact_id: str = "",
        context_data: dict[str, Any] = None,
        source_channel: ChannelType = None,
    ):
        self.trigger_type = trigger_type
        self.trigger_value = trigger_value
        self.conversation_id = conversation_id
        self.contact_id = contact_id
        self.context_data = context_data or {}
        self.source_channel = source_channel

    def __repr__(self):
        return f"<ContextChange {self.trigger_type}:{self.trigger_value} conv={self.conversation_id[:8]}>"


# ──────────────────────────────────────────────────────────────
#  Context Store — pluggable storage backend
# ──────────────────────────────────────────────────────────────

class ContextStore:
    """
    Pluggable storage backend for the context tracker.
    Default implementation uses in-memory dicts.
    Production: swap with Redis or database-backed store.
    """

    def __init__(self):
        self._conversations: dict[str, Conversation] = {}
        self._contacts: dict[str, Contact] = {}
        self._followups: dict[str, FollowUp] = {}
        self._state_bindings: dict[str, StateBinding] = {}
        # Indexes
        self._contact_conversations: dict[str, list[str]] = {}
        self._address_index: dict[str, str] = {}
        self._conv_bindings: dict[str, list[str]] = {}    # conversation_id → [binding_ids]
        self._entity_bindings: dict[str, list[str]] = {}  # "entity_type:entity_id" → [binding_ids]

    # ── Contacts ──────────────────────────────────────────────

    async def upsert_contact(self, contact: Contact) -> Contact:
        self._contacts[contact.id] = contact
        for ch in contact.channels:
            self._address_index[f"{ch.channel}:{ch.address}"] = contact.id
        return contact

    async def get_contact(self, contact_id: str) -> Optional[Contact]:
        return self._contacts.get(contact_id)

    async def find_contact_by_address(self, channel: ChannelType, address: str) -> Optional[Contact]:
        key = f"{channel}:{address}"
        cid = self._address_index.get(key)
        return self._contacts.get(cid) if cid else None

    async def list_contacts(self, filters: dict[str, Any] = None) -> list[Contact]:
        contacts = list(self._contacts.values())
        if filters:
            for k, v in filters.items():
                contacts = [c for c in contacts if getattr(c, k, None) == v]
        return contacts

    # ── Conversations ─────────────────────────────────────────

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        self._conversations[conversation.id] = conversation
        self._contact_conversations.setdefault(conversation.contact_id, []).append(conversation.id)
        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._conversations.get(conversation_id)

    async def update_conversation(self, conversation: Conversation) -> Conversation:
        conversation.updated_at = datetime.utcnow()
        self._conversations[conversation.id] = conversation
        return conversation

    async def get_active_conversation(self, contact_id: str, business_context_key: str = None) -> Optional[Conversation]:
        conv_ids = self._contact_conversations.get(contact_id, [])
        for cid in reversed(conv_ids):
            conv = self._conversations.get(cid)
            if conv and conv.status in (ConversationStatus.ACTIVE, ConversationStatus.WAITING_RESPONSE):
                if business_context_key is None:
                    return conv
                if conv.business_context.get("context_key") == business_context_key:
                    return conv
        return None

    async def get_conversations_for_contact(self, contact_id: str) -> list[Conversation]:
        conv_ids = self._contact_conversations.get(contact_id, [])
        return [self._conversations[cid] for cid in conv_ids if cid in self._conversations]

    async def list_conversations(
        self,
        status: ConversationStatus = None,
        channel: ChannelType = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        convs = list(self._conversations.values())
        if status:
            convs = [c for c in convs if c.status == status]
        if channel:
            convs = [c for c in convs if c.active_channel == channel]
        convs.sort(key=lambda c: c.updated_at, reverse=True)
        return convs[offset:offset + limit]

    # ── Follow-ups ────────────────────────────────────────────

    async def create_followup(self, followup: FollowUp) -> FollowUp:
        self._followups[followup.id] = followup
        return followup

    async def get_followup(self, followup_id: str) -> Optional[FollowUp]:
        return self._followups.get(followup_id)

    async def update_followup(self, followup: FollowUp) -> FollowUp:
        self._followups[followup.id] = followup
        return followup

    async def get_pending_followups(self) -> list[FollowUp]:
        now = datetime.utcnow()
        return [
            f for f in self._followups.values()
            if f.status in (FollowUpStatus.SCHEDULED, FollowUpStatus.AWAITING_REPLY)
            and (f.next_attempt_at is None or f.next_attempt_at <= now)
        ]

    async def list_followups(
        self,
        status: FollowUpStatus = None,
        contact_id: str = None,
        limit: int = 50,
    ) -> list[FollowUp]:
        fups = list(self._followups.values())
        if status:
            fups = [f for f in fups if f.status == status]
        if contact_id:
            fups = [f for f in fups if f.contact_id == contact_id]
        fups.sort(key=lambda f: f.scheduled_at, reverse=True)
        return fups[:limit]

    # ── State Bindings ────────────────────────────────────────

    async def create_state_binding(self, binding: StateBinding) -> StateBinding:
        self._state_bindings[binding.id] = binding
        self._conv_bindings.setdefault(binding.conversation_id, []).append(binding.id)
        if binding.entity_id:
            key = f"{binding.entity_type}:{binding.entity_id}"
            self._entity_bindings.setdefault(key, []).append(binding.id)
        return binding

    async def get_state_binding(self, binding_id: str) -> Optional[StateBinding]:
        return self._state_bindings.get(binding_id)

    async def update_state_binding(self, binding: StateBinding) -> StateBinding:
        binding.updated_at = datetime.utcnow()
        self._state_bindings[binding.id] = binding
        return binding

    async def get_bindings_for_conversation(self, conversation_id: str) -> list[StateBinding]:
        ids = self._conv_bindings.get(conversation_id, [])
        return [self._state_bindings[bid] for bid in ids if bid in self._state_bindings]

    async def get_bindings_for_entity(self, entity_type: str, entity_id: str) -> list[StateBinding]:
        key = f"{entity_type}:{entity_id}"
        ids = self._entity_bindings.get(key, [])
        return [self._state_bindings[bid] for bid in ids if bid in self._state_bindings]

    async def get_active_bindings_for_contact(self, contact_id: str) -> list[StateBinding]:
        return [
            b for b in self._state_bindings.values()
            if b.contact_id == contact_id and not b.is_terminal
        ]

    async def list_state_bindings(
        self,
        process_type: str = None,
        current_state: str = None,
        contact_id: str = None,
        limit: int = 50,
    ) -> list[StateBinding]:
        bindings = list(self._state_bindings.values())
        if process_type:
            bindings = [b for b in bindings if b.process_type == process_type]
        if current_state:
            bindings = [b for b in bindings if b.current_state == current_state]
        if contact_id:
            bindings = [b for b in bindings if b.contact_id == contact_id]
        bindings.sort(key=lambda b: b.updated_at, reverse=True)
        return bindings[:limit]


# ──────────────────────────────────────────────────────────────
#  Context Tracker — high-level API
# ──────────────────────────────────────────────────────────────

class ContextTracker:
    """
    High-level API for managing conversation context.
    Provides cross-channel continuity, business context binding,
    and state machine integration.
    """

    def __init__(self, store: ContextStore = None):
        self.store = store or ContextStore()

    # ── Cross-Channel Message Routing ─────────────────────────

    async def route_inbound_message(
        self,
        channel: ChannelType,
        sender_address: str,
        content: str,
        metadata: dict[str, Any] = None,
    ) -> tuple[Contact, Conversation, Message]:
        """
        Route an inbound message to the right contact and conversation.
        Creates contact/conversation if needed.
        """
        contact = await self.store.find_contact_by_address(channel, sender_address)
        if not contact:
            contact = Contact(
                name=f"Unknown ({sender_address})",
                channels=[ContactChannel(channel=channel, address=sender_address)],
            )
            contact = await self.store.upsert_contact(contact)
            logger.info("created_new_contact", contact_id=contact.id, address=sender_address)

        conversation = await self.store.get_active_conversation(contact.id)
        if not conversation:
            conversation = Conversation(
                contact_id=contact.id,
                status=ConversationStatus.ACTIVE,
                active_channel=channel,
                channels_used=[channel],
            )
            conversation = await self.store.create_conversation(conversation)
            logger.info("created_new_conversation", conv_id=conversation.id, contact_id=contact.id)

        if channel not in conversation.channels_used:
            conversation.channels_used.append(channel)
        conversation.active_channel = channel
        conversation.status = ConversationStatus.ACTIVE

        message = Message(
            conversation_id=conversation.id,
            channel=channel,
            direction=MessageDirection.INBOUND,
            sender_id=contact.id,
            content=content,
            metadata=metadata or {},
        )
        conversation.messages.append(message)
        await self.store.update_conversation(conversation)

        return contact, conversation, message

    async def record_outbound_message(
        self,
        conversation_id: str,
        channel: ChannelType,
        content: str,
        metadata: dict[str, Any] = None,
    ) -> Message:
        """Record an outbound message sent by the agent."""
        conversation = await self.store.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = Message(
            conversation_id=conversation_id,
            channel=channel,
            direction=MessageDirection.OUTBOUND,
            sender_id="agent",
            content=content,
            metadata=metadata or {},
            delivered=True,
        )
        conversation.messages.append(message)
        conversation.attempt_count += 1

        if channel not in conversation.channels_used:
            conversation.channels_used.append(channel)

        conversation.status = ConversationStatus.WAITING_RESPONSE
        await self.store.update_conversation(conversation)
        return message

    # ── State Binding Management ──────────────────────────────

    async def bind_state(self, binding: StateBinding) -> StateBinding:
        """Attach a state binding to a conversation and persist it."""
        binding = await self.store.create_state_binding(binding)
        conv = await self.store.get_conversation(binding.conversation_id)
        if conv and binding.id not in conv.state_bindings:
            conv.state_bindings.append(binding.id)
            await self.store.update_conversation(conv)
        logger.info("state_bound",
                     binding_id=binding.id,
                     conversation_id=binding.conversation_id,
                     process_type=binding.process_type,
                     initial_state=binding.current_state)
        return binding

    async def get_conversation_bindings(self, conversation_id: str) -> list[StateBinding]:
        return await self.store.get_bindings_for_conversation(conversation_id)

    async def persist_binding(self, binding: StateBinding) -> StateBinding:
        return await self.store.update_state_binding(binding)

    # ── Context Change Construction ───────────────────────────

    async def build_context_change(
        self,
        conversation_id: str,
        trigger_type: str,
        trigger_value: str,
        extra_data: dict[str, Any] = None,
        source_channel: ChannelType = None,
    ) -> ContextChange:
        """
        Build a ContextChange enriched with merged context data from the
        conversation, contact, and business context. The state machine uses
        this merged data to evaluate transition conditions.
        """
        conv = await self.store.get_conversation(conversation_id)
        contact = await self.store.get_contact(conv.contact_id) if conv else None

        context_data = {}
        if conv:
            context_data.update(conv.business_context)
            context_data["conversation_status"] = conv.status.value
            context_data["attempt_count"] = conv.attempt_count
            context_data["message_count"] = len(conv.messages)
        if contact:
            context_data["contact_name"] = contact.name
            context_data["contact_role"] = contact.role
            context_data["contact_organization"] = contact.organization
        if extra_data:
            context_data.update(extra_data)

        return ContextChange(
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            conversation_id=conversation_id,
            contact_id=conv.contact_id if conv else "",
            context_data=context_data,
            source_channel=source_channel,
        )

    # ── Business Context ──────────────────────────────────────

    async def bind_business_context(
        self,
        conversation_id: str,
        context: dict[str, Any],
    ) -> Conversation:
        conversation = await self.store.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        conversation.business_context.update(context)
        return await self.store.update_conversation(conversation)

    async def update_binding_business_data(
        self,
        binding_id: str,
        data: dict[str, Any],
    ) -> StateBinding:
        binding = await self.store.get_state_binding(binding_id)
        if not binding:
            raise ValueError(f"Binding {binding_id} not found")
        binding.business_data.update(data)
        return await self.store.update_state_binding(binding)

    async def get_full_context(self, conversation_id: str) -> dict[str, Any]:
        """
        Build the full context for an LLM call — includes:
        - Contact info, business context, conversation history
        - Follow-up details
        - State bindings with current states and transition history
        """
        conversation = await self.store.get_conversation(conversation_id)
        if not conversation:
            return {}

        contact = await self.store.get_contact(conversation.contact_id)
        followup = None
        if conversation.followup_id:
            followup = await self.store.get_followup(conversation.followup_id)

        history = []
        for msg in conversation.messages[-20:]:
            history.append({
                "role": "contact" if msg.direction == MessageDirection.INBOUND else "agent",
                "content": msg.content,
                "channel": msg.channel.value,
                "timestamp": msg.timestamp.isoformat(),
            })

        # State bindings with current position in each business process
        bindings = await self.store.get_bindings_for_conversation(conversation_id)
        state_summaries = []
        for b in bindings:
            state_summaries.append({
                "binding_id": b.id,
                "process_type": b.process_type,
                "current_state": b.current_state,
                "entity_id": b.entity_id,
                "entity_type": b.entity_type,
                "is_terminal": b.is_terminal,
                "transitions": len(b.history),
                "last_transition": (
                    f"{b.last_transition.from_state} → {b.last_transition.to_state}"
                    if b.last_transition else None
                ),
            })

        return {
            "conversation_id": conversation.id,
            "contact": contact.model_dump() if contact else {},
            "business_context": conversation.business_context,
            "conversation_history": history,
            "channels_used": [c.value for c in conversation.channels_used],
            "active_channel": conversation.active_channel.value if conversation.active_channel else None,
            "followup": followup.model_dump() if followup else {},
            "attempt_count": conversation.attempt_count,
            "status": conversation.status.value,
            "state_bindings": state_summaries,
        }

    # ── Conversation Lifecycle ────────────────────────────────

    async def resolve_conversation(self, conversation_id: str, outcome: str = "") -> Conversation:
        conversation = await self.store.get_conversation(conversation_id)
        if conversation:
            conversation.status = ConversationStatus.RESOLVED
            conversation.resolved_at = datetime.utcnow()
            conversation.summary = outcome
            await self.store.update_conversation(conversation)

            if conversation.followup_id:
                followup = await self.store.get_followup(conversation.followup_id)
                if followup:
                    followup.status = FollowUpStatus.COMPLETED
                    followup.completed_at = datetime.utcnow()
                    followup.outcome = outcome
                    await self.store.update_followup(followup)

        return conversation

    async def escalate_conversation(self, conversation_id: str, reason: str = "") -> Conversation:
        conversation = await self.store.get_conversation(conversation_id)
        if conversation:
            conversation.status = ConversationStatus.ESCALATED
            conversation.metadata["escalation_reason"] = reason
            await self.store.update_conversation(conversation)
        return conversation

    # ── Analytics ─────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        all_convs = list(self.store._conversations.values())
        all_fups = list(self.store._followups.values())
        all_bindings = list(self.store._state_bindings.values())

        state_dist: dict[str, dict[str, int]] = {}
        for b in all_bindings:
            state_dist.setdefault(b.process_type, {})
            state_dist[b.process_type][b.current_state] = (
                state_dist[b.process_type].get(b.current_state, 0) + 1
            )

        return {
            "total_conversations": len(all_convs),
            "active_conversations": len([c for c in all_convs if c.status == ConversationStatus.ACTIVE]),
            "waiting_response": len([c for c in all_convs if c.status == ConversationStatus.WAITING_RESPONSE]),
            "resolved": len([c for c in all_convs if c.status == ConversationStatus.RESOLVED]),
            "escalated": len([c for c in all_convs if c.status == ConversationStatus.ESCALATED]),
            "total_followups": len(all_fups),
            "pending_followups": len([f for f in all_fups if f.status == FollowUpStatus.SCHEDULED]),
            "completed_followups": len([f for f in all_fups if f.status == FollowUpStatus.COMPLETED]),
            "channels": {
                ch.value: len([c for c in all_convs if ch in c.channels_used])
                for ch in ChannelType
            },
            "total_contacts": len(self.store._contacts),
            "total_state_bindings": len(all_bindings),
            "active_state_bindings": len([b for b in all_bindings if not b.is_terminal]),
            "state_distribution": state_dist,
        }
