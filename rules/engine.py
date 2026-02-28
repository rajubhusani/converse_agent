"""
Rules Engine — Evaluates conditions and triggers follow-up actions.

Rules are loaded from settings.yaml and/or fetched from the backend.
The engine evaluates conditions against business data and creates
FollowUp objects when conditions are met.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from models.schemas import (
    ChannelType, FollowUp, FollowUpPriority, FollowUpStatus,
    Rule, RuleAction, RuleCondition, RuleTrigger,
)
from context.tracker import ContextTracker
from utils.conditions import evaluate_conditions

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────
#  Rules Engine
# ──────────────────────────────────────────────────────────────

class RulesEngine:
    """
    Manages rules, evaluates them against data, and creates follow-ups.
    """

    def __init__(self, context_tracker: ContextTracker):
        self.context = context_tracker
        self._rules: dict[str, Rule] = {}
        self._event_rules: dict[str, list[Rule]] = {}  # event_name -> rules
        self._schedule_rules: list[Rule] = []

    def load_rules(self, rules_config: list[dict[str, Any]]):
        """Load rules from YAML config."""
        for raw in rules_config:
            rule = Rule(**raw)
            self.register_rule(rule)
        logger.info("rules_loaded", count=len(self._rules))

    def register_rule(self, rule: Rule):
        """Register a single rule."""
        self._rules[rule.id] = rule
        if rule.trigger.type == "event" and rule.trigger.event_name:
            self._event_rules.setdefault(rule.trigger.event_name, []).append(rule)
        elif rule.trigger.type == "schedule":
            self._schedule_rules.append(rule)
        logger.info("rule_registered", rule_id=rule.id, trigger=rule.trigger.type)

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        return self._rules.get(rule_id)

    def list_rules(self) -> list[Rule]:
        return list(self._rules.values())

    # ── Event-Based Evaluation ────────────────────────────────

    async def on_event(self, event_name: str, data: dict[str, Any]) -> list[FollowUp]:
        """
        Evaluate event-triggered rules against incoming data.
        Returns list of FollowUps created.
        """
        rules = self._event_rules.get(event_name, [])
        followups = []

        for rule in rules:
            if not rule.enabled:
                continue
            if evaluate_conditions(rule.conditions, data):
                logger.info("rule_matched", rule_id=rule.id, event=event_name)
                for action in rule.actions:
                    fu = await self._create_followup(rule, action, data)
                    if fu:
                        followups.append(fu)

        return followups

    # ── Scheduled Evaluation ──────────────────────────────────

    async def evaluate_scheduled_rules(self, batch_data: list[dict[str, Any]]) -> list[FollowUp]:
        """
        Evaluate schedule-triggered rules against a batch of data
        (e.g. all pending follow-ups from the backend).
        """
        followups = []

        for rule in self._schedule_rules:
            if not rule.enabled:
                continue
            for data in batch_data:
                if evaluate_conditions(rule.conditions, data):
                    # Check if a follow-up already exists for this
                    contact_id = data.get("contact_id", "")
                    existing = await self.context.store.get_active_conversation(
                        contact_id,
                        business_context_key=data.get("id", ""),
                    )
                    if existing:
                        logger.debug("followup_already_active", rule_id=rule.id, contact_id=contact_id)
                        continue

                    for action in rule.actions:
                        fu = await self._create_followup(rule, action, data)
                        if fu:
                            followups.append(fu)

        return followups

    # ── Follow-Up Creation ────────────────────────────────────

    async def _create_followup(
        self,
        rule: Rule,
        action: RuleAction,
        data: dict[str, Any],
    ) -> Optional[FollowUp]:
        """Create a FollowUp from a matched rule + action."""
        contact_id = data.get("contact_id", "")
        if not contact_id:
            logger.warning("no_contact_id_in_data", rule_id=rule.id)
            return None

        # Determine channel priority
        channel_priority = action.channel_priority or [ChannelType.WHATSAPP, ChannelType.EMAIL]

        # Calculate next attempt time
        next_attempt = datetime.utcnow()
        if action.delay_minutes > 0:
            next_attempt += timedelta(minutes=action.delay_minutes)

        # Resolve entity ID from data using configured field name
        entity_id = ""
        if action.entity_id_field:
            entity_id = str(data.get(action.entity_id_field, ""))
        elif data.get("entity_id"):
            entity_id = str(data["entity_id"])
        elif data.get("id"):
            entity_id = str(data["id"])

        # Determine priority from data
        priority = FollowUpPriority.MEDIUM
        days_overdue = data.get("days_overdue", 0)
        if isinstance(days_overdue, (int, float)):
            if days_overdue > 14:
                priority = FollowUpPriority.URGENT
            elif days_overdue > 7:
                priority = FollowUpPriority.HIGH

        amount = data.get("amount", data.get("order_value", 0))
        if isinstance(amount, (int, float)) and amount > 100000:
            priority = FollowUpPriority.HIGH

        followup = FollowUp(
            rule_id=rule.id,
            contact_id=contact_id,
            status=FollowUpStatus.SCHEDULED,
            priority=priority,
            business_context={
                "context_key": data.get("id", ""),
                "rule_name": rule.name,
                "template": action.template,
                "process_type": action.process_type,
                "entity_type": action.entity_type,
                "entity_id": entity_id,
                "initial_state": action.initial_state,
                **data,
            },
            reason=rule.description or rule.name,
            channel_priority=channel_priority,
            max_attempts=action.escalation.get("after_attempts", 3) if action.escalation else 3,
            next_attempt_at=next_attempt,
            metadata={
                "action": action.model_dump(),
                "escalation": action.escalation,
            },
        )

        followup = await self.context.store.create_followup(followup)
        logger.info(
            "followup_created",
            followup_id=followup.id,
            rule_id=rule.id,
            contact_id=contact_id,
            priority=priority.value,
        )
        return followup
