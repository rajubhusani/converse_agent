"""
Business State Machine — Maps context changes to business state transitions.

This is the generic layer that replaces hardcoded intent→action branching.
Each business process (payment collection, order fulfillment, delivery tracking)
is modelled as a finite state machine loaded from configuration.

Flow:
  Context Change (intent, event, timeout)
    → State Machine evaluates current_state + trigger
    → Transition fires if matched
    → Actions returned to caller (enqueue, update_backend, resolve, escalate)

A single conversation can bind to MULTIPLE state machines simultaneously.
E.g. an order conversation tracks both "payment_collection" and "delivery_tracking".

Usage:
    sm = BusinessStateMachine()
    sm.register_map(payment_map)
    sm.register_map(delivery_map)

    # When context changes:
    results = await sm.apply_trigger(
        binding=state_binding,
        trigger_type="intent",
        trigger_value="payment_confirmed",
        context_data={...}
    )
    # results.actions → [TransitionAction(type="update_backend"), TransitionAction(type="resolve")]
"""
from __future__ import annotations

import structlog
from datetime import datetime
from typing import Any, Optional

from models.schemas import (
    StateMapDef, StateTransitionDef, TransitionAction, TransitionTrigger,
    StateBinding, StateTransitionRecord, RuleCondition,
)
from utils.conditions import evaluate_conditions

logger = structlog.get_logger()


# ──────────────────────────────────────────────────────────────
#  Transition Result
# ──────────────────────────────────────────────────────────────

class TransitionResult:
    """Outcome of applying a trigger to a state binding."""

    def __init__(
        self,
        transitioned: bool,
        binding: StateBinding,
        from_state: str = "",
        to_state: str = "",
        actions: list[TransitionAction] = None,
        record: StateTransitionRecord = None,
    ):
        self.transitioned = transitioned
        self.binding = binding
        self.from_state = from_state
        self.to_state = to_state
        self.actions = actions or []
        self.record = record

    def __bool__(self):
        return self.transitioned

    def __repr__(self):
        if self.transitioned:
            return f"<Transition {self.from_state} → {self.to_state} [{len(self.actions)} actions]>"
        return "<NoTransition>"


# ──────────────────────────────────────────────────────────────
#  Business State Machine
# ──────────────────────────────────────────────────────────────

class BusinessStateMachine:
    """
    Registry of business process state maps.
    Evaluates context triggers against the appropriate state map and returns
    transition results with actions.
    """

    def __init__(self):
        self._maps: dict[str, StateMapDef] = {}

    # ── Registration ──────────────────────────────────────────

    def register_map(self, state_map: StateMapDef):
        errors = self._validate_map(state_map)
        if errors:
            logger.error("invalid_state_map",
                         process_type=state_map.process_type,
                         errors=errors)
            raise ValueError(
                f"Invalid state map '{state_map.process_type}': {'; '.join(errors)}"
            )
        self._maps[state_map.process_type] = state_map
        logger.info("state_map_registered",
                     process_type=state_map.process_type,
                     states=len(state_map.states),
                     transitions=len(state_map.transitions))

    @staticmethod
    def _validate_map(sm: StateMapDef) -> list[str]:
        """Validate a state map definition. Returns list of error messages."""
        errors = []
        state_set = set(sm.states)

        if sm.initial_state not in state_set:
            errors.append(f"initial_state '{sm.initial_state}' not in states")

        for ts in sm.terminal_states:
            if ts not in state_set:
                errors.append(f"terminal_state '{ts}' not in states")

        for i, t in enumerate(sm.transitions):
            for fs in t.from_states:
                if fs != "*" and fs not in state_set:
                    errors.append(f"transition[{i}] from_state '{fs}' not in states")
            if t.to_state not in state_set:
                errors.append(f"transition[{i}] to_state '{t.to_state}' not in states")

        return errors

    def register_maps_from_config(self, config: list[dict[str, Any]]):
        """Load state maps from YAML config list."""
        for raw in config:
            sm = StateMapDef(**raw)
            self.register_map(sm)
        logger.info("state_maps_loaded", count=len(config))

    def get_map(self, process_type: str) -> Optional[StateMapDef]:
        return self._maps.get(process_type)

    def list_maps(self) -> list[StateMapDef]:
        return list(self._maps.values())

    # ── State Creation ────────────────────────────────────────

    def create_binding(
        self,
        process_type: str,
        conversation_id: str,
        contact_id: str,
        entity_id: str = "",
        entity_type: str = "",
        business_data: dict[str, Any] = None,
        initial_state: str = "",
    ) -> Optional[StateBinding]:
        """
        Create a new StateBinding for a conversation + business process.
        Uses the state map's initial_state unless overridden.
        """
        state_map = self._maps.get(process_type)
        if not state_map:
            logger.error("unknown_process_type", process_type=process_type)
            return None

        start = initial_state or state_map.initial_state
        if start not in state_map.states:
            logger.error("invalid_initial_state",
                         state=start,
                         valid=state_map.states)
            return None

        binding = StateBinding(
            conversation_id=conversation_id,
            contact_id=contact_id,
            process_type=process_type,
            current_state=start,
            entity_id=entity_id,
            entity_type=entity_type,
            business_data=business_data or {},
            metadata={"is_terminal": start in state_map.terminal_states},
        )

        logger.info("state_binding_created",
                     binding_id=binding.id,
                     process_type=process_type,
                     initial_state=start,
                     entity_id=entity_id)
        return binding

    # ── Trigger Application ───────────────────────────────────

    def apply_trigger(
        self,
        binding: StateBinding,
        trigger_type: str,
        trigger_value: str,
        context_data: dict[str, Any] = None,
    ) -> TransitionResult:
        """
        Evaluate a trigger against a state binding's current state.

        Args:
            binding:       The active state binding
            trigger_type:  "intent" | "event" | "timeout" | "manual" | "action"
            trigger_value: The specific trigger (e.g. "payment_confirmed", "order_shipped")
            context_data:  Merged business data + conversation context for condition evaluation

        Returns:
            TransitionResult with transitioned=True if a transition fired,
            including the actions to execute.
        """
        state_map = self._maps.get(binding.process_type)
        if not state_map:
            logger.warning("no_state_map_for_binding",
                           process_type=binding.process_type)
            return TransitionResult(transitioned=False, binding=binding)

        if binding.is_terminal:
            logger.debug("binding_already_terminal",
                         binding_id=binding.id,
                         state=binding.current_state)
            return TransitionResult(transitioned=False, binding=binding)

        # Merge business_data with context_data for condition evaluation
        eval_data = {**binding.business_data, **(context_data or {})}

        # Find matching transition
        transition = self._find_transition(
            state_map, binding.current_state, trigger_type, trigger_value, eval_data
        )

        if not transition:
            logger.debug("no_matching_transition",
                         process_type=binding.process_type,
                         current_state=binding.current_state,
                         trigger=f"{trigger_type}:{trigger_value}")
            return TransitionResult(transitioned=False, binding=binding)

        # Execute transition
        from_state = binding.current_state
        to_state = transition.to_state

        record = StateTransitionRecord(
            from_state=from_state,
            to_state=to_state,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            actions_fired=[a.type for a in transition.actions],
            metadata={
                "transition_desc": transition.description,
                "conditions_met": bool(transition.trigger.conditions),
            },
        )

        binding.current_state = to_state
        binding.history.append(record)
        binding.updated_at = datetime.utcnow()
        binding.metadata["is_terminal"] = to_state in state_map.terminal_states

        if binding.is_terminal:
            binding.resolved_at = datetime.utcnow()

        logger.info("state_transition",
                     binding_id=binding.id,
                     process_type=binding.process_type,
                     transition=f"{from_state} → {to_state}",
                     trigger=f"{trigger_type}:{trigger_value}",
                     actions=[a.type for a in transition.actions])

        return TransitionResult(
            transitioned=True,
            binding=binding,
            from_state=from_state,
            to_state=to_state,
            actions=transition.actions,
            record=record,
        )

    def apply_trigger_to_all(
        self,
        bindings: list[StateBinding],
        trigger_type: str,
        trigger_value: str,
        context_data: dict[str, Any] = None,
    ) -> list[TransitionResult]:
        """
        Apply a trigger to ALL bindings on a conversation.
        Returns only the results where a transition actually fired.
        """
        results = []
        for binding in bindings:
            result = self.apply_trigger(binding, trigger_type, trigger_value, context_data)
            if result.transitioned:
                results.append(result)
        return results

    # ── Transition Lookup ─────────────────────────────────────

    def _find_transition(
        self,
        state_map: StateMapDef,
        current_state: str,
        trigger_type: str,
        trigger_value: str,
        eval_data: dict[str, Any],
    ) -> Optional[StateTransitionDef]:
        """
        Find the first matching transition for the given state and trigger.
        Transitions are evaluated in definition order (first match wins).
        """
        for t in state_map.transitions:
            # Check if current state is in from_states (support "*" wildcard)
            if "*" not in t.from_states and current_state not in t.from_states:
                continue

            # Check trigger match
            if not t.trigger.matches(trigger_type, trigger_value):
                continue

            # Check additional conditions
            if t.trigger.conditions and not evaluate_conditions(t.trigger.conditions, eval_data):
                continue

            return t

        return None

    # ── Introspection ─────────────────────────────────────────

    def get_available_transitions(
        self,
        binding: StateBinding,
    ) -> list[StateTransitionDef]:
        """Return all transitions available from the binding's current state."""
        state_map = self._maps.get(binding.process_type)
        if not state_map:
            return []
        return [
            t for t in state_map.transitions
            if binding.current_state in t.from_states or "*" in t.from_states
        ]

    def get_state_info(self, binding: StateBinding) -> dict[str, Any]:
        """Get full state info including available transitions and history."""
        state_map = self._maps.get(binding.process_type)
        available = self.get_available_transitions(binding)
        return {
            "binding_id": binding.id,
            "process_type": binding.process_type,
            "current_state": binding.current_state,
            "is_terminal": binding.is_terminal,
            "entity_id": binding.entity_id,
            "entity_type": binding.entity_type,
            "available_triggers": [
                {
                    "trigger": f"{t.trigger.type}:{t.trigger.value or '|'.join(t.trigger.values)}",
                    "to_state": t.to_state,
                    "description": t.description,
                }
                for t in available
            ],
            "history_length": len(binding.history),
            "last_transition": binding.last_transition.model_dump() if binding.last_transition else None,
            "all_states": state_map.states if state_map else [],
            "terminal_states": state_map.terminal_states if state_map else [],
        }
