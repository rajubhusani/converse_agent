"""Tests for BusinessStateMachine — the core FSM engine."""
import pytest
from models.schemas import (
    StateMapDef, StateTransitionDef, TransitionTrigger, TransitionAction,
    StateBinding, FollowUpPriority, RuleCondition,
)
from context.state_machine import BusinessStateMachine, TransitionResult


class TestStateMachineRegistration:
    def test_register_valid_map(self, payment_state_map):
        sm = BusinessStateMachine()
        sm.register_map(payment_state_map)
        assert sm.get_map("payment_collection") is not None
        assert len(sm.list_maps()) == 1

    def test_register_invalid_initial_state(self):
        sm = BusinessStateMachine()
        bad_map = StateMapDef(
            process_type="bad",
            states=["a", "b"],
            initial_state="nonexistent",
        )
        with pytest.raises(ValueError, match="initial_state"):
            sm.register_map(bad_map)

    def test_register_invalid_terminal_state(self):
        sm = BusinessStateMachine()
        bad_map = StateMapDef(
            process_type="bad",
            states=["a", "b"],
            initial_state="a",
            terminal_states=["c"],
        )
        with pytest.raises(ValueError, match="terminal_state"):
            sm.register_map(bad_map)

    def test_register_invalid_transition_state(self):
        sm = BusinessStateMachine()
        bad_map = StateMapDef(
            process_type="bad",
            states=["a", "b"],
            initial_state="a",
            transitions=[
                StateTransitionDef(
                    from_states=["nonexistent"],
                    to_state="b",
                    trigger=TransitionTrigger(type="intent", value="go"),
                ),
            ],
        )
        with pytest.raises(ValueError, match="from_state"):
            sm.register_map(bad_map)

    def test_register_from_config(self):
        sm = BusinessStateMachine()
        config = [{
            "process_type": "test",
            "states": ["start", "end"],
            "initial_state": "start",
            "terminal_states": ["end"],
            "transitions": [{
                "from_states": ["start"],
                "to_state": "end",
                "trigger": {"type": "intent", "value": "done"},
            }],
        }]
        sm.register_maps_from_config(config)
        assert sm.get_map("test") is not None


class TestBindingCreation:
    def test_create_binding(self, state_machine):
        binding = state_machine.create_binding(
            process_type="payment_collection",
            conversation_id="conv-1",
            contact_id="contact-1",
            entity_id="INV-001",
            entity_type="invoice",
        )
        assert binding is not None
        assert binding.current_state == "pending"
        assert binding.process_type == "payment_collection"
        assert not binding.is_terminal

    def test_create_binding_custom_initial(self, state_machine):
        binding = state_machine.create_binding(
            process_type="payment_collection",
            conversation_id="conv-1",
            contact_id="contact-1",
            initial_state="reminded",
        )
        assert binding.current_state == "reminded"

    def test_create_binding_unknown_process(self, state_machine):
        binding = state_machine.create_binding(
            process_type="nonexistent",
            conversation_id="conv-1",
            contact_id="contact-1",
        )
        assert binding is None

    def test_create_binding_invalid_initial(self, state_machine):
        binding = state_machine.create_binding(
            process_type="payment_collection",
            conversation_id="conv-1",
            contact_id="contact-1",
            initial_state="nonexistent",
        )
        assert binding is None


class TestTriggerApplication:
    def test_basic_transition(self, state_machine, sample_binding):
        # pending → reminded via action:outreach_sent
        result = state_machine.apply_trigger(
            sample_binding, "action", "outreach_sent",
        )
        assert result.transitioned
        assert result.from_state == "pending"
        assert result.to_state == "reminded"
        assert sample_binding.current_state == "reminded"
        assert len(sample_binding.history) == 1

    def test_no_matching_transition(self, state_machine, sample_binding):
        result = state_machine.apply_trigger(
            sample_binding, "intent", "payment_promised",
        )
        # pending has no transition for intent:payment_promised
        assert not result.transitioned
        assert sample_binding.current_state == "pending"

    def test_wildcard_from_state(self, state_machine, sample_binding):
        # Move to reminded first
        state_machine.apply_trigger(sample_binding, "action", "outreach_sent")
        assert sample_binding.current_state == "reminded"

        # Wildcard transition: * → confirmed via intent:payment_confirmed
        result = state_machine.apply_trigger(
            sample_binding, "intent", "payment_confirmed",
        )
        assert result.transitioned
        assert result.to_state == "confirmed"
        assert sample_binding.is_terminal

    def test_terminal_state_blocks_transitions(self, state_machine, sample_binding):
        # Move to terminal state
        sample_binding.current_state = "confirmed"
        sample_binding.metadata["is_terminal"] = True

        result = state_machine.apply_trigger(
            sample_binding, "intent", "escalation_needed",
        )
        assert not result.transitioned

    def test_transition_with_actions(self, state_machine, sample_binding):
        # pending → reminded
        state_machine.apply_trigger(sample_binding, "action", "outreach_sent")
        # reminded → acknowledged (has schedule_check action)
        result = state_machine.apply_trigger(
            sample_binding, "intent", "acknowledged",
        )
        assert result.transitioned
        assert len(result.actions) == 1
        assert result.actions[0].type == "schedule_check"

    def test_transition_with_multiple_actions(self, state_machine, sample_binding):
        # Move to reminded, then to promised
        state_machine.apply_trigger(sample_binding, "action", "outreach_sent")
        result = state_machine.apply_trigger(
            sample_binding, "intent", "payment_promised",
        )
        assert result.transitioned
        assert result.to_state == "promised"
        assert len(result.actions) == 2
        assert result.actions[0].type == "update_backend"
        assert result.actions[1].type == "schedule_check"

    def test_or_match_trigger_values(self, state_machine, sample_binding):
        # The "payment_confirmed" trigger uses values list
        sample_binding.current_state = "reminded"
        sample_binding.metadata["is_terminal"] = False
        result = state_machine.apply_trigger(
            sample_binding, "intent", "payment_confirmed",
        )
        assert result.transitioned
        assert result.to_state == "confirmed"

    def test_event_trigger(self, state_machine, sample_binding):
        sample_binding.current_state = "promised"
        sample_binding.metadata["is_terminal"] = False
        result = state_machine.apply_trigger(
            sample_binding, "event", "payment_received",
        )
        assert result.transitioned
        assert result.to_state == "confirmed"
        assert len(result.actions) == 2  # enqueue_followup + resolve

    def test_conditional_transition_passes(self, state_machine, sample_binding):
        sample_binding.current_state = "overdue"
        sample_binding.metadata["is_terminal"] = False
        result = state_machine.apply_trigger(
            sample_binding, "timeout", "max_attempts_reached",
            context_data={"days_overdue": 20},  # > 14, condition passes
        )
        assert result.transitioned
        assert result.to_state == "escalated"

    def test_conditional_transition_fails(self, state_machine, sample_binding):
        sample_binding.current_state = "overdue"
        sample_binding.metadata["is_terminal"] = False
        result = state_machine.apply_trigger(
            sample_binding, "timeout", "max_attempts_reached",
            context_data={"days_overdue": 5},  # <= 14, condition fails
        )
        assert not result.transitioned

    def test_transition_records_history(self, state_machine, sample_binding):
        state_machine.apply_trigger(sample_binding, "action", "outreach_sent")
        state_machine.apply_trigger(sample_binding, "intent", "acknowledged")
        state_machine.apply_trigger(sample_binding, "intent", "payment_promised")

        assert len(sample_binding.history) == 3
        assert sample_binding.history[0].from_state == "pending"
        assert sample_binding.history[0].to_state == "reminded"
        assert sample_binding.history[2].to_state == "promised"
        assert sample_binding.last_transition.to_state == "promised"


class TestApplyTriggerToAll:
    def test_applies_to_multiple_bindings(self, state_machine):
        b1 = state_machine.create_binding("payment_collection", "c1", "ct1")
        b2 = state_machine.create_binding("payment_collection", "c2", "ct2")
        b1.current_state = "reminded"
        b1.metadata["is_terminal"] = False
        b2.current_state = "reminded"
        b2.metadata["is_terminal"] = False

        results = state_machine.apply_trigger_to_all(
            [b1, b2], "intent", "acknowledged",
        )
        assert len(results) == 2
        assert all(r.transitioned for r in results)

    def test_returns_only_transitioned(self, state_machine):
        b1 = state_machine.create_binding("payment_collection", "c1", "ct1")
        b2 = state_machine.create_binding("payment_collection", "c2", "ct2")
        # b1 is pending, b2 is reminded — only b2 matches intent:acknowledged
        b2.current_state = "reminded"
        b2.metadata["is_terminal"] = False

        results = state_machine.apply_trigger_to_all(
            [b1, b2], "intent", "acknowledged",
        )
        assert len(results) == 1
        assert results[0].binding.id == b2.id


class TestIntrospection:
    def test_get_available_transitions(self, state_machine, sample_binding):
        available = state_machine.get_available_transitions(sample_binding)
        # pending can transition via action:outreach_sent plus wildcards
        assert len(available) >= 1
        trigger_types = [t.trigger.type for t in available]
        assert "action" in trigger_types

    def test_get_state_info(self, state_machine, sample_binding):
        info = state_machine.get_state_info(sample_binding)
        assert info["current_state"] == "pending"
        assert info["process_type"] == "payment_collection"
        assert not info["is_terminal"]
        assert len(info["all_states"]) == 8
        assert "confirmed" in info["terminal_states"]
