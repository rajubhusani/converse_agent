"""Tests for the agentic dialogue template enhancements."""
import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from models.schemas import ChannelType, RuleCondition
from templates.models import (
    DialogueFlow, DialogueStep, StepType, ChannelVariant,
    VariableDef, BranchArm, ToolDef, FlowExecutionResult,
)
from templates.registry import DialogueFlowRegistry
from templates.executor import DialogueExecutor
from templates.memory import FlowMemoryStore, MemoryType, MemoryEntry, MemoryScope
from templates.tool_registry import (
    ToolRegistry, ToolSchema, ToolCategory,
    create_default_tool_registry,
)
from templates.planner import AgenticPlanner, PlanObjective, PlanResult
from templates.sessions import (
    classify_intent, extract_entities,
    llm_classify_intent, llm_extract_entities,
)


# ══════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════

@pytest.fixture
def memory_store():
    return FlowMemoryStore()


@pytest.fixture
def tool_registry():
    return create_default_tool_registry()


@pytest.fixture
def flow_registry():
    registry = DialogueFlowRegistry()
    # Register a simple sub-flow for call_flow tests
    sub_flow = DialogueFlow(
        id="sub_greet",
        name="Sub Greeting",
        composable=True,
        variables=[
            VariableDef(name="name", source="contact_name", default="friend"),
        ],
        steps=[
            DialogueStep(id="greet", type=StepType.MESSAGE,
                         content="Hello from sub-flow, {{name}}!"),
        ],
    )
    registry.register(sub_flow)
    return registry


@pytest.fixture
def executor(flow_registry, memory_store):
    return DialogueExecutor(
        llm_generate=None,
        tool_caller=None,
        builtin_tools={},
        flow_registry=flow_registry,
        memory_store=memory_store,
    )


@pytest.fixture
def base_context():
    return {
        "contact": {"id": "c1", "name": "Alice"},
        "conversation_id": "conv1",
        "business_context": {"amount": 5000, "invoice_number": "INV-001"},
        "business_data": {"amount": 5000},
        "state_bindings": [],
    }


# ══════════════════════════════════════════════════════
#  TOOL REGISTRY TESTS
# ══════════════════════════════════════════════════════

class TestToolRegistry:

    def test_register_and_get(self, tool_registry):
        assert tool_registry.get("$check_business_hours") is not None
        assert tool_registry.get("nonexistent") is None

    def test_default_builtins_registered(self, tool_registry):
        assert tool_registry.count >= 3
        names = [t.name for t in tool_registry.list_all()]
        assert "$check_business_hours" in names
        assert "$get_conversation_age" in names
        assert "$get_attempt_summary" in names

    def test_register_backend_endpoint(self, tool_registry):
        tool_registry.register_backend_endpoint(
            name="check_payment",
            endpoint="/api/payments/check",
            description="Check payment status",
            output_keys=["paid", "amount", "date"],
            category=ToolCategory.DATA_FETCH,
        )
        tool = tool_registry.get("check_payment")
        assert tool is not None
        assert tool.endpoint == "/api/payments/check"
        assert "paid" in tool.output_keys

    def test_register_builtin_with_handler(self, tool_registry):
        async def my_tool(ctx):
            return {"result": "ok"}
        tool_registry.register_builtin(
            name="$my_custom",
            description="Custom tool",
            handler=my_tool,
            output_keys=["result"],
        )
        assert tool_registry.get("$my_custom") is not None
        assert tool_registry.get_handler("$my_custom") is my_tool

    def test_list_for_context(self, tool_registry):
        # Register a state-specific tool
        tool_registry.register(ToolSchema(
            name="escalation_tool",
            description="Only for escalated state",
            applicable_states=["escalated"],
        ))
        # Should not appear for "outreach_1" state
        tools = tool_registry.list_for_context(state="outreach_1")
        tool_names = [t.name for t in tools]
        assert "escalation_tool" not in tool_names
        # Should appear for "escalated" state
        tools = tool_registry.list_for_context(state="escalated")
        tool_names = [t.name for t in tools]
        assert "escalation_tool" in tool_names

    def test_list_for_context_process_filter(self, tool_registry):
        tool_registry.register(ToolSchema(
            name="payment_only",
            description="Only for payment process",
            applicable_process_types=["payment_collection"],
        ))
        tools = tool_registry.list_for_context(process_type="feedback_collection")
        assert all(t.name != "payment_only" for t in tools)
        tools = tool_registry.list_for_context(process_type="payment_collection")
        assert any(t.name == "payment_only" for t in tools)

    def test_describe_for_llm(self, tool_registry):
        desc = tool_registry.describe_for_llm()
        assert "Available tools:" in desc
        assert "$check_business_hours" in desc

    def test_validate_tool_reference(self, tool_registry):
        assert tool_registry.validate_tool_reference("$check_business_hours") is None
        assert tool_registry.validate_tool_reference("$nonexistent") is not None
        assert "Unknown" in tool_registry.validate_tool_reference("$nonexistent")

    def test_disabled_tool_excluded(self, tool_registry):
        tool_registry.register(ToolSchema(
            name="disabled_tool",
            description="Disabled",
            enabled=False,
        ))
        tools = tool_registry.list_all()
        assert all(t.name != "disabled_tool" for t in tools)

    def test_matches_context(self):
        tool = ToolSchema(
            name="test", description="test",
            applicable_process_types=["payment"],
            applicable_states=["overdue"],
        )
        assert tool.matches_context("payment", "overdue") is True
        assert tool.matches_context("payment", "new") is False
        assert tool.matches_context("feedback", "overdue") is False
        # No filters = matches everything
        tool2 = ToolSchema(name="open", description="open")
        assert tool2.matches_context("anything", "anything") is True


# ══════════════════════════════════════════════════════
#  FLOW MEMORY TESTS
# ══════════════════════════════════════════════════════

class TestFlowMemory:

    def test_remember_and_recall(self, memory_store):
        memory_store.remember(
            scope_type="contact", scope_id="c1",
            key="preferred_channel", value="whatsapp",
            mem_type=MemoryType.PREFERENCE,
        )
        entry = memory_store.recall("contact", "c1", "preferred_channel")
        assert entry is not None
        assert entry.value == "whatsapp"
        assert entry.type == MemoryType.PREFERENCE

    def test_recall_value(self, memory_store):
        memory_store.remember("contact", "c1", "name", "Alice")
        assert memory_store.recall_value("contact", "c1", "name") == "Alice"
        assert memory_store.recall_value("contact", "c1", "missing", "default") == "default"

    def test_higher_confidence_wins(self, memory_store):
        memory_store.remember("contact", "c1", "status", "unknown", confidence=0.5)
        memory_store.remember("contact", "c1", "status", "active", confidence=0.9)
        assert memory_store.recall_value("contact", "c1", "status") == "active"

    def test_lower_confidence_does_not_replace(self, memory_store):
        memory_store.remember("contact", "c1", "status", "confirmed", confidence=0.9)
        memory_store.remember("contact", "c1", "status", "maybe", confidence=0.3)
        assert memory_store.recall_value("contact", "c1", "status") == "confirmed"

    def test_scope_isolation(self, memory_store):
        memory_store.remember("contact", "c1", "key", "contact_val")
        memory_store.remember("conversation", "conv1", "key", "conv_val")
        assert memory_store.recall_value("contact", "c1", "key") == "contact_val"
        assert memory_store.recall_value("conversation", "conv1", "key") == "conv_val"

    def test_expired_memory(self, memory_store):
        memory_store.remember(
            "contact", "c1", "temp", "value",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert memory_store.recall("contact", "c1", "temp") is None

    def test_build_context_overlay(self, memory_store):
        memory_store.remember("contact", "c1", "pref", "whatsapp")
        memory_store.remember("conversation", "conv1", "status", "active")
        overlay = memory_store.build_context_overlay(
            contact_id="c1", conversation_id="conv1",
        )
        assert overlay["memory.pref"] == "whatsapp"
        assert overlay["memory.status"] == "active"

    def test_describe_for_llm(self, memory_store):
        memory_store.remember("contact", "c1", "paid", True, mem_type=MemoryType.FACT)
        desc = memory_store.describe_for_llm(contact_id="c1")
        assert "paid" in desc
        assert "fact" in desc.lower()

    def test_recall_by_type(self, memory_store):
        scope = memory_store.get_scope("contact", "c1")
        scope.remember(MemoryEntry(
            type=MemoryType.FACT, key="a", value=1,
        ))
        scope.remember(MemoryEntry(
            type=MemoryType.PREFERENCE, key="b", value=2,
        ))
        scope.remember(MemoryEntry(
            type=MemoryType.FACT, key="c", value=3,
        ))
        facts = scope.recall_by_type(MemoryType.FACT)
        assert len(facts) == 2
        prefs = scope.recall_by_type(MemoryType.PREFERENCE)
        assert len(prefs) == 1

    def test_recall_by_tag(self, memory_store):
        scope = memory_store.get_scope("contact", "c1")
        scope.remember(MemoryEntry(
            type=MemoryType.FACT, key="a", value=1, tags=["payment"],
        ))
        scope.remember(MemoryEntry(
            type=MemoryType.FACT, key="b", value=2, tags=["feedback"],
        ))
        payment = scope.recall_by_tag("payment")
        assert len(payment) == 1

    def test_get_stats(self, memory_store):
        memory_store.remember("contact", "c1", "a", 1)
        memory_store.remember("conversation", "conv1", "b", 2)
        stats = memory_store.get_stats()
        assert stats["scopes"] == 2
        assert stats["total_entries"] == 2

    def test_clean_expired(self, memory_store):
        scope = memory_store.get_scope("contact", "c1")
        scope.remember(MemoryEntry(
            type=MemoryType.FACT, key="old", value="x",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        ))
        scope.remember(MemoryEntry(
            type=MemoryType.FACT, key="fresh", value="y",
        ))
        scope.clean_expired()
        assert "old" not in scope.entries
        assert "fresh" in scope.entries


# ══════════════════════════════════════════════════════
#  NEW STEP TYPE TESTS
# ══════════════════════════════════════════════════════

class TestCallFlowStep:

    @pytest.mark.asyncio
    async def test_call_flow_basic(self, executor, base_context):
        flow = DialogueFlow(
            id="parent", name="Parent",
            steps=[
                DialogueStep(
                    id="call_sub", type=StepType.CALL_FLOW,
                    sub_flow_id="sub_greet",
                    sub_flow_input={"contact_name": "contact.name"},
                ),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "completed"
        assert "Hello from sub-flow" in result.rendered_message
        assert "Alice" in result.rendered_message

    @pytest.mark.asyncio
    async def test_call_flow_missing(self, executor, base_context):
        flow = DialogueFlow(
            id="parent", name="Parent",
            steps=[
                DialogueStep(
                    id="call_sub", type=StepType.CALL_FLOW,
                    sub_flow_id="nonexistent_flow",
                ),
            ],
        )
        result = await executor.execute(flow, base_context)
        failed = [s for s in result.steps_executed if s.status == "failed"]
        assert len(failed) == 1
        assert "not found" in failed[0].error


class TestParallelStep:

    @pytest.mark.asyncio
    async def test_parallel_builtins(self, executor, base_context):
        flow = DialogueFlow(
            id="par", name="Parallel",
            steps=[
                DialogueStep(
                    id="gather", type=StepType.PARALLEL,
                    parallel_tools=[
                        ToolDef(endpoint="$get_attempt_summary", result_key="attempt"),
                        ToolDef(endpoint="$get_conversation_age", result_key="age"),
                    ],
                    parallel_merge_key="all_data",
                ),
                DialogueStep(
                    id="msg", type=StepType.MESSAGE,
                    content="Attempt: {{attempt.attempt_count}}, Age: {{age.age_seconds}}s",
                ),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "completed"
        assert "Attempt:" in result.rendered_message

    @pytest.mark.asyncio
    async def test_parallel_empty_tools(self, executor, base_context):
        flow = DialogueFlow(
            id="par_empty", name="Empty Parallel",
            steps=[
                DialogueStep(id="gather", type=StepType.PARALLEL, parallel_tools=[]),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "completed"


class TestLoopStep:

    @pytest.mark.asyncio
    async def test_loop_basic(self, executor, base_context):
        flow = DialogueFlow(
            id="loop_test", name="Loop",
            steps=[
                DialogueStep(
                    id="init", type=StepType.ACTION,
                    action_type="set_variable",
                    action_config={"key": "counter", "value": 0},
                ),
                DialogueStep(
                    id="loop", type=StepType.LOOP,
                    loop_steps=["msg_step"],
                    loop_condition=[
                        RuleCondition(field="_loop_iteration", operator="lt", value=3),
                    ],
                    loop_max_iterations=5,
                ),
                DialogueStep(
                    id="msg_step", type=StepType.MESSAGE,
                    content="Iteration!",
                ),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_loop_max_iterations(self, executor, base_context):
        """Loop should stop at max_iterations even if condition is still true."""
        flow = DialogueFlow(
            id="loop_max", name="Loop Max",
            steps=[
                DialogueStep(
                    id="loop", type=StepType.LOOP,
                    loop_steps=["msg"],
                    loop_condition=[
                        RuleCondition(field="always_true", operator="eq", value=True),
                    ],
                    loop_max_iterations=2,
                ),
                DialogueStep(id="msg", type=StepType.MESSAGE, content="Hi"),
            ],
        )
        base_context["always_true"] = True
        result = await executor.execute(flow, base_context)
        loop_step = [s for s in result.steps_executed if s.step_type == "loop"][0]
        assert loop_step.metadata["iterations"] == 2


class TestRememberStep:

    @pytest.mark.asyncio
    async def test_remember_stores_memory(self, executor, base_context, memory_store):
        flow = DialogueFlow(
            id="rem", name="Remember",
            steps=[
                DialogueStep(
                    id="store", type=StepType.REMEMBER,
                    remember_key="contact_mood",
                    remember_value="happy",
                    remember_type="observation",
                    remember_scope="conversation",
                ),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "completed"
        assert len(result.memories_created) == 1
        # Verify in memory store
        entry = memory_store.recall("conversation", "conv1", "contact_mood")
        assert entry is not None
        assert entry.value == "happy"

    @pytest.mark.asyncio
    async def test_remember_with_interpolation(self, executor, base_context, memory_store):
        flow = DialogueFlow(
            id="rem_interp", name="Remember Interp",
            variables=[
                VariableDef(name="amt", source="business_data.amount", format="currency"),
            ],
            steps=[
                DialogueStep(
                    id="store", type=StepType.REMEMBER,
                    remember_key="last_amount",
                    remember_value="{{amt}}",
                    remember_type="fact",
                    remember_scope="contact",
                ),
            ],
        )
        result = await executor.execute(flow, base_context)
        entry = memory_store.recall("contact", "c1", "last_amount")
        assert entry is not None
        assert "5,000" in str(entry.value) or "5000" in str(entry.value)


# ══════════════════════════════════════════════════════
#  LIFECYCLE HOOKS TESTS
# ══════════════════════════════════════════════════════

class TestLifecycleHooks:

    @pytest.mark.asyncio
    async def test_on_enter_hook(self, executor, base_context, memory_store):
        flow = DialogueFlow(
            id="hooks", name="Hooks Test",
            on_enter=[
                {"type": "remember", "key": "flow_started", "value": "true", "scope": "conversation"},
                {"type": "log", "message": "Starting flow for {{contact.name}}"},
            ],
            steps=[
                DialogueStep(id="msg", type=StepType.MESSAGE, content="Hello!"),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "completed"
        assert len(result.hook_results) >= 1
        # Verify memory was stored
        entry = memory_store.recall("conversation", "conv1", "flow_started")
        assert entry is not None

    @pytest.mark.asyncio
    async def test_on_exit_hook(self, executor, base_context, memory_store):
        flow = DialogueFlow(
            id="hooks_exit", name="Exit Hook",
            on_exit=[
                {"type": "remember", "key": "flow_completed", "value": "true", "scope": "conversation"},
            ],
            steps=[
                DialogueStep(id="msg", type=StepType.MESSAGE, content="Done!"),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "completed"
        entry = memory_store.recall("conversation", "conv1", "flow_completed")
        assert entry is not None

    @pytest.mark.asyncio
    async def test_on_error_hook(self, executor, base_context, memory_store):
        flow = DialogueFlow(
            id="hooks_error", name="Error Hook",
            on_error=[
                {"type": "remember", "key": "flow_failed", "value": "true", "scope": "conversation"},
            ],
            steps=[
                DialogueStep(id="gate", type=StepType.GATE,
                             conditions=[
                                 RuleCondition(field="impossible", operator="eq", value=True),
                             ]),
                # Gate aborts → on_error should fire
            ],
        )
        result = await executor.execute(flow, base_context)
        assert result.status == "aborted"
        entry = memory_store.recall("conversation", "conv1", "flow_failed")
        assert entry is not None

    @pytest.mark.asyncio
    async def test_context_change_hook(self, executor, base_context):
        flow = DialogueFlow(
            id="hooks_ctx", name="Context Hook",
            on_enter=[
                {
                    "type": "context_change",
                    "trigger_type": "action",
                    "trigger_value": "flow_started",
                },
            ],
            steps=[
                DialogueStep(id="msg", type=StepType.MESSAGE, content="Hi!"),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert len(result.context_changes) >= 1
        assert result.context_changes[0]["trigger_value"] == "flow_started"


# ══════════════════════════════════════════════════════
#  MEMORY OVERLAY INJECTION TESTS
# ══════════════════════════════════════════════════════

class TestMemoryOverlay:

    @pytest.mark.asyncio
    async def test_memory_available_in_templates(self, executor, base_context, memory_store):
        # Pre-store a memory
        memory_store.remember(
            "conversation", "conv1",
            key="previous_outcome", value="promised_payment",
        )

        flow = DialogueFlow(
            id="overlay", name="Overlay Test",
            steps=[
                DialogueStep(id="msg", type=StepType.MESSAGE,
                             content="Previous: {{memory.previous_outcome}}"),
            ],
        )
        result = await executor.execute(flow, base_context)
        assert "promised_payment" in result.rendered_message


# ══════════════════════════════════════════════════════
#  NEW STEP TYPE IN MODELS
# ══════════════════════════════════════════════════════

class TestNewStepTypes:

    def test_call_flow_step_type(self):
        assert StepType.CALL_FLOW.value == "call_flow"

    def test_loop_step_type(self):
        assert StepType.LOOP.value == "loop"

    def test_parallel_step_type(self):
        assert StepType.PARALLEL.value == "parallel"

    def test_remember_step_type(self):
        assert StepType.REMEMBER.value == "remember"

    def test_dialogue_step_new_fields(self):
        step = DialogueStep(
            id="test", type=StepType.CALL_FLOW,
            sub_flow_id="sub_flow_1",
            sub_flow_input={"var": "source.path"},
            sub_flow_output={"result": "output.key"},
        )
        assert step.sub_flow_id == "sub_flow_1"
        assert step.sub_flow_input == {"var": "source.path"}

    def test_flow_lifecycle_hooks(self):
        flow = DialogueFlow(
            id="test", name="Test",
            on_enter=[{"type": "log", "message": "enter"}],
            on_exit=[{"type": "log", "message": "exit"}],
            on_error=[{"type": "log", "message": "error"}],
            objective="Test objective",
            success_criteria=["complete"],
            composable=True,
            steps=[],
        )
        assert len(flow.on_enter) == 1
        assert flow.objective == "Test objective"
        assert flow.composable is True

    def test_flow_execution_result_new_fields(self):
        result = FlowExecutionResult(
            flow_id="f1", flow_name="Test",
            is_dynamic=True,
            replan_count=1,
            memories_created=[{"key": "test"}],
            sub_flow_results=[{"sub_flow_id": "sub1"}],
        )
        assert result.is_dynamic is True
        assert result.replan_count == 1
        assert len(result.memories_created) == 1


# ══════════════════════════════════════════════════════
#  LLM ENTITY EXTRACTION TESTS
# ══════════════════════════════════════════════════════

class TestLLMExtraction:

    @pytest.mark.asyncio
    async def test_llm_classify_intent_fallback(self):
        """Without LLM, falls back to keyword matching."""
        result = await llm_classify_intent(
            "Yes I will pay", ["confirmed", "denied"],
        )
        assert result["matched_intent"] in ("confirmed", "unknown", "payment_promise")

    @pytest.mark.asyncio
    async def test_llm_classify_intent_with_mock_llm(self):
        """With LLM, uses structured classification."""
        mock_llm = AsyncMock(return_value='{"matched_intent": "payment_confirmed", "confidence": 0.95, "reasoning": "explicit confirmation"}')
        result = await llm_classify_intent(
            "I already paid yesterday",
            ["payment_confirmed", "payment_promised"],
            llm_generate=mock_llm,
        )
        assert result["matched_intent"] == "payment_confirmed"
        assert result["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_llm_extract_entities_fallback(self):
        """Without LLM, falls back to regex."""
        result = await llm_extract_entities(
            "I paid ₹5000 on 15/01/2025",
            ["amount", "date"],
        )
        assert "amount" in result
        assert result["amount"] == 5000.0

    @pytest.mark.asyncio
    async def test_llm_extract_entities_with_mock_llm(self):
        """With LLM, handles complex cases."""
        mock_llm = AsyncMock(return_value='{"amount": 2500, "date": "2025-01-21"}')
        result = await llm_extract_entities(
            "about two and a half thousand, by next Tuesday",
            ["amount", "date"],
            llm_generate=mock_llm,
        )
        assert result["amount"] == 2500
        assert result["date"] == "2025-01-21"

    @pytest.mark.asyncio
    async def test_llm_classify_handles_error(self):
        """LLM error falls back to keyword matching."""
        mock_llm = AsyncMock(side_effect=Exception("API error"))
        result = await llm_classify_intent(
            "yes confirmed",
            ["confirmed", "denied"],
            llm_generate=mock_llm,
        )
        # Should still return a result from keyword fallback
        assert "matched_intent" in result


# ══════════════════════════════════════════════════════
#  PLANNER TESTS
# ══════════════════════════════════════════════════════

class TestAgenticPlanner:

    @pytest.fixture
    def planner(self, tool_registry, flow_registry, memory_store):
        mock_llm = AsyncMock()
        return AgenticPlanner(
            llm_generate=mock_llm,
            tool_registry=tool_registry,
            flow_registry=flow_registry,
            memory_store=memory_store,
        )

    def test_plan_objective(self):
        obj = PlanObjective(
            goal="Collect payment",
            process_type="payment_collection",
            current_state="outreach_1",
        )
        assert obj.goal == "Collect payment"
        assert obj.max_steps == 12

    @pytest.mark.asyncio
    async def test_suggest_next_action(self, planner, base_context):
        planner._llm.return_value = '{"action": "send_message", "reasoning": "test", "priority": "medium"}'
        suggestion = await planner.suggest_next_action(
            base_context, process_type="payment_collection",
        )
        assert suggestion["action"] == "send_message"

    @pytest.mark.asyncio
    async def test_suggest_handles_error(self, planner, base_context):
        planner._llm.side_effect = Exception("LLM error")
        suggestion = await planner.suggest_next_action(base_context)
        assert suggestion["action"] == "send_message"  # fallback

    @pytest.mark.asyncio
    async def test_plan_and_execute_success(self, planner, base_context):
        # Mock LLM to return a simple plan
        plan_json = '[{"id": "greet", "type": "message", "content": "Hello {{contact.name}}!"}]'
        planner._llm.return_value = plan_json

        executor = DialogueExecutor(flow_registry=planner._flows, memory_store=planner._memory)
        objective = PlanObjective(goal="Greet the contact")

        result = await planner.plan_and_execute(
            objective, base_context, ChannelType.CHAT, executor,
        )
        assert result.status == "completed"
        assert result.flow_result is not None
        assert "Hello Alice" in result.flow_result.rendered_message

    @pytest.mark.asyncio
    async def test_plan_and_execute_no_plan(self, planner, base_context):
        planner._llm.return_value = "invalid json"
        executor = DialogueExecutor()
        objective = PlanObjective(goal="Do something")

        result = await planner.plan_and_execute(
            objective, base_context, ChannelType.CHAT, executor,
        )
        assert result.status == "no_plan"

    @pytest.mark.asyncio
    async def test_plan_parses_complex_steps(self, planner, base_context):
        plan_json = '''[
            {"id": "check", "type": "tool_call", "tool": {"endpoint": "$get_attempt_summary", "result_key": "info"}},
            {"id": "branch", "type": "branch", "arms": [
                {"conditions": [{"field": "info.is_first_attempt", "operator": "eq", "value": true}], "goto": "first"},
                {"goto": "followup"}
            ]},
            {"id": "first", "type": "message", "content": "First contact!"},
            {"id": "followup", "type": "message", "content": "Following up!"}
        ]'''
        planner._llm.return_value = plan_json

        executor = DialogueExecutor(flow_registry=planner._flows, memory_store=planner._memory)
        objective = PlanObjective(goal="Payment reminder")

        result = await planner.plan_and_execute(
            objective, base_context, ChannelType.CHAT, executor,
        )
        assert result.status == "completed"
        assert result.plan_used is not None
        assert len(result.plan_used.steps) == 4
        assert "planner-generated" in result.plan_used.tags

    @pytest.mark.asyncio
    async def test_plan_accumulates_memories(self, planner, base_context, memory_store):
        plan_json = '[{"id": "msg", "type": "message", "content": "Hi!"}]'
        planner._llm.return_value = plan_json

        executor = DialogueExecutor(flow_registry=planner._flows, memory_store=memory_store)
        objective = PlanObjective(goal="Test", process_type="test")

        result = await planner.plan_and_execute(
            objective, base_context, ChannelType.CHAT, executor,
        )
        # Should have stored at least one memory (last_plan)
        assert len(result.memories_created) > 0


# ══════════════════════════════════════════════════════
#  INTEGRATION: FULL AGENTIC FLOW EXECUTION
# ══════════════════════════════════════════════════════

class TestAgenticFlowIntegration:

    @pytest.mark.asyncio
    async def test_full_agentic_flow(self, flow_registry, memory_store):
        """End-to-end: register flow, execute with memory, verify results."""

        # Register a flow that uses multiple agentic features
        flow = DialogueFlow(
            id="full_agentic",
            name="Full Agentic Test",
            on_enter=[
                {"type": "remember", "key": "test_started", "value": "yes", "scope": "conversation"},
            ],
            on_exit=[
                {"type": "remember", "key": "test_completed", "value": "yes", "scope": "conversation"},
            ],
            variables=[
                VariableDef(name="name", source="contact.name", default="user"),
            ],
            steps=[
                # Parallel data gathering
                DialogueStep(
                    id="gather", type=StepType.PARALLEL,
                    parallel_tools=[
                        ToolDef(endpoint="$get_attempt_summary", result_key="attempts"),
                        ToolDef(endpoint="$check_business_hours", result_key="hours"),
                    ],
                ),
                # Branch on hours
                DialogueStep(
                    id="check", type=StepType.BRANCH,
                    arms=[
                        BranchArm(
                            conditions=[
                                RuleCondition(field="hours.is_business_hours", operator="eq", value=True),
                            ],
                            goto="proceed",
                        ),
                        BranchArm(goto="off_hours"),
                    ],
                ),
                # Business hours path
                DialogueStep(
                    id="proceed", type=StepType.MESSAGE,
                    content="Hello {{name}}, attempt #{{attempts.attempt_count}}!",
                    next="remember_interaction",
                ),
                # Off hours path
                DialogueStep(
                    id="off_hours", type=StepType.MESSAGE,
                    content="We'll contact you during business hours, {{name}}.",
                    next="remember_interaction",
                ),
                # Remember the interaction
                DialogueStep(
                    id="remember_interaction", type=StepType.REMEMBER,
                    remember_key="last_greeting",
                    remember_value="greeted {{name}}",
                    remember_type="event",
                    remember_scope="conversation",
                    next="call_sub",
                ),
                # Call a sub-flow
                DialogueStep(
                    id="call_sub", type=StepType.CALL_FLOW,
                    sub_flow_id="sub_greet",
                    sub_flow_input={"contact_name": "contact.name"},
                ),
            ],
        )
        flow_registry.register(flow)

        executor = DialogueExecutor(
            flow_registry=flow_registry,
            memory_store=memory_store,
        )

        context = {
            "contact": {"id": "c_test", "name": "Bob"},
            "conversation_id": "conv_test",
            "business_data": {},
        }

        result = await executor.execute(flow, context, ChannelType.CHAT)

        # Verify execution completed
        assert result.status == "completed"
        assert "Bob" in result.rendered_message

        # Verify hooks fired
        assert len(result.hook_results) >= 2  # on_enter + on_exit

        # Verify memories stored
        assert len(result.memories_created) >= 1
        entry = memory_store.recall("conversation", "conv_test", "last_greeting")
        assert entry is not None
        assert "Bob" in entry.value

        # Verify on_enter memory
        started = memory_store.recall("conversation", "conv_test", "test_started")
        assert started is not None
        assert started.value == "yes"

        # Verify sub-flow was called
        assert any(s.step_type == "call_flow" for s in result.steps_executed)
        assert "Hello from sub-flow" in result.rendered_message
