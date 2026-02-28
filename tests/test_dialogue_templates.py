"""Tests for the agentic dialogue template system."""
import pytest
from models.schemas import ChannelType, RuleCondition
from templates.models import (
    DialogueFlow, DialogueStep, StepType, ChannelVariant,
    VariableDef, BranchArm, ToolDef, FlowExecutionResult,
)
from templates.registry import DialogueFlowRegistry
from templates.executor import DialogueExecutor


# ──────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────

@pytest.fixture
def simple_flow() -> DialogueFlow:
    """A basic flow with message steps."""
    return DialogueFlow(
        id="test_greeting",
        name="Test Greeting",
        variables=[
            VariableDef(name="contact_name", source="contact.name", default="there"),
            VariableDef(name="amount", source="business_data.amount", format="currency", default=0),
        ],
        steps=[
            DialogueStep(id="greeting", type=StepType.MESSAGE,
                         content="Hi {{contact_name}}!"),
            DialogueStep(id="body", type=StepType.MESSAGE,
                         content="Your outstanding amount is {{amount}}."),
            DialogueStep(id="closing", type=StepType.MESSAGE,
                         content="Please let us know if you have questions."),
        ],
    )


@pytest.fixture
def branching_flow() -> DialogueFlow:
    """A flow with tool call + branch + gate steps."""
    return DialogueFlow(
        id="test_branching",
        name="Branching Flow",
        variables=[
            VariableDef(name="contact_name", source="contact.name", default="there"),
        ],
        steps=[
            DialogueStep(id="check_status", type=StepType.TOOL_CALL,
                         tool=ToolDef(endpoint="$get_attempt_summary",
                                      result_key="attempt_info")),
            DialogueStep(id="route", type=StepType.BRANCH,
                         arms=[
                             BranchArm(conditions=[
                                 RuleCondition(field="attempt_info.is_first_attempt",
                                               operator="eq", value=True),
                             ], goto="first_msg", description="First attempt"),
                             BranchArm(goto="followup_msg", description="Follow-up"),
                         ]),
            DialogueStep(id="first_msg", type=StepType.MESSAGE,
                         content="Hello {{contact_name}}, this is our first contact."),
            DialogueStep(id="followup_msg", type=StepType.MESSAGE,
                         content="Hi {{contact_name}}, following up on our previous conversation."),
        ],
    )


@pytest.fixture
def gated_flow() -> DialogueFlow:
    """A flow with a gate that can abort."""
    return DialogueFlow(
        id="test_gated",
        name="Gated Flow",
        steps=[
            DialogueStep(id="check_hours", type=StepType.GATE,
                         conditions=[
                             RuleCondition(field="is_business_hours",
                                           operator="eq", value=True),
                         ],
                         fail_goto="off_hours"),
            DialogueStep(id="proceed", type=StepType.MESSAGE,
                         content="We are open! How can I help?"),
            DialogueStep(id="off_hours", type=StepType.MESSAGE,
                         content="We are closed. We'll contact you during business hours."),
        ],
    )


@pytest.fixture
def channel_flow() -> DialogueFlow:
    """A flow with channel-specific variants."""
    return DialogueFlow(
        id="test_channel",
        name="Channel Aware",
        steps=[
            DialogueStep(id="main", type=StepType.MESSAGE,
                         content="Default message.",
                         channel_variants=[
                             ChannelVariant(channel=ChannelType.WHATSAPP,
                                            content_override="Short WA message 📱"),
                             ChannelVariant(channel=ChannelType.EMAIL,
                                            content_override="Dear valued customer,\n\nThis is the email version."),
                             ChannelVariant(channel=ChannelType.VOICE, skip=True),
                         ]),
            DialogueStep(id="footer", type=StepType.MESSAGE,
                         content="Reply to this message for help.",
                         channel_variants=[
                             ChannelVariant(channel=ChannelType.VOICE, skip=True),
                         ]),
        ],
    )


@pytest.fixture
def action_flow() -> DialogueFlow:
    """A flow that emits context changes."""
    return DialogueFlow(
        id="test_action",
        name="Action Flow",
        steps=[
            DialogueStep(id="msg", type=StepType.MESSAGE,
                         content="Processing your request..."),
            DialogueStep(id="emit", type=StepType.ACTION,
                         action_type="context_change",
                         action_config={
                             "trigger_type": "action",
                             "trigger_value": "outreach_sent",
                             "extra_data": {"source": "dialogue_flow"},
                         }),
        ],
    )


@pytest.fixture
def executor() -> DialogueExecutor:
    """Executor with no LLM (tests message/branch/gate/tool/action steps)."""
    return DialogueExecutor()


@pytest.fixture
def registry(simple_flow, branching_flow, gated_flow) -> DialogueFlowRegistry:
    reg = DialogueFlowRegistry()
    reg.register(simple_flow)
    reg.register(branching_flow)
    reg.register(gated_flow)
    return reg


# ──────────────────────────────────────────────────────
#  Registry Tests
# ──────────────────────────────────────────────────────

class TestDialogueFlowRegistry:
    def test_register_and_get(self, registry):
        assert registry.get("test_greeting") is not None
        assert registry.get("nonexistent") is None

    def test_list_all(self, registry):
        assert len(registry.list_all()) == 3

    def test_resolve_exact(self, registry):
        flow = registry.resolve(template_name="test_greeting")
        assert flow.id == "test_greeting"

    def test_resolve_by_tag(self, registry):
        # Add a tagged flow
        tagged = DialogueFlow(
            id="tagged_flow", name="Tagged",
            tags=["process:payment_collection", "state:reminded"],
            steps=[DialogueStep(id="s1", type=StepType.MESSAGE, content="hi")],
        )
        registry.register(tagged)
        flow = registry.resolve(process_type="payment_collection", current_state="reminded")
        assert flow.id == "tagged_flow"

    def test_resolve_with_guards(self, registry):
        guarded = DialogueFlow(
            id="high_value", name="High Value",
            tags=["process:payment_collection"],
            guards=[RuleCondition(field="amount", operator="gt", value=100000)],
            priority=20,
            steps=[DialogueStep(id="s1", type=StepType.MESSAGE, content="VIP")],
        )
        low_value = DialogueFlow(
            id="standard_payment", name="Standard",
            tags=["process:payment_collection"],
            priority=5,
            steps=[DialogueStep(id="s1", type=StepType.MESSAGE, content="Standard")],
        )
        registry.register(guarded)
        registry.register(low_value)

        # High value context — guard passes, gets high_value
        flow = registry.resolve(
            process_type="payment_collection",
            context_data={"amount": 500000},
        )
        assert flow.id == "high_value"

        # Low value context — guard fails on high_value, falls to standard
        flow = registry.resolve(
            process_type="payment_collection",
            context_data={"amount": 5000},
        )
        assert flow is not None
        assert flow.id == "standard_payment"

    def test_validation_rejects_broken_refs(self):
        reg = DialogueFlowRegistry()
        bad_flow = DialogueFlow(
            id="bad", name="Bad",
            steps=[
                DialogueStep(id="s1", type=StepType.BRANCH,
                             arms=[BranchArm(goto="nonexistent")]),
            ],
        )
        with pytest.raises(ValueError, match="nonexistent"):
            reg.register(bad_flow)

    def test_register_from_config(self):
        reg = DialogueFlowRegistry()
        config = [{
            "id": "config_flow",
            "name": "From Config",
            "steps": [
                {"id": "step1", "type": "message", "content": "Hello {{name}}"},
                {"id": "step2", "type": "branch", "arms": [
                    {"goto": "step1", "description": "loop back"},
                ]},
            ],
            "variables": [{"name": "name", "source": "contact.name"}],
            "tags": ["process:test"],
        }]
        reg.register_from_config(config)
        assert reg.get("config_flow") is not None


# ──────────────────────────────────────────────────────
#  Executor Tests
# ──────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestDialogueExecutor:

    async def test_simple_message_flow(self, executor, simple_flow):
        ctx = {
            "contact": {"name": "Rajesh Kumar"},
            "business_data": {"amount": 250000},
        }
        result = await executor.execute(simple_flow, ctx, ChannelType.CHAT)

        assert result.status == "completed"
        assert "Rajesh Kumar" in result.rendered_message
        assert "₹2,50,000.00" in result.rendered_message
        assert len(result.steps_executed) == 3

    async def test_variable_defaults(self, executor, simple_flow):
        ctx = {}  # no contact or business data
        result = await executor.execute(simple_flow, ctx, ChannelType.CHAT)
        assert "there" in result.rendered_message  # default contact name
        assert result.status == "completed"

    async def test_branching_first_attempt(self, executor, branching_flow):
        ctx = {"contact": {"name": "Priya"}, "attempt_count": 0, "max_attempts": 5}
        result = await executor.execute(branching_flow, ctx, ChannelType.CHAT)
        assert "first contact" in result.rendered_message
        assert result.status == "completed"

    async def test_branching_followup(self, executor, branching_flow):
        ctx = {"contact": {"name": "Amit"}, "attempt_count": 2, "max_attempts": 5}
        result = await executor.execute(branching_flow, ctx, ChannelType.CHAT)
        assert "following up" in result.rendered_message

    async def test_gate_passes(self, executor, gated_flow):
        ctx = {"is_business_hours": True}
        result = await executor.execute(gated_flow, ctx, ChannelType.CHAT)
        assert "We are open" in result.rendered_message

    async def test_gate_fails_redirects(self, executor, gated_flow):
        ctx = {"is_business_hours": False}
        result = await executor.execute(gated_flow, ctx, ChannelType.CHAT)
        assert "closed" in result.rendered_message

    async def test_gate_fails_aborts(self, executor):
        """Gate with no fail_goto should abort the flow."""
        flow = DialogueFlow(
            id="abort_flow", name="Abort",
            steps=[
                DialogueStep(id="gate", type=StepType.GATE,
                             conditions=[
                                 RuleCondition(field="ready", operator="eq", value=True),
                             ]),
                DialogueStep(id="msg", type=StepType.MESSAGE, content="Proceed"),
            ],
        )
        result = await executor.execute(flow, {"ready": False}, ChannelType.CHAT)
        assert result.status == "aborted"
        assert result.rendered_message == ""

    async def test_channel_variant_whatsapp(self, executor, channel_flow):
        result = await executor.execute(channel_flow, {}, ChannelType.WHATSAPP)
        assert "Short WA message" in result.rendered_message
        assert "📱" in result.rendered_message

    async def test_channel_variant_email(self, executor, channel_flow):
        result = await executor.execute(channel_flow, {}, ChannelType.EMAIL)
        assert "Dear valued customer" in result.rendered_message

    async def test_channel_skip(self, executor, channel_flow):
        result = await executor.execute(channel_flow, {}, ChannelType.VOICE)
        # Both steps are skipped for voice
        assert result.rendered_message == ""

    async def test_action_step_emits_context_change(self, executor, action_flow):
        result = await executor.execute(action_flow, {}, ChannelType.CHAT)
        assert result.status == "completed"
        assert len(result.context_changes) == 1
        assert result.context_changes[0]["trigger_value"] == "outreach_sent"

    async def test_builtin_tool_call(self, executor):
        flow = DialogueFlow(
            id="tool_test", name="Tool Test",
            steps=[
                DialogueStep(id="check", type=StepType.TOOL_CALL,
                             tool=ToolDef(endpoint="$get_attempt_summary",
                                          result_key="attempt_info")),
                DialogueStep(id="branch", type=StepType.BRANCH,
                             arms=[
                                 BranchArm(
                                     conditions=[
                                         RuleCondition(field="attempt_info.is_first_attempt",
                                                       operator="eq", value=True),
                                     ],
                                     goto="first"),
                                 BranchArm(goto="retry"),
                             ]),
                DialogueStep(id="first", type=StepType.MESSAGE,
                             content="First attempt!"),
                DialogueStep(id="retry", type=StepType.MESSAGE,
                             content="Retrying..."),
            ],
        )
        result = await executor.execute(
            flow, {"attempt_count": 0, "max_attempts": 5}, ChannelType.CHAT,
        )
        assert "First attempt" in result.rendered_message

    async def test_collect_step_returns_waiting(self, executor):
        flow = DialogueFlow(
            id="collect_test", name="Collect Test",
            steps=[
                DialogueStep(id="ask", type=StepType.MESSAGE,
                             content="Do you confirm?"),
                DialogueStep(id="wait", type=StepType.COLLECT,
                             expected_intents=["confirmed", "denied"],
                             timeout_seconds=3600),
            ],
        )
        result = await executor.execute(flow, {}, ChannelType.CHAT)
        assert result.status == "waiting"
        assert "Do you confirm" in result.rendered_message

    async def test_explicit_next_step(self, executor):
        flow = DialogueFlow(
            id="jump_test", name="Jump",
            steps=[
                DialogueStep(id="start", type=StepType.MESSAGE,
                             content="Start.", next="end"),
                DialogueStep(id="skip_me", type=StepType.MESSAGE,
                             content="This should not appear."),
                DialogueStep(id="end", type=StepType.MESSAGE,
                             content="End."),
            ],
        )
        result = await executor.execute(flow, {}, ChannelType.CHAT)
        assert "Start." in result.rendered_message
        assert "End." in result.rendered_message
        assert "should not appear" not in result.rendered_message

    async def test_max_steps_prevents_infinite_loop(self, executor):
        flow = DialogueFlow(
            id="loop_test", name="Loop",
            steps=[
                DialogueStep(id="a", type=StepType.MESSAGE,
                             content="A", next="b"),
                DialogueStep(id="b", type=StepType.MESSAGE,
                             content="B", next="a"),
            ],
        )
        result = await executor.execute(flow, {}, ChannelType.CHAT)
        assert result.status == "completed"
        assert len(result.steps_executed) == 50  # MAX_STEPS

    async def test_generate_without_llm_falls_back(self, executor):
        """Without LLM, generate step returns user_prompt as content."""
        flow = DialogueFlow(
            id="gen_test", name="Gen",
            steps=[
                DialogueStep(id="gen", type=StepType.GENERATE,
                             system_prompt="You are helpful.",
                             user_prompt="Please confirm order {{order_id}}.",
                             content="Fallback content."),
            ],
        )
        result = await executor.execute(
            flow, {"order_id": "ORD-123"}, ChannelType.CHAT,
        )
        assert "ORD-123" in result.rendered_message

    async def test_set_variable_action(self, executor):
        flow = DialogueFlow(
            id="setvar_test", name="SetVar",
            steps=[
                DialogueStep(id="set_greeting", type=StepType.ACTION,
                             action_type="set_variable",
                             action_config={"key": "greeting", "value": "Hello {{contact_name}}!"}),
                DialogueStep(id="use_it", type=StepType.MESSAGE,
                             content="{{greeting}} How are you?"),
            ],
        )
        result = await executor.execute(
            flow, {"contact_name": "Rajesh"}, ChannelType.CHAT,
        )
        assert "Hello Rajesh!" in result.rendered_message

    async def test_message_replace_mode(self, executor):
        """append=False should replace the buffer, not append."""
        flow = DialogueFlow(
            id="replace_test", name="Replace",
            steps=[
                DialogueStep(id="draft", type=StepType.MESSAGE,
                             content="Draft message."),
                DialogueStep(id="replace", type=StepType.MESSAGE,
                             content="Final message only.", append=False),
            ],
        )
        result = await executor.execute(flow, {}, ChannelType.CHAT)
        assert "Draft" not in result.rendered_message
        assert "Final message only." in result.rendered_message


# ──────────────────────────────────────────────────────
#  Format Tests
# ──────────────────────────────────────────────────────

class TestVariableFormatting:
    def test_currency_format(self):
        ex = DialogueExecutor()
        assert "₹2,50,000.00" in str(ex._format_value(250000, "currency"))
        assert "₹500.00" in str(ex._format_value(500, "currency"))

    def test_capitalize(self):
        ex = DialogueExecutor()
        assert ex._format_value("rajesh kumar", "capitalize") == "Rajesh Kumar"

    def test_upper(self):
        ex = DialogueExecutor()
        assert ex._format_value("hello", "upper") == "HELLO"

    def test_no_format(self):
        ex = DialogueExecutor()
        assert ex._format_value("hello", "") == "hello"
        assert ex._format_value(42, "") == 42
