"""
Dialogue Flow Models — Agentic conversation templates.

A DialogueFlow is NOT a static text blob. It's a directed plan of execution
steps that an agent runs through, making real-time decisions at each node.

Each step can:
  - Render text with variable interpolation (message)
  - Generate content via LLM with structured prompts (generate)
  - Call a backend tool and use the result in subsequent steps (tool_call)
  - Branch to different paths based on conditions (branch)
  - Collect + interpret user input (collect)
  - Trigger state machine transitions or system actions (action)
  - Guard against continuing if a condition fails (gate)

The executor runs the plan step by step, resolving variables from live
conversation context, and produces a final rendered message.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from models.schemas import ChannelType, FollowUpPriority, RuleCondition


# ──────────────────────────────────────────────────────────────
#  Step Types
# ──────────────────────────────────────────────────────────────

class StepType(str, Enum):
    """What kind of work a dialogue step does."""
    MESSAGE = "message"           # Render + queue a message fragment
    GENERATE = "generate"         # LLM generates content with structured prompt
    TOOL_CALL = "tool_call"       # Call backend/internal tool, store result
    BRANCH = "branch"             # Conditional routing to different steps
    LLM_ROUTE = "llm_route"      # LLM decides which branch to take (agentic routing)
    COLLECT = "collect"           # Pause for user input, extract entities
    ACTION = "action"             # Fire a state-machine context change
    GATE = "gate"                 # Continue only if condition is met
    CALL_FLOW = "call_flow"       # Invoke a sub-flow by id and merge results
    LOOP = "loop"                 # Repeat steps until condition met or max iterations
    PARALLEL = "parallel"         # Run multiple tool calls concurrently
    REMEMBER = "remember"         # Store a fact/observation in flow memory


# ──────────────────────────────────────────────────────────────
#  Channel Variant — per-channel rendering overrides
# ──────────────────────────────────────────────────────────────

class ChannelVariant(BaseModel):
    """Channel-specific rendering instructions."""
    channel: ChannelType
    tone: str = ""                                # e.g. "formal", "concise", "conversational"
    max_length: int = 0                           # character limit (0 = unlimited)
    content_override: str = ""                    # if set, replaces the step content entirely
    system_prompt_suffix: str = ""                # appended to LLM system prompt for generate steps
    skip: bool = False                            # skip this step entirely on this channel


# ──────────────────────────────────────────────────────────────
#  Variable Definition — how to pull data from context
# ──────────────────────────────────────────────────────────────

class VariableDef(BaseModel):
    """
    Declares a variable that the template needs.
    The executor resolves it from the live context before rendering.
    """
    name: str                                     # template variable name (e.g. "amount")
    source: str                                   # dot-path into context (e.g. "business_data.amount")
    format: str = ""                              # format hint: "currency", "date", "phone", "capitalize"
    default: Any = ""                             # fallback value if source is missing
    required: bool = False                        # if True, abort the flow when missing


# ──────────────────────────────────────────────────────────────
#  Branch Arm — one possible route in a branch step
# ──────────────────────────────────────────────────────────────

class BranchArm(BaseModel):
    """One conditional arm in a branch step."""
    conditions: list[RuleCondition] = []          # conditions to evaluate (empty = default arm)
    goto: str                                     # step_id to jump to
    description: str = ""


# ──────────────────────────────────────────────────────────────
#  Tool Definition — what to call and how to use the result
# ──────────────────────────────────────────────────────────────

class ToolDef(BaseModel):
    """
    A tool the dialogue executor can invoke mid-flow.

    endpoint can be:
      - A backend connector endpoint name (e.g. "check_payment_status")
      - A built-in function (e.g. "$check_business_hours", "$get_last_message_time")
    """
    endpoint: str                                 # backend endpoint or built-in function
    payload_template: dict[str, str] = {}         # keys→dot-paths resolved from context
    result_key: str = ""                          # store result under this key in flow context
    timeout_seconds: int = 10


# ──────────────────────────────────────────────────────────────
#  Dialogue Step — one node in the execution plan
# ──────────────────────────────────────────────────────────────

class DialogueStep(BaseModel):
    """
    A single step in the dialogue flow.

    Every step has an id and type. Depending on type, different
    fields are relevant:

    message:  content (with {{var}} interpolation)
    generate: system_prompt, user_prompt, constraints
    tool_call: tool
    branch:   arms (list of BranchArm)
    llm_route: system_prompt, arms (LLM picks which arm to take)
    collect:  expected_intents, entity_extraction, timeout_seconds
    action:   action_type, action_config
    gate:     conditions (list of RuleCondition), fail_goto
    """
    id: str
    type: StepType
    description: str = ""

    # ── message fields ────────────────────────────────
    content: str = ""                              # text with {{variable}} placeholders
    append: bool = True                            # append to message buffer (True) or replace

    # ── generate fields ───────────────────────────────
    system_prompt: str = ""                        # system prompt for LLM
    user_prompt: str = ""                          # user prompt (can reference {{vars}})
    constraints: dict[str, Any] = {}               # max_tokens, temperature, etc.

    # ── tool_call fields ──────────────────────────────
    tool: Optional[ToolDef] = None

    # ── branch fields ─────────────────────────────────
    arms: list[BranchArm] = []

    # ── collect fields ────────────────────────────────
    expected_intents: list[str] = []               # intents we're waiting for
    entity_extraction: list[str] = []              # entities to extract from response
    timeout_seconds: int = 0                       # how long to wait (0 = indefinite)
    timeout_goto: str = ""                         # step to jump to on timeout

    # ── action fields ─────────────────────────────────
    action_type: str = ""                          # context_change | update_binding | emit_event
    action_config: dict[str, Any] = {}

    # ── gate fields ───────────────────────────────────
    conditions: list[RuleCondition] = []
    fail_goto: str = ""                            # step to jump to on gate failure (empty = abort)

    # ── call_flow fields ──────────────────────────────
    sub_flow_id: str = ""                          # id of the sub-flow to invoke
    sub_flow_input: dict[str, str] = {}            # variable mapping: sub_flow_var → source_path
    sub_flow_output: dict[str, str] = {}           # result mapping: ctx_key → sub_flow result key

    # ── loop fields ───────────────────────────────────
    loop_steps: list[str] = []                     # step ids to repeat
    loop_condition: list[RuleCondition] = []       # continue looping while this is true
    loop_max_iterations: int = 3                   # safety limit
    loop_exit_goto: str = ""                       # step to jump to when loop ends

    # ── parallel fields ───────────────────────────────
    parallel_tools: list[ToolDef] = []             # tools to call concurrently
    parallel_merge_key: str = ""                   # store merged results under this key

    # ── remember fields ───────────────────────────────
    remember_key: str = ""                         # memory key to store
    remember_value: str = ""                       # value (supports {{var}} interpolation)
    remember_type: str = "fact"                    # fact | preference | event | observation
    remember_scope: str = "conversation"           # contact | conversation | entity

    # ── flow control ──────────────────────────────────
    next: str = ""                                 # explicit next step (overrides sequential)
    channel_variants: list[ChannelVariant] = []    # per-channel overrides for this step


# ──────────────────────────────────────────────────────────────
#  Dialogue Flow — the complete agentic template
# ──────────────────────────────────────────────────────────────

class DialogueFlow(BaseModel):
    """
    A complete agentic conversation template.

    This is NOT a static text blob. It's a plan that the executor runs
    through, making real-time decisions at each step based on live
    context, backend data, and LLM reasoning.

    Example:
        payment_overdue_reminder:
          1. GATE: check business hours → if off-hours, delay
          2. TOOL_CALL: check_payment_status → store result
          3. BRANCH: if already_paid → goto thank_you; if partial → goto partial_ack
          4. GENERATE: LLM produces contextual reminder based on history + overdue days
          5. MESSAGE: append urgency footer based on escalation level
          6. ACTION: emit "outreach_sent" context change
    """
    id: str
    name: str
    description: str = ""
    version: str = "1.0"

    # ── Variables ─────────────────────────────────────
    variables: list[VariableDef] = []              # declared variables resolved from context

    # ── Guards ────────────────────────────────────────
    guards: list[RuleCondition] = []               # conditions that must be true to use this flow
    priority: int = 0                              # higher priority flows are preferred when multiple match

    # ── Channel Defaults ──────────────────────────────
    default_channel_config: dict[str, ChannelVariant] = {}  # channel_type → defaults for all steps

    # ── Steps ─────────────────────────────────────────
    steps: list[DialogueStep] = []
    entry_step: str = ""                           # starting step id (default: first step)

    # ── Lifecycle Hooks ───────────────────────────────
    on_enter: list[dict[str, Any]] = []            # actions to run when flow starts
    on_exit: list[dict[str, Any]] = []             # actions to run when flow completes
    on_error: list[dict[str, Any]] = []            # actions to run on failure
    on_collect_timeout: list[dict[str, Any]] = []  # actions when a collect step times out

    # ── Planning Metadata ─────────────────────────────
    objective: str = ""                            # What this flow aims to achieve (for planner)
    success_criteria: list[str] = []               # How to know the flow succeeded
    failure_fallback: str = ""                     # Flow id to run if this one fails
    composable: bool = True                        # Can the planner use this as a sub-flow?

    # ── Metadata ──────────────────────────────────────
    tags: list[str] = []                           # for filtering/search
    metadata: dict[str, Any] = {}

    @property
    def step_index(self) -> dict[str, DialogueStep]:
        return {s.id: s for s in self.steps}

    def get_entry(self) -> Optional[DialogueStep]:
        if self.entry_step:
            return self.step_index.get(self.entry_step)
        return self.steps[0] if self.steps else None


# ──────────────────────────────────────────────────────────────
#  Execution Result — what the executor produces
# ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Outcome of executing one dialogue step."""
    step_id: str
    step_type: str
    status: str = "ok"                             # ok | skipped | failed | waiting
    content: str = ""                              # rendered content (if message/generate)
    tool_result: dict[str, Any] = {}               # tool call result (if tool_call)
    branch_taken: str = ""                         # which arm was taken (if branch)
    action_result: dict[str, Any] = {}             # action outcome (if action)
    error: str = ""
    metadata: dict[str, Any] = {}


class FlowExecutionResult(BaseModel):
    """Complete result of executing a dialogue flow."""
    flow_id: str
    flow_name: str
    status: str = "completed"                      # completed | aborted | waiting | error
    rendered_message: str = ""                      # final composed message to send
    steps_executed: list[StepResult] = []
    variables_resolved: dict[str, Any] = {}
    context_changes: list[dict[str, Any]] = []     # context changes emitted during execution
    memories_created: list[dict[str, Any]] = []    # memories stored during execution
    sub_flow_results: list[dict[str, Any]] = []    # results from call_flow steps
    hook_results: list[dict[str, Any]] = []        # results from lifecycle hooks
    channel: Optional[ChannelType] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    is_dynamic: bool = False                       # True if generated by the planner
    replan_count: int = 0
    metadata: dict[str, Any] = {}
