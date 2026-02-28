"""
Agentic Planner — LLM-driven dynamic dialogue planning.

This is the "brain" that activates when no pre-defined dialogue flow
matches the current situation. Instead of falling back to raw LLM
generation, the planner:

  1. Reads the full context (contact, state, memory, conversation history)
  2. Enumerates available tools and sub-flows
  3. Asks the LLM to produce a *plan* — a sequence of typed steps
  4. Validates and executes the plan through the same executor pipeline
  5. Accumulates learnings into flow memory

This gives the system genuine agency: it can handle novel situations
by composing available capabilities on the fly, rather than being
limited to pre-authored flows.

The planner also supports:
  - Re-planning: if a step fails or produces unexpected results,
    the planner can adjust the remaining plan
  - Goal-directed execution: the planner works toward a stated objective
  - Constraint awareness: respects channel limits, business hours, etc.

Architecture:
  Orchestrator → AgenticPlanner.plan_and_execute(context, objective, channel)
    → LLM produces a plan (list of typed steps)
    → Plan is validated against tool registry + flow registry
    → DialogueExecutor runs the plan
    → Results are accumulated into flow memory
    → Final message is returned to orchestrator
"""
from __future__ import annotations

import json
import structlog
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from models.schemas import ChannelType, RuleCondition
from templates.models import (
    DialogueFlow, DialogueStep, StepType,
    VariableDef, BranchArm, ToolDef,
    FlowExecutionResult,
)
from templates.tool_registry import ToolRegistry
from templates.memory import FlowMemoryStore, MemoryType
from templates.registry import DialogueFlowRegistry

logger = structlog.get_logger()

# Maximum steps the planner can generate in a single plan
MAX_PLAN_STEPS = 12
# Maximum re-plan attempts
MAX_REPLANS = 2


class PlanObjective:
    """What the planner is trying to accomplish."""

    def __init__(
        self,
        goal: str,                                    # Natural language goal
        process_type: str = "",                       # Business process context
        current_state: str = "",                      # Current state in state machine
        constraints: dict[str, Any] = None,           # Hard constraints
        preferred_approach: str = "",                  # Hint from rules/config
        max_steps: int = MAX_PLAN_STEPS,
    ):
        self.goal = goal
        self.process_type = process_type
        self.current_state = current_state
        self.constraints = constraints or {}
        self.preferred_approach = preferred_approach
        self.max_steps = max_steps


class PlanResult:
    """Outcome of a planning + execution cycle."""

    def __init__(self):
        self.flow_result: Optional[FlowExecutionResult] = None
        self.plan_used: Optional[DialogueFlow] = None
        self.replans: int = 0
        self.memories_created: list[dict[str, Any]] = []
        self.status: str = "pending"                   # pending | executed | failed | no_plan


class AgenticPlanner:
    """
    LLM-driven planner that composes dialogue flows dynamically.

    The planner sits between the orchestrator and the executor:
    when no pre-defined flow matches, the planner asks the LLM to
    compose a plan from available tools and sub-flows.
    """

    def __init__(
        self,
        llm_generate: Callable,
        tool_registry: ToolRegistry,
        flow_registry: DialogueFlowRegistry,
        memory_store: FlowMemoryStore,
    ):
        self._llm = llm_generate
        self._tools = tool_registry
        self._flows = flow_registry
        self._memory = memory_store

    async def plan_and_execute(
        self,
        objective: PlanObjective,
        context: dict[str, Any],
        channel: ChannelType,
        executor,                                     # DialogueExecutor instance
    ) -> PlanResult:
        """
        Main entry: plan a dialogue flow and execute it.

        1. Build planning prompt with context + available capabilities
        2. LLM produces a structured plan
        3. Validate the plan
        4. Execute via the standard DialogueExecutor
        5. If a step fails, optionally re-plan
        6. Accumulate learnings into memory
        """
        result = PlanResult()

        # 1. Build the planning prompt
        prompt = self._build_planning_prompt(objective, context, channel)

        # 2. Ask LLM to produce a plan
        plan_flow = await self._generate_plan(prompt, objective, channel)
        if not plan_flow:
            result.status = "no_plan"
            return result

        result.plan_used = plan_flow

        # 3. Execute the plan
        flow_result = await executor.execute(plan_flow, context, channel)
        result.flow_result = flow_result

        # 4. Re-plan if needed
        replan_count = 0
        while (
            flow_result.status in ("aborted", "error")
            and replan_count < MAX_REPLANS
        ):
            replan_count += 1
            logger.info("replanning",
                         attempt=replan_count,
                         previous_status=flow_result.status)

            replan_prompt = self._build_replan_prompt(
                objective, context, channel, flow_result,
            )
            plan_flow = await self._generate_plan(replan_prompt, objective, channel)
            if not plan_flow:
                break

            flow_result = await executor.execute(plan_flow, context, channel)
            result.flow_result = flow_result
            result.plan_used = plan_flow

        result.replans = replan_count
        result.status = flow_result.status if flow_result else "failed"

        # 5. Accumulate learnings
        result.memories_created = await self._accumulate_learnings(
            objective, context, flow_result,
        )

        return result

    async def suggest_next_action(
        self,
        context: dict[str, Any],
        process_type: str = "",
        current_state: str = "",
    ) -> dict[str, Any]:
        """
        Ask the LLM what the agent should do next, without executing.
        Returns a structured suggestion for the orchestrator.
        """
        tools_desc = self._tools.describe_for_llm(process_type, current_state)
        flows_desc = self._describe_available_flows(process_type)
        memory_desc = self._get_memory_description(context)

        prompt = f"""You are an intelligent conversation agent planner.

Current situation:
- Process: {process_type or 'general'}
- State: {current_state or 'unknown'}
- Channel: {context.get('_channel', 'chat')}

{memory_desc}

Conversation context:
{json.dumps(context.get('business_context', {}), indent=2)}

{tools_desc}

{flows_desc}

Based on the current situation, what should the agent do next?
Respond with ONLY a JSON object:
{{
  "action": "send_message" | "use_tool" | "run_flow" | "wait" | "escalate" | "resolve",
  "reasoning": "brief explanation",
  "tool_name": "if action is use_tool",
  "flow_id": "if action is run_flow",
  "message_hint": "if action is send_message, a brief description of what to say",
  "priority": "low" | "medium" | "high" | "urgent"
}}"""

        try:
            response = await self._llm(
                "You are a conversation planning agent. Respond only with valid JSON.",
                prompt,
                {"max_tokens": 300, "temperature": 0.3},
            )
            text = response.strip()
            if text.startswith("```"):
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)
        except Exception as e:
            logger.error("suggest_next_action_failed", error=str(e))
            return {
                "action": "send_message",
                "reasoning": "Planning failed, defaulting to message generation",
                "priority": "medium",
            }

    # ══════════════════════════════════════════════════
    #  PLAN GENERATION
    # ══════════════════════════════════════════════════

    def _build_planning_prompt(
        self,
        objective: PlanObjective,
        context: dict[str, Any],
        channel: ChannelType,
    ) -> str:
        """Build the full planning prompt for the LLM."""
        tools_desc = self._tools.describe_for_llm(
            objective.process_type, objective.current_state,
        )
        flows_desc = self._describe_available_flows(objective.process_type)
        memory_desc = self._get_memory_description(context)
        channel_guidance = self._channel_guidance(channel)

        return f"""You are an intelligent conversation agent planner.
Your job is to compose a dialogue plan to achieve the given objective.

OBJECTIVE: {objective.goal}

CONTEXT:
- Process type: {objective.process_type or 'general'}
- Current state: {objective.current_state or 'unknown'}
- Channel: {channel.value}
{channel_guidance}

BUSINESS DATA:
{json.dumps(context.get('business_context', {}), indent=2)}

CONTACT:
{json.dumps(context.get('contact', {}), indent=2)}

{memory_desc}

CONVERSATION HISTORY:
{json.dumps(context.get('conversation_history', [])[-5:], indent=2)}

{tools_desc}

{flows_desc}

{f"PREFERRED APPROACH: {objective.preferred_approach}" if objective.preferred_approach else ""}

{f"CONSTRAINTS: {json.dumps(objective.constraints)}" if objective.constraints else ""}

STEP TYPES you can use:
  message:    Render text with {{{{variable}}}} placeholders
  generate:   Have the LLM generate contextual content (system_prompt + user_prompt)
  tool_call:  Call a registered tool (endpoint name from the tools list above)
  branch:     Conditional routing (conditions on context fields)
  gate:       Check a condition — abort or redirect if false
  action:     Emit a context_change or set_variable
  collect:    Pause and wait for user input (with expected_intents + entity_extraction)

Compose a plan as a JSON array of steps. Each step needs:
  {{"id": "unique_id", "type": "step_type", ...type-specific fields...}}

For MESSAGE steps:
  {{"id": "...", "type": "message", "content": "text with {{{{var}}}} placeholders"}}

For GENERATE steps:
  {{"id": "...", "type": "generate", "system_prompt": "...", "user_prompt": "..."}}

For TOOL_CALL steps:
  {{"id": "...", "type": "tool_call", "tool": {{"endpoint": "tool_name", "result_key": "key_name", "payload_template": {{}}}}}}

For BRANCH steps:
  {{"id": "...", "type": "branch", "arms": [{{"conditions": [{{"field": "...", "operator": "eq", "value": ...}}], "goto": "step_id"}}, {{"goto": "default_step_id"}}]}}

For ACTION steps:
  {{"id": "...", "type": "action", "action_type": "context_change", "action_config": {{"trigger_type": "...", "trigger_value": "..."}}}}

For COLLECT steps:
  {{"id": "...", "type": "collect", "expected_intents": ["confirmed", "denied"], "entity_extraction": ["amount", "date"]}}

Rules:
- Maximum {objective.max_steps} steps
- EVERY plan must include at least one message or generate step (the user needs to receive something)
- Put data-gathering steps (tool_call, gate) BEFORE message composition
- Use branch steps when the next action depends on data
- Keep messages appropriate for the channel ({channel.value})

Respond with ONLY the JSON array of steps. No explanation."""

    def _build_replan_prompt(
        self,
        objective: PlanObjective,
        context: dict[str, Any],
        channel: ChannelType,
        failed_result: FlowExecutionResult,
    ) -> str:
        """Build a re-planning prompt after a failed execution."""
        failed_steps = [
            {"id": s.step_id, "type": s.step_type, "status": s.status, "error": s.error}
            for s in failed_result.steps_executed
            if s.status in ("failed", "aborted")
        ]

        base_prompt = self._build_planning_prompt(objective, context, channel)
        return base_prompt + f"""

PREVIOUS PLAN FAILED. Here's what went wrong:
{json.dumps(failed_steps, indent=2)}

Please create an ALTERNATIVE plan that avoids the failed approaches.
If a tool call failed, try a different approach or skip that data.
Respond with ONLY the JSON array of steps."""

    async def _generate_plan(
        self,
        prompt: str,
        objective: PlanObjective,
        channel: ChannelType,
    ) -> Optional[DialogueFlow]:
        """Ask the LLM to produce a plan and parse it into a DialogueFlow."""
        try:
            response = await self._llm(
                "You are a dialogue plan composer. You produce structured JSON plans for conversation agents. "
                "Respond ONLY with a JSON array of step objects. No markdown, no explanation.",
                prompt,
                {"max_tokens": 2000, "temperature": 0.4},
            )

            # Parse the response
            text = response.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                if text.startswith("json"):
                    text = text[4:].strip()

            steps_data = json.loads(text)
            if not isinstance(steps_data, list):
                logger.error("plan_not_a_list", response_type=type(steps_data).__name__)
                return None

            # Convert to DialogueFlow
            steps = self._parse_plan_steps(steps_data)
            if not steps:
                return None

            flow = DialogueFlow(
                id=f"planner_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                name=f"Dynamic plan: {objective.goal[:60]}",
                description=f"Auto-generated plan for: {objective.goal}",
                steps=steps,
                tags=["dynamic", "planner-generated"],
                metadata={
                    "objective": objective.goal,
                    "process_type": objective.process_type,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            logger.info("plan_generated",
                         flow_id=flow.id,
                         steps=len(flow.steps),
                         objective=objective.goal[:80])
            return flow

        except json.JSONDecodeError as e:
            logger.error("plan_json_parse_error", error=str(e))
            return None
        except Exception as e:
            logger.error("plan_generation_failed", error=str(e))
            return None

    def _parse_plan_steps(self, steps_data: list[dict]) -> list[DialogueStep]:
        """Convert raw LLM step dicts into validated DialogueStep objects."""
        steps = []
        for raw in steps_data[:MAX_PLAN_STEPS]:
            try:
                step_type = StepType(raw.get("type", "message"))

                # Build tool definition if present
                tool = None
                if raw.get("tool"):
                    tool_data = raw["tool"]
                    tool = ToolDef(
                        endpoint=tool_data.get("endpoint", ""),
                        result_key=tool_data.get("result_key", ""),
                        payload_template=tool_data.get("payload_template", {}),
                        timeout_seconds=tool_data.get("timeout_seconds", 10),
                    )

                # Build branch arms if present
                arms = []
                for arm_data in raw.get("arms", []):
                    conditions = [
                        RuleCondition(**c) for c in arm_data.get("conditions", [])
                    ]
                    arms.append(BranchArm(
                        conditions=conditions,
                        goto=arm_data.get("goto", ""),
                        description=arm_data.get("description", ""),
                    ))

                # Build conditions for gate steps
                conditions = [
                    RuleCondition(**c) for c in raw.get("conditions", [])
                ]

                step = DialogueStep(
                    id=raw.get("id", f"step_{len(steps)}"),
                    type=step_type,
                    description=raw.get("description", ""),
                    content=raw.get("content", ""),
                    append=raw.get("append", True),
                    system_prompt=raw.get("system_prompt", ""),
                    user_prompt=raw.get("user_prompt", ""),
                    constraints=raw.get("constraints", {}),
                    tool=tool,
                    arms=arms,
                    expected_intents=raw.get("expected_intents", []),
                    entity_extraction=raw.get("entity_extraction", []),
                    timeout_seconds=raw.get("timeout_seconds", 0),
                    timeout_goto=raw.get("timeout_goto", ""),
                    action_type=raw.get("action_type", ""),
                    action_config=raw.get("action_config", {}),
                    conditions=conditions,
                    fail_goto=raw.get("fail_goto", ""),
                    next=raw.get("next", ""),
                )
                steps.append(step)
            except Exception as e:
                logger.warning("plan_step_parse_error",
                               step_index=len(steps),
                               error=str(e),
                               raw=raw)
                continue

        return steps

    # ══════════════════════════════════════════════════
    #  MEMORY ACCUMULATION
    # ══════════════════════════════════════════════════

    async def _accumulate_learnings(
        self,
        objective: PlanObjective,
        context: dict[str, Any],
        flow_result: Optional[FlowExecutionResult],
    ) -> list[dict[str, Any]]:
        """Extract and store learnings from execution results."""
        if not flow_result:
            return []

        memories = []
        contact_id = context.get("contact", {}).get("id", "")
        conversation_id = context.get("conversation_id", "")
        flow_id = flow_result.flow_id

        # Record what plan was used and whether it worked
        if contact_id:
            self._memory.remember(
                scope_type="contact",
                scope_id=contact_id,
                key=f"last_plan_{objective.process_type or 'general'}",
                value={
                    "objective": objective.goal,
                    "status": flow_result.status,
                    "steps": len(flow_result.steps_executed),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                mem_type=MemoryType.EVENT,
                source_flow_id=flow_id,
            )
            memories.append({"type": "event", "key": "last_plan"})

        # Extract tool results as facts
        for step_result in flow_result.steps_executed:
            if step_result.tool_result and step_result.status == "ok":
                scope = "conversation" if conversation_id else "contact"
                scope_id = conversation_id or contact_id
                if scope_id:
                    for key, value in step_result.tool_result.items():
                        if key not in ("status", "error"):
                            self._memory.remember(
                                scope_type=scope,
                                scope_id=scope_id,
                                key=f"tool_{key}",
                                value=value,
                                mem_type=MemoryType.FACT,
                                source_flow_id=flow_id,
                                source_step_id=step_result.step_id,
                            )
                            memories.append({"type": "fact", "key": f"tool_{key}"})

        # Record context changes as events
        for change in flow_result.context_changes:
            if conversation_id:
                self._memory.remember(
                    scope_type="conversation",
                    scope_id=conversation_id,
                    key=f"change_{change.get('trigger_value', 'unknown')}",
                    value=change,
                    mem_type=MemoryType.EVENT,
                    source_flow_id=flow_id,
                )
                memories.append({"type": "event", "key": f"change_{change.get('trigger_value')}"})

        return memories

    # ══════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════

    def _describe_available_flows(self, process_type: str = "") -> str:
        """Describe registered sub-flows the planner could reference."""
        if process_type:
            flows = self._flows.list_by_tag(f"process:{process_type}")
        else:
            flows = self._flows.list_all()

        if not flows:
            return "No pre-defined dialogue flows available."

        lines = ["Available dialogue flows (can be referenced for known patterns):"]
        for f in flows[:15]:
            state_tags = [t for t in f.tags if t.startswith("state:")]
            states = f" [states: {', '.join(t.split(':')[1] for t in state_tags)}]" if state_tags else ""
            lines.append(f"  • {f.id}: {f.description or f.name}{states}")
        return "\n".join(lines)

    def _get_memory_description(self, context: dict[str, Any]) -> str:
        """Get accumulated memory for this context."""
        contact_id = context.get("contact", {}).get("id", "")
        conversation_id = context.get("conversation_id", "")

        # Get entity info from state bindings
        entity_type = ""
        entity_id = ""
        bindings = context.get("state_bindings", [])
        if bindings:
            b = bindings[0] if isinstance(bindings[0], dict) else {}
            entity_type = b.get("entity_type", "")
            entity_id = b.get("entity_id", "")

        desc = self._memory.describe_for_llm(
            contact_id=contact_id,
            conversation_id=conversation_id,
            entity_type=entity_type,
            entity_id=entity_id,
        )

        return f"ACCUMULATED KNOWLEDGE:\n{desc}"

    @staticmethod
    def _channel_guidance(channel: ChannelType) -> str:
        """Channel-specific planning guidance."""
        guidance = {
            ChannelType.WHATSAPP: "WhatsApp: Messages must be SHORT (2-4 sentences). Casual, direct tone. Can use one emoji.",
            ChannelType.EMAIL: "Email: Can be longer and more detailed. Professional tone. Include greeting and sign-off.",
            ChannelType.VOICE: "Voice/TTS: Extremely short (1-3 sentences). Natural spoken language. No formatting.",
            ChannelType.SMS: "SMS: Under 160 characters. Very concise.",
            ChannelType.CHAT: "Chat: Moderate length. Conversational tone.",
        }
        return guidance.get(channel, "")
