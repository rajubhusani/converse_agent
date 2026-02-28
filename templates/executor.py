"""
Dialogue Flow Executor — The agentic brain that runs conversation plans.

This is NOT a simple template renderer. It's a step-by-step executor that:
  - Resolves variables from live context at execution time
  - Calls backend tools mid-flow and uses results in subsequent steps
  - Evaluates conditions to branch between dialogue paths
  - Generates content via LLM with structured, step-specific prompts
  - Adapts rendering per channel (length, tone, format)
  - Fires state machine context changes as side effects
  - Composes a final message from all message/generate steps

Architecture:
  Orchestrator → DialogueExecutor.execute(flow, context, channel)
    → resolve variables
    → walk steps: gate → tool_call → branch → generate → message → action
    → produce FlowExecutionResult with rendered_message + side effects
    → Orchestrator sends rendered_message via channel adapter
    → Orchestrator processes any context_changes emitted

The executor is stateless — all context is passed in, all results passed out.
No hidden side effects; the orchestrator decides what to do with the output.
"""
from __future__ import annotations

import re
import structlog
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from models.schemas import ChannelType, RuleCondition
from templates.models import (
    DialogueFlow, DialogueStep, StepType, ChannelVariant,
    VariableDef, StepResult, FlowExecutionResult,
)
from utils.conditions import evaluate_conditions, get_nested_value

logger = structlog.get_logger()

MAX_STEPS = 50  # prevent infinite loops in branching


class DialogueExecutor:
    """
    Executes a DialogueFlow against live context.

    Dependencies are injected via the constructor so the executor
    can call the LLM engine and backend without circular imports.
    """

    def __init__(
        self,
        llm_generate: Callable = None,
        tool_caller: Callable = None,
        builtin_tools: dict[str, Callable] = None,
        flow_registry=None,
        memory_store=None,
    ):
        """
        Args:
            llm_generate: async fn(system_prompt, user_prompt, constraints) → str
            tool_caller:  async fn(endpoint, payload) → dict
            builtin_tools: dict of "$name" → async fn(context) → dict
            flow_registry: DialogueFlowRegistry (for call_flow steps)
            memory_store:  FlowMemoryStore (for remember steps + context overlay)
        """
        self._llm_generate = llm_generate
        self._tool_caller = tool_caller
        self._builtins = builtin_tools or {}
        self._flow_registry = flow_registry
        self._memory_store = memory_store
        self._register_default_builtins()

    def _register_default_builtins(self):
        """Register built-in tools available to all flows."""

        async def check_business_hours(ctx: dict) -> dict:
            hour = datetime.now(timezone.utc).hour
            tz_offset = ctx.get("timezone_offset", 5.5)  # default IST
            local_hour = (hour + tz_offset) % 24
            is_open = 9 <= local_hour < 18
            return {
                "is_business_hours": is_open,
                "local_hour": local_hour,
                "recommendation": "proceed" if is_open else "delay",
            }

        async def get_conversation_age(ctx: dict) -> dict:
            created = ctx.get("conversation_created_at", "")
            if created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    age = (datetime.now(timezone.utc) - created_dt).total_seconds()
                    return {"age_seconds": age, "age_hours": age / 3600, "age_days": age / 86400}
                except (ValueError, TypeError):
                    pass
            return {"age_seconds": 0, "age_hours": 0, "age_days": 0}

        async def get_attempt_summary(ctx: dict) -> dict:
            attempt = ctx.get("attempt_count", 0)
            max_attempts = ctx.get("max_attempts", 5)
            return {
                "attempt_count": attempt,
                "max_attempts": max_attempts,
                "is_first_attempt": attempt == 0,
                "is_final_attempt": attempt >= max_attempts - 1,
                "remaining_attempts": max(0, max_attempts - attempt - 1),
                "urgency": "low" if attempt == 0 else ("medium" if attempt < 3 else "high"),
            }

        self._builtins.setdefault("$check_business_hours", check_business_hours)
        self._builtins.setdefault("$get_conversation_age", get_conversation_age)
        self._builtins.setdefault("$get_attempt_summary", get_attempt_summary)

    # ══════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════

    async def resume(
        self,
        flow: DialogueFlow,
        resume_step_id: str,
        user_response: str,
        intent_result: dict[str, Any],
        entity_result: dict[str, Any],
        saved_context: dict[str, Any],
        saved_buffer: list[str],
        channel: ChannelType = ChannelType.CHAT,
    ) -> FlowExecutionResult:
        """
        Resume a paused flow from a collect step.

        When a flow hits a collect step, it saves its state. When the user
        responds, this method injects the response, intent classification,
        and extracted entities into the context, then continues execution
        from the step *after* the collect step (or the timeout_goto).

        Args:
            flow:           The original dialogue flow
            resume_step_id: The collect step that paused the flow
            user_response:  The user's reply text
            intent_result:  Result from classify_intent()
            entity_result:  Result from extract_entities()
            saved_context:  The execution context at time of pause
            saved_buffer:   The message buffer at time of pause
            channel:        Target channel
        """
        step_index = flow.step_index
        collect_step = step_index.get(resume_step_id)

        if not collect_step:
            return FlowExecutionResult(
                flow_id=flow.id, flow_name=flow.name,
                status="error", channel=channel,
            )

        # Inject user response data into context
        exec_ctx = {
            **saved_context,
            "user_response": user_response,
            "user_intent": intent_result,
            "extracted_entities": entity_result,
            "matched_intent": intent_result.get("matched_intent", "unknown"),
        }

        # Merge extracted entities at top level for easy access
        for key, val in entity_result.items():
            exec_ctx[f"extracted_{key}"] = val

        # Determine which step to continue from
        next_step = None
        if collect_step.next and collect_step.next in step_index:
            next_step = step_index[collect_step.next]
        else:
            # Get the step after the collect step sequentially
            for i, s in enumerate(flow.steps):
                if s.id == resume_step_id and i + 1 < len(flow.steps):
                    next_step = flow.steps[i + 1]
                    break

        if not next_step:
            # Flow ends after the collect — compose what we have
            return FlowExecutionResult(
                flow_id=flow.id, flow_name=flow.name,
                status="completed", channel=channel,
                rendered_message=self._compose_message(saved_buffer, channel, flow),
            )

        # Continue execution from next_step with enriched context
        result = FlowExecutionResult(
            flow_id=flow.id, flow_name=flow.name, channel=channel,
        )
        result.variables_resolved = exec_ctx.get("_variables", {})

        message_buffer = list(saved_buffer)
        current_step = next_step
        steps_executed = 0

        while current_step and steps_executed < MAX_STEPS:
            steps_executed += 1

            if self._should_skip_step(current_step, channel, flow):
                current_step = self._get_next_step(current_step, step_index, flow)
                continue

            step_result = await self._execute_step(
                current_step, exec_ctx, channel, flow, message_buffer,
            )
            result.steps_executed.append(step_result)
            exec_ctx["_step_results"][current_step.id] = {
                "status": step_result.status,
                "content": step_result.content,
                "tool_result": step_result.tool_result,
                "branch_taken": step_result.branch_taken,
            }

            if step_result.status == "aborted":
                result.status = "aborted"
                break
            if step_result.status == "waiting":
                result.status = "waiting"
                break

            if step_result.action_result and step_result.action_result.get("context_change"):
                result.context_changes.append(step_result.action_result["context_change"])

            if step_result.content:
                if current_step.append:
                    message_buffer.append(step_result.content)
                else:
                    message_buffer = [step_result.content]

            if step_result.branch_taken and step_result.branch_taken in step_index:
                current_step = step_index[step_result.branch_taken]
            elif current_step.next and current_step.next in step_index:
                current_step = step_index[current_step.next]
            else:
                current_step = self._get_next_step(current_step, step_index, flow)

        if result.status != "aborted":
            result.rendered_message = self._compose_message(message_buffer, channel, flow)
            if result.status != "waiting":
                result.status = "completed"

        result.completed_at = datetime.utcnow()
        return result

    async def execute(
        self,
        flow: DialogueFlow,
        context: dict[str, Any],
        channel: ChannelType = ChannelType.CHAT,
    ) -> FlowExecutionResult:
        """
        Execute a dialogue flow against live context.

        Lifecycle:
          1. Inject memory overlay into context
          2. Run on_enter hooks
          3. Resolve declared variables
          4. Walk steps (gate → tool → branch → generate → message → action → remember)
          5. Run on_exit hooks (or on_error if failed)
          6. Compose final message

        Args:
            flow:    The dialogue flow to execute
            context: Merged context (contact, business_data, conversation, state_bindings, etc.)
            channel: Target channel (affects rendering)

        Returns:
            FlowExecutionResult with rendered_message and side effects
        """
        result = FlowExecutionResult(
            flow_id=flow.id,
            flow_name=flow.name,
            channel=channel,
        )

        # 0. Inject memory overlay if available
        if self._memory_store:
            contact_id = context.get("contact", {}).get("id", "")
            conversation_id = context.get("conversation_id", "")
            memory_overlay = self._memory_store.build_context_overlay(
                contact_id=contact_id,
                conversation_id=conversation_id,
            )
            context = {**context, **memory_overlay}

        # 1. Run on_enter hooks
        if flow.on_enter:
            enter_results = await self._run_hooks(flow.on_enter, context, flow)
            result.hook_results.extend(enter_results)
            # Propagate context changes from hooks
            for hr in enter_results:
                if hr.get("hook") == "context_change" and hr.get("change"):
                    result.context_changes.append(hr["change"])

        # 2. Resolve declared variables
        flow_vars = self._resolve_variables(flow.variables, context)
        result.variables_resolved = flow_vars

        # Build merged execution context (original context + resolved vars + runtime)
        exec_ctx = {
            **context,
            **flow_vars,
            "_channel": channel.value,
            "_flow_id": flow.id,
            "_step_results": {},       # results from previous steps, keyed by step_id
            "_tool_results": {},       # results from tool calls, keyed by result_key
        }

        # 3. Walk steps
        message_buffer: list[str] = []
        step_index = flow.step_index
        current_step = flow.get_entry()
        steps_executed = 0

        while current_step and steps_executed < MAX_STEPS:
            steps_executed += 1

            # Check channel skip
            if self._should_skip_step(current_step, channel, flow):
                logger.debug("step_skipped_for_channel",
                             step=current_step.id, channel=channel.value)
                current_step = self._get_next_step(current_step, step_index, flow)
                continue

            # Execute step
            step_result = await self._execute_step(
                current_step, exec_ctx, channel, flow, message_buffer,
            )
            result.steps_executed.append(step_result)

            # Store step result in context for subsequent steps
            exec_ctx["_step_results"][current_step.id] = {
                "status": step_result.status,
                "content": step_result.content,
                "tool_result": step_result.tool_result,
                "branch_taken": step_result.branch_taken,
            }

            # Handle step outcomes
            if step_result.status == "aborted":
                result.status = "aborted"
                break

            if step_result.status == "waiting":
                result.status = "waiting"
                break

            # Collect context changes from action steps
            if step_result.action_result and step_result.action_result.get("context_change"):
                result.context_changes.append(step_result.action_result["context_change"])

            # Collect context changes from call_flow steps
            if step_result.metadata.get("context_changes"):
                result.context_changes.extend(step_result.metadata["context_changes"])

            # Track memory operations from remember steps
            if step_result.step_type == "remember" and step_result.status == "ok":
                result.memories_created.append(step_result.metadata)

            # Track sub-flow results
            if step_result.step_type == "call_flow" and step_result.status == "ok":
                result.sub_flow_results.append({
                    "sub_flow_id": step_result.metadata.get("sub_flow_id"),
                    "status": step_result.metadata.get("sub_flow_status"),
                })

            # Append content to message buffer
            if step_result.content:
                if current_step.append:
                    message_buffer.append(step_result.content)
                else:
                    message_buffer = [step_result.content]

            # Determine next step
            if step_result.branch_taken and step_result.branch_taken in step_index:
                current_step = step_index[step_result.branch_taken]
            elif current_step.next and current_step.next in step_index:
                current_step = step_index[current_step.next]
            else:
                current_step = self._get_next_step(current_step, step_index, flow)

        # 4. Compose final message
        if result.status != "aborted":
            result.rendered_message = self._compose_message(message_buffer, channel, flow)
            if result.status != "waiting":
                result.status = "completed"

        # 5. Run lifecycle hooks
        if result.status in ("completed", "waiting") and flow.on_exit:
            exit_results = await self._run_hooks(flow.on_exit, exec_ctx, flow)
            result.hook_results.extend(exit_results)
        elif result.status in ("aborted", "error") and flow.on_error:
            error_results = await self._run_hooks(flow.on_error, exec_ctx, flow)
            result.hook_results.extend(error_results)

        result.completed_at = datetime.utcnow()

        logger.info("dialogue_flow_executed",
                     flow_id=flow.id,
                     status=result.status,
                     steps=len(result.steps_executed),
                     message_length=len(result.rendered_message),
                     context_changes=len(result.context_changes),
                     memories=len(result.memories_created))

        return result

    # ══════════════════════════════════════════════════════════
    #  STEP EXECUTORS
    # ══════════════════════════════════════════════════════════

    async def _execute_step(
        self,
        step: DialogueStep,
        ctx: dict[str, Any],
        channel: ChannelType,
        flow: DialogueFlow,
        message_buffer: list[str],
    ) -> StepResult:
        """Dispatch to the appropriate step executor."""
        try:
            if step.type == StepType.MESSAGE:
                return await self._exec_message(step, ctx, channel, flow)
            elif step.type == StepType.GENERATE:
                return await self._exec_generate(step, ctx, channel, flow, message_buffer)
            elif step.type == StepType.TOOL_CALL:
                return await self._exec_tool_call(step, ctx)
            elif step.type == StepType.BRANCH:
                return await self._exec_branch(step, ctx)
            elif step.type == StepType.LLM_ROUTE:
                return await self._exec_llm_route(step, ctx, channel, flow)
            elif step.type == StepType.GATE:
                return await self._exec_gate(step, ctx)
            elif step.type == StepType.ACTION:
                return await self._exec_action(step, ctx)
            elif step.type == StepType.COLLECT:
                return await self._exec_collect(step, ctx)
            elif step.type == StepType.CALL_FLOW:
                return await self._exec_call_flow(step, ctx, channel)
            elif step.type == StepType.LOOP:
                return await self._exec_loop(step, ctx, channel, flow, message_buffer)
            elif step.type == StepType.PARALLEL:
                return await self._exec_parallel(step, ctx)
            elif step.type == StepType.REMEMBER:
                return await self._exec_remember(step, ctx)
            else:
                return StepResult(step_id=step.id, step_type=step.type.value,
                                  status="failed", error=f"Unknown step type: {step.type}")
        except Exception as e:
            logger.error("step_execution_error",
                         step_id=step.id, step_type=step.type.value, error=str(e))
            return StepResult(step_id=step.id, step_type=step.type.value,
                              status="failed", error=str(e))

    # ── MESSAGE ───────────────────────────────────────

    async def _exec_message(
        self, step: DialogueStep, ctx: dict, channel: ChannelType, flow: DialogueFlow,
    ) -> StepResult:
        """Render a message template with variable interpolation."""
        content = self._get_channel_content(step, channel, flow)
        rendered = self._interpolate(content, ctx)

        # Apply channel length limit
        max_len = self._get_channel_max_length(step, channel, flow)
        if max_len > 0 and len(rendered) > max_len:
            rendered = rendered[:max_len - 3] + "..."

        return StepResult(
            step_id=step.id, step_type="message",
            status="ok", content=rendered,
        )

    # ── GENERATE ──────────────────────────────────────

    async def _exec_generate(
        self, step: DialogueStep, ctx: dict, channel: ChannelType,
        flow: DialogueFlow, message_buffer: list[str],
    ) -> StepResult:
        """Use LLM to generate content with structured, step-specific prompts."""
        if not self._llm_generate:
            # Fallback: if no LLM, interpolate user_prompt (preferred) or content
            fallback_src = step.user_prompt or step.content
            fallback = self._interpolate(fallback_src, ctx)
            return StepResult(
                step_id=step.id, step_type="generate",
                status="ok", content=fallback,
                metadata={"fallback": True},
            )

        # Build system prompt with channel awareness
        system = self._interpolate(step.system_prompt, ctx)
        channel_suffix = self._get_channel_tone_instruction(step, channel, flow)
        if channel_suffix:
            system += f"\n\n{channel_suffix}"

        # Inject prior message buffer into user prompt so LLM has full picture
        user = self._interpolate(step.user_prompt, ctx)
        if message_buffer:
            user = f"Message so far:\n{''.join(message_buffer)}\n\n{user}"

        # Merge step constraints with channel limits
        constraints = {**step.constraints}
        max_len = self._get_channel_max_length(step, channel, flow)
        if max_len > 0:
            constraints.setdefault("max_tokens", max_len // 4)  # rough char→token

        try:
            generated = await self._llm_generate(system, user, constraints)
            return StepResult(
                step_id=step.id, step_type="generate",
                status="ok", content=generated,
            )
        except Exception as e:
            logger.error("generate_step_failed", step_id=step.id, error=str(e))
            fallback = self._interpolate(step.content, ctx) if step.content else ""
            return StepResult(
                step_id=step.id, step_type="generate",
                status="ok", content=fallback,
                metadata={"fallback": True, "error": str(e)},
            )

    # ── TOOL_CALL ─────────────────────────────────────

    async def _exec_tool_call(self, step: DialogueStep, ctx: dict) -> StepResult:
        """Call a backend endpoint or built-in tool and store the result."""
        if not step.tool:
            return StepResult(step_id=step.id, step_type="tool_call",
                              status="failed", error="No tool defined")

        endpoint = step.tool.endpoint

        # Resolve payload template from context
        payload = {}
        for key, source_path in step.tool.payload_template.items():
            payload[key] = self._resolve_value(source_path, ctx)

        # Call built-in or backend
        try:
            if endpoint.startswith("$"):
                # Built-in tool
                builtin = self._builtins.get(endpoint)
                if not builtin:
                    return StepResult(step_id=step.id, step_type="tool_call",
                                      status="failed", error=f"Unknown builtin: {endpoint}")
                result = await builtin({**ctx, **payload})
            elif self._tool_caller:
                result = await self._tool_caller(endpoint, payload)
            else:
                logger.warning("no_tool_caller_configured", endpoint=endpoint)
                result = {"status": "skipped", "reason": "no_tool_caller"}
        except Exception as e:
            logger.error("tool_call_failed", endpoint=endpoint, error=str(e))
            result = {"status": "error", "error": str(e)}

        # Store in context under result_key
        if step.tool.result_key:
            ctx["_tool_results"][step.tool.result_key] = result
            # Also make available at top-level for easy condition evaluation
            ctx[step.tool.result_key] = result

        return StepResult(
            step_id=step.id, step_type="tool_call",
            status="ok", tool_result=result,
        )

    # ── BRANCH ────────────────────────────────────────

    # ── LLM_ROUTE ─────────────────────────────────────

    async def _exec_llm_route(
        self, step: DialogueStep, ctx: dict,
        channel: ChannelType, flow: DialogueFlow,
    ) -> StepResult:
        """
        LLM-as-judge routing: the model reads the context and
        picks which branch arm to follow.

        Each arm must have a description. The LLM is given the list
        of options and must respond with the arm label (goto value).
        Falls back to condition-based matching if LLM is unavailable.
        """
        if not self._llm_generate or not step.arms:
            # Fallback to condition-based branching
            return await self._exec_branch(step, ctx)

        # Build the routing prompt
        options = []
        for i, arm in enumerate(step.arms):
            options.append(f"  {arm.goto}: {arm.description}")

        system = self._interpolate(step.system_prompt or "", ctx) or (
            "You are a routing agent. Based on the conversation context, "
            "decide which action path to take. Respond with ONLY the route "
            "label — nothing else."
        )

        user_prompt = self._interpolate(step.user_prompt or "", ctx)
        user_prompt += (
            f"\n\nConversation context:\n{ctx.get('conversation_history', 'No history')}"
            f"\n\nLatest message: {ctx.get('user_message', 'N/A')}"
            f"\n\nCurrent state: {ctx.get('current_state', 'unknown')}"
            f"\n\nChoose ONE of the following routes:\n"
            + "\n".join(options)
            + "\n\nRespond with ONLY the route label."
        )

        try:
            response = await self._llm_generate(
                system, user_prompt, {"max_tokens": 50, "temperature": 0.1},
            )
            chosen = response.strip().lower()

            # Find matching arm
            for arm in step.arms:
                if arm.goto.lower() == chosen or arm.goto.lower() in chosen:
                    return StepResult(
                        step_id=step.id, step_type="llm_route",
                        status="ok", branch_taken=arm.goto,
                        metadata={
                            "llm_choice": response.strip(),
                            "arm_description": arm.description,
                        },
                    )

            # LLM response didn't match any arm — use last arm as default
            default_arm = step.arms[-1]
            return StepResult(
                step_id=step.id, step_type="llm_route",
                status="ok", branch_taken=default_arm.goto,
                metadata={
                    "llm_choice": response.strip(),
                    "fallback": True,
                    "arm_description": default_arm.description,
                },
            )
        except Exception as e:
            logger.error("llm_route_failed", step_id=step.id, error=str(e))
            # Fallback to condition-based
            return await self._exec_branch(step, ctx)

    async def _exec_branch(self, step: DialogueStep, ctx: dict) -> StepResult:
        """Evaluate branch arms and route to the matching step."""
        for arm in step.arms:
            if not arm.conditions:
                # Default arm (no conditions = always matches)
                return StepResult(
                    step_id=step.id, step_type="branch",
                    status="ok", branch_taken=arm.goto,
                    metadata={"arm_description": arm.description or "default"},
                )
            if evaluate_conditions(arm.conditions, ctx):
                return StepResult(
                    step_id=step.id, step_type="branch",
                    status="ok", branch_taken=arm.goto,
                    metadata={"arm_description": arm.description},
                )

        # No arm matched
        return StepResult(
            step_id=step.id, step_type="branch",
            status="ok", branch_taken="",
            metadata={"no_arm_matched": True},
        )

    # ── GATE ──────────────────────────────────────────

    async def _exec_gate(self, step: DialogueStep, ctx: dict) -> StepResult:
        """Check conditions — proceed or abort/redirect."""
        if evaluate_conditions(step.conditions, ctx):
            return StepResult(step_id=step.id, step_type="gate", status="ok")

        if step.fail_goto:
            return StepResult(
                step_id=step.id, step_type="gate",
                status="ok", branch_taken=step.fail_goto,
                metadata={"gate_failed": True},
            )

        return StepResult(
            step_id=step.id, step_type="gate",
            status="aborted", error="Gate condition not met",
        )

    # ── ACTION ────────────────────────────────────────

    async def _exec_action(self, step: DialogueStep, ctx: dict) -> StepResult:
        """Emit a context change or other system action."""
        action_type = step.action_type

        if action_type == "context_change":
            change = {
                "trigger_type": step.action_config.get("trigger_type", "action"),
                "trigger_value": self._interpolate(
                    step.action_config.get("trigger_value", ""), ctx,
                ),
                "extra_data": {
                    k: self._resolve_value(v, ctx) if isinstance(v, str) else v
                    for k, v in step.action_config.get("extra_data", {}).items()
                },
            }
            return StepResult(
                step_id=step.id, step_type="action",
                status="ok",
                action_result={"context_change": change},
            )

        elif action_type == "update_binding":
            return StepResult(
                step_id=step.id, step_type="action",
                status="ok",
                action_result={
                    "update_binding": {
                        k: self._resolve_value(v, ctx) if isinstance(v, str) else v
                        for k, v in step.action_config.items()
                    }
                },
            )

        elif action_type == "set_variable":
            # Set a runtime variable for subsequent steps
            key = step.action_config.get("key", "")
            value = step.action_config.get("value", "")
            if key:
                ctx[key] = self._interpolate(str(value), ctx) if isinstance(value, str) else value
            return StepResult(step_id=step.id, step_type="action", status="ok")

        return StepResult(
            step_id=step.id, step_type="action",
            status="ok", metadata={"action_type": action_type},
        )

    # ── COLLECT ───────────────────────────────────────

    async def _exec_collect(self, step: DialogueStep, ctx: dict) -> StepResult:
        """
        Mark the flow as waiting for user input.
        The orchestrator will resume the flow when a matching
        inbound message arrives.
        """
        return StepResult(
            step_id=step.id, step_type="collect",
            status="waiting",
            metadata={
                "expected_intents": step.expected_intents,
                "entity_extraction": step.entity_extraction,
                "timeout_seconds": step.timeout_seconds,
                "timeout_goto": step.timeout_goto,
            },
        )

    # ── CALL_FLOW ─────────────────────────────────────

    async def _exec_call_flow(
        self, step: DialogueStep, ctx: dict, channel: ChannelType,
    ) -> StepResult:
        """
        Invoke a sub-flow by id and merge its results back.

        This enables composable dialogue flows: a payment reminder
        can call a "verify_payment_status" sub-flow, an escalation
        flow can call "collect_complaint_details", etc.

        The sub-flow runs with a mapped subset of the parent context.
        Its output (message content + variable results) are merged back.
        """
        if not self._flow_registry:
            return StepResult(
                step_id=step.id, step_type="call_flow",
                status="failed", error="No flow registry configured",
            )

        sub_flow = self._flow_registry.get(step.sub_flow_id)
        if not sub_flow:
            return StepResult(
                step_id=step.id, step_type="call_flow",
                status="failed",
                error=f"Sub-flow '{step.sub_flow_id}' not found",
            )

        # Map input variables from parent context to sub-flow context
        sub_ctx = dict(ctx)
        for sub_var, source_path in step.sub_flow_input.items():
            sub_ctx[sub_var] = self._resolve_value(source_path, ctx)

        # Execute sub-flow recursively
        sub_result = await self.execute(sub_flow, sub_ctx, channel)

        # Map output variables back to parent context
        for ctx_key, result_key in step.sub_flow_output.items():
            # Look in sub-flow's resolved variables
            if result_key in sub_result.variables_resolved:
                ctx[ctx_key] = sub_result.variables_resolved[result_key]

        # Propagate context changes
        content = sub_result.rendered_message if sub_result.status != "aborted" else ""

        return StepResult(
            step_id=step.id, step_type="call_flow",
            status=sub_result.status if sub_result.status != "waiting" else "ok",
            content=content,
            metadata={
                "sub_flow_id": step.sub_flow_id,
                "sub_flow_status": sub_result.status,
                "sub_steps_executed": len(sub_result.steps_executed),
                "context_changes": sub_result.context_changes,
            },
        )

    # ── LOOP ──────────────────────────────────────────

    async def _exec_loop(
        self, step: DialogueStep, ctx: dict,
        channel: ChannelType, flow: DialogueFlow,
        message_buffer: list[str],
    ) -> StepResult:
        """
        Repeat a set of steps until a condition is met or max iterations reached.

        Use cases:
        - Retry asking for confirmation up to 3 times
        - Iterate through a list of items
        - Keep collecting data until all required fields are present
        """
        step_index = flow.step_index
        iteration = 0
        all_content = []

        while iteration < step.loop_max_iterations:
            # Check exit condition
            if step.loop_condition and not evaluate_conditions(step.loop_condition, ctx):
                break

            # Execute loop body steps
            for loop_step_id in step.loop_steps:
                loop_step = step_index.get(loop_step_id)
                if not loop_step:
                    continue

                loop_result = await self._execute_step(
                    loop_step, ctx, channel, flow, message_buffer,
                )

                if loop_result.content:
                    all_content.append(loop_result.content)

                if loop_result.status in ("aborted", "waiting"):
                    return StepResult(
                        step_id=step.id, step_type="loop",
                        status=loop_result.status,
                        content="\n".join(all_content),
                        metadata={
                            "iterations": iteration + 1,
                            "stopped_at": loop_step_id,
                        },
                    )

                # Store loop step results
                ctx["_step_results"][loop_step_id] = {
                    "status": loop_result.status,
                    "content": loop_result.content,
                    "tool_result": loop_result.tool_result,
                }

            iteration += 1
            ctx["_loop_iteration"] = iteration

        return StepResult(
            step_id=step.id, step_type="loop",
            status="ok",
            content="\n".join(all_content) if all_content else "",
            branch_taken=step.loop_exit_goto if step.loop_exit_goto else "",
            metadata={"iterations": iteration},
        )

    # ── PARALLEL ──────────────────────────────────────

    async def _exec_parallel(self, step: DialogueStep, ctx: dict) -> StepResult:
        """
        Run multiple tool calls concurrently and merge results.

        This is critical for performance: instead of sequential
        tool calls (check_status → get_balance → get_history),
        run them all at once and merge the results.
        """
        import asyncio

        if not step.parallel_tools:
            return StepResult(
                step_id=step.id, step_type="parallel",
                status="ok", metadata={"no_tools": True},
            )

        async def call_one(tool_def):
            payload = {}
            for key, source_path in tool_def.payload_template.items():
                payload[key] = self._resolve_value(source_path, ctx)

            try:
                if tool_def.endpoint.startswith("$"):
                    builtin = self._builtins.get(tool_def.endpoint)
                    if builtin:
                        return tool_def.result_key, await builtin({**ctx, **payload})
                    return tool_def.result_key, {"error": f"Unknown builtin: {tool_def.endpoint}"}
                elif self._tool_caller:
                    return tool_def.result_key, await self._tool_caller(tool_def.endpoint, payload)
                return tool_def.result_key, {"status": "skipped"}
            except Exception as e:
                return tool_def.result_key, {"error": str(e)}

        # Run all tool calls concurrently
        tasks = [call_one(t) for t in step.parallel_tools]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results into context
        merged = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            key, value = item
            if key:
                ctx["_tool_results"][key] = value
                ctx[key] = value
                merged[key] = value

        # Also store under the merge key if specified
        if step.parallel_merge_key:
            ctx[step.parallel_merge_key] = merged
            ctx["_tool_results"][step.parallel_merge_key] = merged

        return StepResult(
            step_id=step.id, step_type="parallel",
            status="ok",
            tool_result=merged,
            metadata={"tools_called": len(step.parallel_tools)},
        )

    # ── REMEMBER ──────────────────────────────────────

    async def _exec_remember(self, step: DialogueStep, ctx: dict) -> StepResult:
        """
        Store a fact, preference, or observation in flow memory.

        This enables flows to accumulate knowledge that persists
        across executions. A payment flow can remember that the
        contact prefers WhatsApp; an escalation flow can record
        the nature of the complaint.
        """
        if not self._memory_store:
            return StepResult(
                step_id=step.id, step_type="remember",
                status="ok", metadata={"skipped": "no_memory_store"},
            )

        from templates.memory import MemoryType as MT

        key = self._interpolate(step.remember_key, ctx) if step.remember_key else ""
        value = self._interpolate(step.remember_value, ctx) if step.remember_value else ""

        if not key:
            return StepResult(
                step_id=step.id, step_type="remember",
                status="failed", error="No remember_key specified",
            )

        # Determine scope id
        scope_type = step.remember_scope or "conversation"
        scope_id = ""
        if scope_type == "contact":
            scope_id = ctx.get("contact", {}).get("id", "")
        elif scope_type == "conversation":
            scope_id = ctx.get("conversation_id", "")
        elif scope_type == "entity":
            bindings = ctx.get("state_bindings", [])
            if bindings:
                b = bindings[0] if isinstance(bindings[0], dict) else {}
                scope_id = f"{b.get('entity_type', '')}:{b.get('entity_id', '')}"

        if not scope_id:
            scope_id = ctx.get("conversation_id", "unknown")

        # Map string to MemoryType
        type_map = {
            "fact": MT.FACT, "preference": MT.PREFERENCE,
            "event": MT.EVENT, "observation": MT.OBSERVATION,
            "summary": MT.SUMMARY, "intent": MT.INTENT,
            "entity": MT.ENTITY,
        }
        mem_type = type_map.get(step.remember_type, MT.FACT)

        self._memory_store.remember(
            scope_type=scope_type,
            scope_id=scope_id,
            key=key,
            value=value,
            mem_type=mem_type,
            source_flow_id=ctx.get("_flow_id", ""),
            source_step_id=step.id,
        )

        return StepResult(
            step_id=step.id, step_type="remember",
            status="ok",
            metadata={
                "memory_key": key,
                "memory_scope": f"{scope_type}:{scope_id}",
                "memory_type": step.remember_type,
            },
        )

    # ── LIFECYCLE HOOKS ───────────────────────────────

    async def _run_hooks(
        self,
        hooks: list[dict[str, Any]],
        ctx: dict[str, Any],
        flow: DialogueFlow,
    ) -> list[dict[str, Any]]:
        """
        Execute lifecycle hooks (on_enter, on_exit, on_error).

        Hook format:
          {"type": "remember", "key": "...", "value": "...", "scope": "..."}
          {"type": "context_change", "trigger_type": "...", "trigger_value": "..."}
          {"type": "tool_call", "endpoint": "...", "payload": {...}}
          {"type": "log", "message": "..."}
        """
        results = []
        for hook in hooks:
            try:
                hook_type = hook.get("type", "")

                if hook_type == "remember" and self._memory_store:
                    from templates.memory import MemoryType as MT
                    key = self._interpolate(hook.get("key", ""), ctx)
                    value = self._interpolate(str(hook.get("value", "")), ctx)
                    scope = hook.get("scope", "conversation")
                    scope_id = ctx.get("conversation_id", "") if scope == "conversation" else ctx.get("contact", {}).get("id", "")
                    self._memory_store.remember(
                        scope_type=scope, scope_id=scope_id or "unknown",
                        key=key, value=value,
                        source_flow_id=flow.id,
                    )
                    results.append({"hook": "remember", "key": key, "status": "ok"})

                elif hook_type == "context_change":
                    change = {
                        "trigger_type": self._interpolate(hook.get("trigger_type", ""), ctx),
                        "trigger_value": self._interpolate(hook.get("trigger_value", ""), ctx),
                        "extra_data": hook.get("extra_data", {}),
                    }
                    results.append({"hook": "context_change", "change": change, "status": "ok"})

                elif hook_type == "tool_call" and self._tool_caller:
                    endpoint = hook.get("endpoint", "")
                    payload = {
                        k: self._interpolate(str(v), ctx) if isinstance(v, str) else v
                        for k, v in hook.get("payload", {}).items()
                    }
                    result = await self._tool_caller(endpoint, payload)
                    results.append({"hook": "tool_call", "endpoint": endpoint, "status": "ok", "result": result})

                elif hook_type == "log":
                    msg = self._interpolate(hook.get("message", ""), ctx)
                    logger.info("flow_hook_log", flow_id=flow.id, message=msg)
                    results.append({"hook": "log", "message": msg, "status": "ok"})

            except Exception as e:
                logger.warning("hook_execution_error", hook=hook, error=str(e))
                results.append({"hook": hook.get("type", "?"), "status": "error", "error": str(e)})

        return results

    # ══════════════════════════════════════════════════════════
    #  VARIABLE RESOLUTION
    # ══════════════════════════════════════════════════════════

    def _resolve_variables(
        self,
        var_defs: list[VariableDef],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve declared variables from context."""
        resolved = {}
        for var in var_defs:
            value = get_nested_value(context, var.source)
            if value is None:
                if var.required:
                    logger.warning("required_variable_missing",
                                    variable=var.name, source=var.source)
                value = var.default

            # Apply format
            value = self._format_value(value, var.format)
            resolved[var.name] = value

        return resolved

    def _resolve_value(self, source_path: str, ctx: dict) -> Any:
        """Resolve a single dot-path value from context."""
        if source_path.startswith("{{") and source_path.endswith("}}"):
            source_path = source_path[2:-2].strip()
        return get_nested_value(ctx, source_path) or source_path

    @staticmethod
    def _format_value(value: Any, fmt: str) -> Any:
        """Apply formatting to a resolved variable."""
        if value is None or not fmt:
            return value
        try:
            if fmt == "currency":
                if isinstance(value, (int, float)):
                    # Indian-style grouping: 2,50,000.00
                    integer_part = int(value)
                    decimal_part = f"{value:.2f}".split(".")[1]
                    s = str(integer_part)
                    if len(s) > 3:
                        last3 = s[-3:]
                        rest = s[:-3]
                        # Group remaining digits in pairs from right
                        groups = []
                        while rest:
                            groups.append(rest[-2:] if len(rest) >= 2 else rest)
                            rest = rest[:-2]
                        groups.reverse()
                        formatted = ",".join(groups) + "," + last3
                    else:
                        formatted = s
                    return f"₹{formatted}.{decimal_part}"
            elif fmt == "currency_usd":
                if isinstance(value, (int, float)):
                    return f"${value:,.2f}"
            elif fmt == "date":
                if isinstance(value, str):
                    dt = datetime.fromisoformat(value)
                    return dt.strftime("%d %b %Y")
            elif fmt == "capitalize":
                return str(value).title()
            elif fmt == "upper":
                return str(value).upper()
            elif fmt == "phone":
                s = str(value)
                if len(s) > 10:
                    return f"+{s[:2]} {s[2:7]} {s[7:]}"
        except (ValueError, TypeError):
            pass
        return value

    # ══════════════════════════════════════════════════════════
    #  TEMPLATE INTERPOLATION
    # ══════════════════════════════════════════════════════════

    def _interpolate(self, template: str, ctx: dict[str, Any]) -> str:
        """Replace {{variable}} placeholders with values from context."""
        if not template:
            return ""

        def replacer(match):
            key = match.group(1).strip()
            # Support dot notation in interpolation
            val = get_nested_value(ctx, key)
            if val is None:
                val = ctx.get(key, match.group(0))  # leave unreplaced if not found
            return str(val) if val is not None else ""

        return re.sub(r"\{\{([^}]+)\}\}", replacer, template)

    # ══════════════════════════════════════════════════════════
    #  CHANNEL AWARENESS
    # ══════════════════════════════════════════════════════════

    def _should_skip_step(
        self, step: DialogueStep, channel: ChannelType, flow: DialogueFlow,
    ) -> bool:
        """Check if this step should be skipped for the given channel."""
        for variant in step.channel_variants:
            if variant.channel == channel and variant.skip:
                return True
        return False

    def _get_channel_content(
        self, step: DialogueStep, channel: ChannelType, flow: DialogueFlow,
    ) -> str:
        """Get channel-specific content override, falling back to step content."""
        for variant in step.channel_variants:
            if variant.channel == channel and variant.content_override:
                return variant.content_override
        return step.content

    def _get_channel_max_length(
        self, step: DialogueStep, channel: ChannelType, flow: DialogueFlow,
    ) -> int:
        """Get the character limit for this step on this channel."""
        # Step-level variant
        for variant in step.channel_variants:
            if variant.channel == channel and variant.max_length > 0:
                return variant.max_length
        # Flow-level default
        ch_key = channel.value
        if ch_key in flow.default_channel_config:
            return flow.default_channel_config[ch_key].max_length
        # Sensible defaults
        defaults = {ChannelType.WHATSAPP: 1000, ChannelType.VOICE: 300, ChannelType.SMS: 160}
        return defaults.get(channel, 0)

    def _get_channel_tone_instruction(
        self, step: DialogueStep, channel: ChannelType, flow: DialogueFlow,
    ) -> str:
        """Build a tone instruction for LLM generate steps."""
        tone = ""
        # Step-level
        for variant in step.channel_variants:
            if variant.channel == channel:
                tone = variant.tone or variant.system_prompt_suffix
                break
        # Flow-level default
        if not tone:
            ch_key = channel.value
            if ch_key in flow.default_channel_config:
                cfg = flow.default_channel_config[ch_key]
                tone = cfg.tone or cfg.system_prompt_suffix

        if not tone:
            # Sensible defaults
            tones = {
                ChannelType.WHATSAPP: "Keep the message SHORT (2-4 sentences). Use simple, direct language.",
                ChannelType.VOICE: "This will be spoken aloud. Use natural, conversational language. Keep it very brief (1-3 sentences).",
                ChannelType.EMAIL: "Write in professional email format. Can be slightly longer.",
                ChannelType.SMS: "Extremely concise. Under 160 characters.",
            }
            tone = tones.get(channel, "")

        return tone

    # ══════════════════════════════════════════════════════════
    #  MESSAGE COMPOSITION
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _get_next_step(
        current: DialogueStep,
        step_index: dict[str, DialogueStep],
        flow: DialogueFlow,
    ) -> Optional[DialogueStep]:
        """Get the next sequential step in the flow (by position)."""
        steps = flow.steps
        for i, s in enumerate(steps):
            if s.id == current.id and i + 1 < len(steps):
                return steps[i + 1]
        return None

    # ══════════════════════════════════════════════════════════
    #  MESSAGE COMPOSITION
    # ══════════════════════════════════════════════════════════

    def _compose_message(
        self, buffer: list[str], channel: ChannelType, flow: DialogueFlow,
    ) -> str:
        """
        Compose the final message from the buffer.
        Joins fragments with channel-appropriate separators.
        """
        if not buffer:
            return ""

        # Channel-specific joining
        if channel == ChannelType.EMAIL:
            separator = "\n\n"
        elif channel in (ChannelType.WHATSAPP, ChannelType.SMS):
            separator = "\n"
        elif channel == ChannelType.VOICE:
            separator = " "
        else:
            separator = "\n\n"

        composed = separator.join(part.strip() for part in buffer if part.strip())

        # Final length enforcement
        max_len = 0
        ch_key = channel.value
        if ch_key in flow.default_channel_config:
            max_len = flow.default_channel_config[ch_key].max_length
        if max_len > 0 and len(composed) > max_len:
            composed = composed[:max_len - 3] + "..."

        return composed
