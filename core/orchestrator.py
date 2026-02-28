"""
Orchestrator — The central coordinator for all conversational flow.

Architecture:
  Inbound:  webhook/WS → route message → detect intent
            → build ContextChange → state machine evaluates all bindings
            → TransitionActions returned → ActionExecutor runs them
            → generate + send response

  Outbound: Rules Engine → create FollowUp + StateBinding
            → enqueue to message queue → Consumer calls dispatch_followup()

  Backend Event: event webhook → build ContextChange
            → state machine evaluates all bindings for entity
            → TransitionActions returned → ActionExecutor runs them

The hardcoded if/elif intent branching is gone. ALL business logic
lives in the state map configuration. The orchestrator is a generic
execution engine.
"""
from __future__ import annotations

import asyncio
import structlog
from datetime import datetime, timedelta
from typing import Any, Optional

from config.settings import get_settings
from models.schemas import (
    ChannelType, Contact, Conversation, ConversationStatus,
    FollowUp, FollowUpStatus, FollowUpPriority, MessageDirection,
    StateBinding, TransitionAction,
)
from context.tracker import ContextTracker, ContextChange
from context.state_machine import BusinessStateMachine, TransitionResult
from core.engine import ConversationEngine
from rules.engine import RulesEngine
from backend.connector import BackendConnector, create_backend_connector
from channels.base import ChannelRegistry
from job_queue.message_queue import MessageQueue, QueueJob, Queues, get_message_queue
from templates.registry import DialogueFlowRegistry
from templates.executor import DialogueExecutor
from templates.models import FlowExecutionResult
from templates.sessions import (
    DialogueSessionStore, DialogueSession,
    classify_intent, extract_entities,
    llm_classify_intent, llm_extract_entities,
)
from templates.planner import AgenticPlanner, PlanObjective
from templates.memory import FlowMemoryStore, MemoryType
from templates.tool_registry import ToolRegistry, create_default_tool_registry

logger = structlog.get_logger()


class Orchestrator:
    """
    Generic orchestrator. Business logic lives in state maps, not here.

    This class:
    1. Routes messages and events
    2. Feeds context changes to the state machine
    3. Executes whatever actions the state machine returns
    4. Manages the dispatch lifecycle for outbound follow-ups
    5. Falls back to agentic planning when no pre-defined flow matches
    """

    def __init__(
        self,
        context: ContextTracker,
        engine: ConversationEngine,
        rules: RulesEngine,
        state_machine: BusinessStateMachine,
        channels: ChannelRegistry,
        backend: BackendConnector = None,
        queue: MessageQueue = None,
        dialogue_registry: DialogueFlowRegistry = None,
        tool_registry: ToolRegistry = None,
        memory_store: FlowMemoryStore = None,
    ):
        self.context = context
        self.engine = engine
        self.rules = rules
        self.state_machine = state_machine
        self.channels = channels
        self.backend = backend or create_backend_connector()
        self.queue = queue or get_message_queue()
        self._settings = get_settings()

        # Agentic template system
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.memory_store = memory_store or FlowMemoryStore()
        self.dialogue_registry = dialogue_registry or DialogueFlowRegistry()
        self.dialogue_executor = DialogueExecutor(
            llm_generate=self.engine.generate_for_step,
            tool_caller=self._dialogue_tool_caller,
            builtin_tools={},
            flow_registry=self.dialogue_registry,
            memory_store=self.memory_store,
        )
        self.dialogue_sessions = DialogueSessionStore()
        self.planner = AgenticPlanner(
            llm_generate=self.engine.generate_for_step,
            tool_registry=self.tool_registry,
            flow_registry=self.dialogue_registry,
            memory_store=self.memory_store,
        )

    async def _dialogue_tool_caller(
        self, endpoint: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Bridge between dialogue executor tool_call steps and the backend."""
        return await self.backend.call_endpoint(endpoint, payload)

    # ══════════════════════════════════════════════════════════
    #  INBOUND — Message received from a contact
    # ══════════════════════════════════════════════════════════

    async def handle_inbound_message(
        self,
        channel: ChannelType,
        sender_address: str,
        content: str,
        metadata: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Main entry point for ALL inbound messages across ALL channels.

        Flow:
        1. Route to contact + conversation
        2. Detect intent via LLM
        3. Build ContextChange from detected intent
        4. State machine evaluates all bindings on this conversation
        5. Execute transition actions (enqueue, resolve, escalate, update_backend)
        6. Generate + send response
        """
        logger.info("inbound_message",
                     channel=channel.value,
                     sender=sender_address,
                     content=content[:100])

        # 1. Route message
        contact, conversation, message = await self.context.route_inbound_message(
            channel=channel,
            sender_address=sender_address,
            content=content,
            metadata=metadata,
        )

        # ── SESSION RESUMPTION ────────────────────────────
        # If there's an active dialogue session waiting for user input
        # on this conversation, resume the flow with the user's response
        # instead of running the full standard pipeline.
        session = self.dialogue_sessions.get(conversation.id)
        if session:
            resume_result = await self._resume_dialogue_session(
                session, content, conversation, contact, channel,
            )
            if resume_result:
                return resume_result
        # ── END SESSION RESUMPTION ────────────────────────

        # 2. Get full context and detect intent
        full_context = await self.context.get_full_context(conversation.id)
        intent_data = await self.engine.detect_intent(full_context, content)

        intent = intent_data.get("intent", "unknown")
        logger.info("intent_detected",
                     intent=intent,
                     sentiment=intent_data.get("sentiment"),
                     conversation_id=conversation.id)

        # 3. Build context change from intent
        change = await self.context.build_context_change(
            conversation_id=conversation.id,
            trigger_type="intent",
            trigger_value=intent,
            extra_data={
                "intent_data": intent_data,
                "extracted_info": intent_data.get("extracted_info", {}),
                "sentiment": intent_data.get("sentiment", "neutral"),
            },
            source_channel=channel,
        )

        # 4. Apply to all state bindings on this conversation
        bindings = await self.context.get_conversation_bindings(conversation.id)
        transition_results = self.state_machine.apply_trigger_to_all(
            bindings=bindings,
            trigger_type=change.trigger_type,
            trigger_value=change.trigger_value,
            context_data=change.context_data,
        )

        # 5. Execute all actions from all transitions
        action_results = []
        for result in transition_results:
            await self.context.persist_binding(result.binding)
            executed = await self._execute_actions(
                result.actions, conversation, contact, result,
            )
            action_results.extend(executed)

        # 6. Generate response — use dialogue flow if available, else LLM
        full_context = await self.context.get_full_context(conversation.id)

        flow_result = await self._execute_dialogue_flow(
            template_name="",  # inbound doesn't specify — resolve by context
            process_type=self._get_active_process_type(transition_results),
            current_state=self._get_active_state(transition_results),
            context=full_context,
            channel=channel,
            user_message=content,
            conversation_id=conversation.id,
        )

        if flow_result and flow_result.rendered_message:
            response_text = flow_result.rendered_message
            # Process any context changes emitted by the flow
            await self._process_flow_context_changes(flow_result, conversation.id)
        else:
            response_text = await self.engine.generate_response(full_context, channel, content)

        if response_text:
            adapter = self.channels.get(channel)
            if adapter:
                send_result = await adapter.send_message(contact, response_text)
                await self.context.record_outbound_message(
                    conversation.id, channel, response_text,
                    metadata={"delivery": send_result, "intent": intent_data},
                )

        # Accumulate interaction knowledge into memory
        await self._accumulate_interaction_memory(
            contact=contact,
            conversation=conversation,
            intent_data=intent_data,
            user_message=content,
            channel=channel,
            transition_results=transition_results,
        )

        return {
            "conversation_id": conversation.id,
            "contact_id": contact.id,
            "intent": intent_data,
            "response": response_text,
            "channel": channel.value,
            "transitions": [
                {
                    "process_type": r.binding.process_type,
                    "from": r.from_state,
                    "to": r.to_state,
                    "actions": [a.type for a in r.actions],
                }
                for r in transition_results
            ],
            "action_results": action_results,
        }

    # ══════════════════════════════════════════════════════════
    #  BACKEND EVENT — Business system event drives state changes
    # ══════════════════════════════════════════════════════════

    async def handle_backend_event(
        self,
        event_name: str,
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Process a real-time event from the backend system.
        E.g. "payment_received", "shipment_dispatched", "order_cancelled".

        Two paths:
        A) If the event references a known entity → find its state bindings
           → apply trigger → execute transition actions.
        B) Also evaluate rules engine for new follow-ups to enqueue.
        """
        logger.info("backend_event", event=event_name, data_keys=list(data.keys()))
        results = []

        # Path A: Drive existing state bindings via the event
        entity_type = data.get("entity_type", "")
        entity_id = data.get("entity_id", "")

        if entity_type and entity_id:
            bindings = await self.context.store.get_bindings_for_entity(
                entity_type, entity_id,
            )
            if bindings:
                transition_results = self.state_machine.apply_trigger_to_all(
                    bindings=bindings,
                    trigger_type="event",
                    trigger_value=event_name,
                    context_data=data,
                )
                for result in transition_results:
                    await self.context.persist_binding(result.binding)
                    # Load the conversation and contact for action execution
                    conv = await self.context.store.get_conversation(result.binding.conversation_id)
                    contact = await self.context.store.get_contact(result.binding.contact_id) if conv else None
                    executed = await self._execute_actions(
                        result.actions, conv, contact, result,
                    )
                    results.append({
                        "source": "state_machine",
                        "binding_id": result.binding.id,
                        "process_type": result.binding.process_type,
                        "transition": f"{result.from_state} → {result.to_state}",
                        "actions": executed,
                    })

        # Path B: Also evaluate rules engine for new follow-ups
        rule_followups = await self.rules.on_event(event_name, data)
        for followup in rule_followups:
            job = await self.enqueue_followup(followup)
            results.append({
                "source": "rules_engine",
                "followup_id": followup.id,
                "job_id": job.job_id,
                "status": "enqueued",
            })

        return results

    # ══════════════════════════════════════════════════════════
    #  CONTEXT CHANGE — Generic trigger for any context change
    # ══════════════════════════════════════════════════════════

    async def apply_context_change(
        self,
        change: ContextChange,
    ) -> list[dict[str, Any]]:
        """
        Generic entry point: apply any context change to all relevant
        state bindings and execute resulting actions.

        This is the unified interface for:
        - Timeouts ("no response after 48h")
        - Manual operator overrides
        - System-level actions ("first_outreach_sent")
        - Anything not covered by inbound/event paths
        """
        bindings = await self.context.get_conversation_bindings(change.conversation_id)
        transition_results = self.state_machine.apply_trigger_to_all(
            bindings=bindings,
            trigger_type=change.trigger_type,
            trigger_value=change.trigger_value,
            context_data=change.context_data,
        )

        results = []
        for result in transition_results:
            await self.context.persist_binding(result.binding)
            conv = await self.context.store.get_conversation(change.conversation_id)
            contact = await self.context.store.get_contact(change.contact_id)
            executed = await self._execute_actions(result.actions, conv, contact, result)
            results.append({
                "binding_id": result.binding.id,
                "transition": f"{result.from_state} → {result.to_state}",
                "actions": executed,
            })

        return results

    # ══════════════════════════════════════════════════════════
    #  ACTION EXECUTOR — Runs declarative TransitionActions
    # ══════════════════════════════════════════════════════════

    async def _execute_actions(
        self,
        actions: list[TransitionAction],
        conversation: Optional[Conversation],
        contact: Optional[Contact],
        transition_result: TransitionResult,
    ) -> list[dict[str, Any]]:
        """
        Execute a list of TransitionActions returned by the state machine.

        Action types:
          enqueue_followup  — Create a FollowUp and publish to the dispatch queue
          update_backend    — Push state change to the business backend system
          resolve           — Mark the conversation as resolved
          escalate          — Escalate the conversation
          notify            — Send a notification (email/slack to internal team)
          schedule_check    — Schedule a future context change (e.g. "check again in 48h")
        """
        results = []
        binding = transition_result.binding

        for action in actions:
            try:
                result = await self._execute_single_action(
                    action, conversation, contact, binding, transition_result,
                )
                results.append(result)
            except Exception as e:
                logger.error("action_execution_error",
                             action_type=action.type,
                             binding_id=binding.id,
                             error=str(e))
                results.append({
                    "action": action.type,
                    "status": "error",
                    "error": str(e),
                })

        return results

    async def _execute_single_action(
        self,
        action: TransitionAction,
        conversation: Optional[Conversation],
        contact: Optional[Contact],
        binding: StateBinding,
        transition_result: TransitionResult,
    ) -> dict[str, Any]:
        """Execute one TransitionAction. Returns a result dict."""

        action_type = action.type

        # ── enqueue_followup ─────────────────────────────────
        if action_type == "enqueue_followup":
            followup = FollowUp(
                rule_id=f"state:{binding.process_type}",
                contact_id=binding.contact_id,
                conversation_id=binding.conversation_id,
                status=FollowUpStatus.SCHEDULED,
                priority=action.priority,
                business_context={
                    "context_key": binding.entity_id or binding.id,
                    "process_type": binding.process_type,
                    "current_state": binding.current_state,
                    "template": action.template,
                    "entity_id": binding.entity_id,
                    "entity_type": binding.entity_type,
                    **binding.business_data,
                },
                reason=f"State transition: {transition_result.from_state} → {transition_result.to_state}",
                channel_priority=action.channel_priority or [ChannelType.WHATSAPP, ChannelType.EMAIL],
                next_attempt_at=(
                    datetime.utcnow() + timedelta(minutes=action.delay_minutes)
                    if action.delay_minutes > 0 else None
                ),
                metadata=action.metadata,
            )
            followup = await self.context.store.create_followup(followup)
            job = await self.enqueue_followup(followup)
            logger.info("action_enqueue_followup",
                        followup_id=followup.id,
                        job_id=job.job_id,
                        template=action.template)
            return {
                "action": "enqueue_followup",
                "status": "enqueued",
                "followup_id": followup.id,
                "job_id": job.job_id,
            }

        # ── update_backend ───────────────────────────────────
        elif action_type == "update_backend":
            payload = {
                "process_type": binding.process_type,
                "entity_id": binding.entity_id,
                "entity_type": binding.entity_type,
                "new_state": binding.current_state,
                "previous_state": transition_result.from_state,
                "contact_id": binding.contact_id,
                **action.backend_payload,
            }
            endpoint = action.backend_endpoint
            if endpoint:
                result = await self.backend.call_endpoint(endpoint, payload)
            else:
                # Default: update follow-up status on backend
                if conversation and conversation.followup_id:
                    result = await self.backend.update_followup_status(
                        conversation.followup_id,
                        binding.current_state,
                        f"State: {transition_result.from_state} → {binding.current_state}",
                    )
                else:
                    result = {"status": "skipped", "reason": "no_endpoint_or_followup"}
            logger.info("action_update_backend",
                        endpoint=endpoint,
                        new_state=binding.current_state)
            return {"action": "update_backend", "status": "sent", "result": result}

        # ── resolve ──────────────────────────────────────────
        elif action_type == "resolve":
            if conversation:
                summary = await self.engine.summarize_conversation(
                    await self.context.get_full_context(conversation.id)
                )
                await self.context.resolve_conversation(conversation.id, summary)
                logger.info("action_resolve",
                            conversation_id=conversation.id)
            return {"action": "resolve", "status": "resolved"}

        # ── escalate ─────────────────────────────────────────
        elif action_type == "escalate":
            if conversation:
                reason = action.metadata.get(
                    "reason",
                    f"Escalated from state {transition_result.from_state}",
                )
                await self.context.escalate_conversation(conversation.id, reason)
                logger.info("action_escalate",
                            conversation_id=conversation.id,
                            reason=reason)
            return {"action": "escalate", "status": "escalated"}

        # ── notify ───────────────────────────────────────────
        elif action_type == "notify":
            # Internal notification (e.g. Slack, email to manager)
            notify_target = action.metadata.get("target", "")
            notify_message = action.metadata.get(
                "message",
                f"State change: {binding.process_type} "
                f"{transition_result.from_state} → {binding.current_state} "
                f"for {binding.entity_type}:{binding.entity_id}",
            )
            logger.info("action_notify",
                        target=notify_target,
                        process=binding.process_type)
            # In production, send via Slack/email adapter
            return {
                "action": "notify",
                "status": "sent",
                "target": notify_target,
                "message": notify_message,
            }

        # ── schedule_check ───────────────────────────────────
        elif action_type == "schedule_check":
            # Schedule a future ContextChange (e.g. "if no response in 48h, check again")
            delay = action.delay_minutes or 60
            check_trigger = action.metadata.get("check_trigger_type", "timeout")
            check_value = action.metadata.get(
                "check_trigger_value",
                f"no_response_{delay}m",
            )
            # Create a delayed follow-up that will fire a context change
            followup = FollowUp(
                rule_id=f"scheduled_check:{binding.process_type}",
                contact_id=binding.contact_id,
                conversation_id=binding.conversation_id,
                status=FollowUpStatus.SCHEDULED,
                priority=FollowUpPriority.LOW,
                business_context={
                    "context_key": binding.entity_id or binding.id,
                    "process_type": binding.process_type,
                    "check_trigger_type": check_trigger,
                    "check_trigger_value": check_value,
                    "template": action.template,
                },
                reason=f"Scheduled state check after {delay}min",
                channel_priority=action.channel_priority or [ChannelType.WHATSAPP, ChannelType.EMAIL],
                next_attempt_at=datetime.utcnow() + timedelta(minutes=delay),
                metadata={"is_scheduled_check": True, **action.metadata},
            )
            followup = await self.context.store.create_followup(followup)
            job = await self.enqueue_followup(followup)
            logger.info("action_schedule_check",
                        followup_id=followup.id,
                        delay_minutes=delay,
                        check_trigger=f"{check_trigger}:{check_value}")
            return {
                "action": "schedule_check",
                "status": "scheduled",
                "followup_id": followup.id,
                "fires_at": (datetime.utcnow() + timedelta(minutes=delay)).isoformat(),
            }

        # ── unknown ──────────────────────────────────────────
        else:
            logger.warning("unknown_action_type", action_type=action_type)
            return {"action": action_type, "status": "unknown"}

    # ══════════════════════════════════════════════════════════
    #  DIALOGUE FLOW EXECUTION
    # ══════════════════════════════════════════════════════════

    async def _execute_dialogue_flow(
        self,
        template_name: str,
        process_type: str,
        current_state: str,
        context: dict[str, Any],
        channel: ChannelType,
        user_message: str = "",
        conversation_id: str = "",
    ) -> Optional[FlowExecutionResult]:
        """
        Resolve and execute the best dialogue flow for the given context.

        Resolution cascade:
        1. Try exact template match from registry
        2. Try process_type + state match from registry
        3. Fall back to agentic planner (LLM composes a dynamic flow)

        Returns None only if the planner also fails (caller falls back to raw LLM).
        If the flow pauses at a collect step, saves a DialogueSession.
        """
        flow = self.dialogue_registry.resolve(
            template_name=template_name,
            process_type=process_type,
            current_state=current_state,
            context_data=context,
            channel=channel,
        )

        # Enrich context with user_message for flows that need it
        exec_context = {
            **context,
            "user_message": user_message,
        }

        if flow:
            # Execute the pre-defined flow
            result = await self.dialogue_executor.execute(flow, exec_context, channel)
        else:
            # No pre-defined flow → engage the agentic planner
            logger.info("no_matching_flow_engaging_planner",
                         process_type=process_type,
                         current_state=current_state,
                         template_name=template_name)

            objective = PlanObjective(
                goal=self._infer_objective(process_type, current_state, user_message, context),
                process_type=process_type,
                current_state=current_state,
            )

            plan_result = await self.planner.plan_and_execute(
                objective=objective,
                context=exec_context,
                channel=channel,
                executor=self.dialogue_executor,
            )

            if plan_result.flow_result and plan_result.flow_result.rendered_message:
                result = plan_result.flow_result
                result.is_dynamic = True
                result.replan_count = plan_result.replans
            else:
                return None  # Planner failed too — caller falls back to raw LLM

        # If flow is waiting (hit a collect step), save session for resumption
        if result.status == "waiting" and conversation_id:
            waiting_step = next(
                (s for s in result.steps_executed if s.status == "waiting"),
                None,
            )
            if waiting_step:
                from datetime import datetime, timezone, timedelta
                meta = waiting_step.metadata
                timeout_secs = meta.get("timeout_seconds", 0)
                session = DialogueSession(
                    id=conversation_id,
                    flow_id=flow.id if flow else result.flow_id,
                    paused_at_step=waiting_step.step_id,
                    channel=channel,
                    execution_context=exec_context,
                    message_buffer=[s.content for s in result.steps_executed if s.content],
                    expected_intents=meta.get("expected_intents", []),
                    entity_extraction=meta.get("entity_extraction", []),
                    timeout_at=(
                        datetime.now(timezone.utc) + timedelta(seconds=timeout_secs)
                        if timeout_secs > 0 else None
                    ),
                    timeout_goto=meta.get("timeout_goto", ""),
                )
                self.dialogue_sessions.save(session)

        # Accumulate memories from flow execution
        if result.memories_created:
            logger.info("flow_memories_stored",
                         flow_id=result.flow_id,
                         count=len(result.memories_created))

        logger.info("dialogue_flow_result",
                     flow_id=result.flow_id,
                     status=result.status,
                     message_length=len(result.rendered_message),
                     steps=len(result.steps_executed),
                     context_changes=len(result.context_changes),
                     is_dynamic=result.is_dynamic)

        return result

    def _infer_objective(
        self,
        process_type: str,
        current_state: str,
        user_message: str,
        context: dict[str, Any],
    ) -> str:
        """Infer a planning objective from available context."""
        parts = []
        if process_type:
            parts.append(f"Handle the {process_type} process")
        if current_state:
            parts.append(f"in state '{current_state}'")
        if user_message:
            parts.append(f"responding to: '{user_message[:100]}'")
        elif not process_type:
            parts.append("Generate an appropriate follow-up message")

        business = context.get("business_context", {})
        if business:
            reason = business.get("reason", "")
            if reason:
                parts.append(f"regarding: {reason[:80]}")

        return ". ".join(parts) if parts else "Generate an appropriate conversational response"

    async def _resume_dialogue_session(
        self,
        session: DialogueSession,
        user_message: str,
        conversation: Conversation,
        contact: Contact,
        channel: ChannelType,
    ) -> Optional[dict[str, Any]]:
        """
        Resume a paused dialogue flow from a saved session.

        1. Classify user intent against expected intents
        2. Extract entities
        3. Resume the flow from the paused collect step
        4. Handle the result (send message, save new session if paused again)
        5. Remove the session if flow completed
        """
        flow = self.dialogue_registry.get(session.flow_id)
        if not flow:
            logger.warning("session_flow_not_found",
                           flow_id=session.flow_id,
                           conversation_id=session.id)
            self.dialogue_sessions.remove(session.id)
            return None

        # Check if session expired
        if session.is_expired and session.timeout_goto:
            logger.info("session_expired_resuming_timeout",
                         flow_id=session.flow_id,
                         timeout_goto=session.timeout_goto)
            # Resume from timeout_goto step instead
            resume_step = session.timeout_goto
        else:
            resume_step = session.paused_at_step

        # Classify intent and extract entities (LLM-first, regex fallback)
        intent_result = await llm_classify_intent(
            user_message, session.expected_intents,
            context={"business_context": conversation.business_context},
            llm_generate=self.engine.generate_for_step,
        )
        entity_result = await llm_extract_entities(
            user_message, session.entity_extraction,
            context={"business_context": conversation.business_context},
            llm_generate=self.engine.generate_for_step,
        )

        logger.info("session_resume",
                     flow_id=session.flow_id,
                     step=resume_step,
                     intent=intent_result.get("matched_intent"),
                     entities=list(entity_result.keys()))

        # Resume the flow
        result = await self.dialogue_executor.resume(
            flow=flow,
            resume_step_id=resume_step,
            user_response=user_message,
            intent_result=intent_result,
            entity_result=entity_result,
            saved_context=session.execution_context,
            saved_buffer=session.message_buffer,
            channel=channel,
        )

        # Handle result
        if result.status == "waiting":
            # Flow paused again — update session
            waiting_step = next(
                (s for s in result.steps_executed if s.status == "waiting"), None,
            )
            if waiting_step:
                from datetime import datetime, timezone, timedelta
                meta = waiting_step.metadata
                timeout_secs = meta.get("timeout_seconds", 0)
                session.paused_at_step = waiting_step.step_id
                session.expected_intents = meta.get("expected_intents", [])
                session.entity_extraction = meta.get("entity_extraction", [])
                session.timeout_at = (
                    datetime.now(timezone.utc) + timedelta(seconds=timeout_secs)
                    if timeout_secs > 0 else None
                )
                session.timeout_goto = meta.get("timeout_goto", "")
                session.resumed_count += 1
                self.dialogue_sessions.save(session)
        else:
            # Flow completed or aborted — remove session
            self.dialogue_sessions.remove(session.id)

        # Process context changes from the resumed flow
        if result.context_changes:
            await self._process_flow_context_changes(result, conversation.id)

        # Send the response
        response_text = result.rendered_message
        if response_text:
            adapter = self.channels.get(channel)
            if adapter:
                await adapter.send_message(contact, response_text)
            await self.context.record_outbound_message(
                conversation_id=conversation.id,
                channel=channel,
                content=response_text,
            )

        return {
            "conversation_id": conversation.id,
            "contact_id": contact.id,
            "channel": channel.value,
            "response": response_text,
            "flow_id": result.flow_id,
            "flow_status": result.status,
            "resumed_from_session": True,
            "intent_detected": intent_result.get("matched_intent"),
            "entities_extracted": entity_result,
        }

    async def _process_flow_context_changes(
        self,
        flow_result: FlowExecutionResult,
        conversation_id: str,
    ):
        """Process any context changes emitted by dialogue flow action steps."""
        for change_data in flow_result.context_changes:
            change = await self.context.build_context_change(
                conversation_id=conversation_id,
                trigger_type=change_data.get("trigger_type", "action"),
                trigger_value=change_data.get("trigger_value", ""),
                extra_data=change_data.get("extra_data", {}),
            )
            bindings = await self.context.get_conversation_bindings(conversation_id)
            transitions = self.state_machine.apply_trigger_to_all(
                bindings, change.trigger_type, change.trigger_value, change.context_data,
            )
            for t in transitions:
                await self.context.persist_binding(t.binding)

    async def _accumulate_interaction_memory(
        self,
        contact: Contact,
        conversation: Conversation,
        intent_data: dict[str, Any],
        user_message: str,
        channel: ChannelType,
        transition_results: list = None,
    ):
        """
        Automatically accumulate knowledge from every interaction.

        Stores:
        - Contact's preferred channel (based on response patterns)
        - Detected intents (pattern over time)
        - Sentiment trajectory
        - Key extracted entities
        - State transitions that occurred
        """
        transition_results = transition_results or []

        # Track channel preference
        self.memory_store.remember(
            scope_type="contact",
            scope_id=contact.id,
            key="last_active_channel",
            value=channel.value,
            mem_type=MemoryType.PREFERENCE,
        )

        # Track intent
        intent = intent_data.get("intent", "unknown")
        if intent != "unknown":
            self.memory_store.remember(
                scope_type="conversation",
                scope_id=conversation.id,
                key=f"intent_{intent}",
                value={
                    "intent": intent,
                    "confidence": intent_data.get("confidence", 0),
                    "message_snippet": user_message[:80],
                },
                mem_type=MemoryType.INTENT,
            )

        # Track sentiment
        sentiment = intent_data.get("sentiment", "neutral")
        self.memory_store.remember(
            scope_type="conversation",
            scope_id=conversation.id,
            key="latest_sentiment",
            value=sentiment,
            mem_type=MemoryType.OBSERVATION,
        )

        # Track extracted info
        extracted = intent_data.get("extracted_info", {})
        for key, value in extracted.items():
            if value:
                self.memory_store.remember(
                    scope_type="conversation",
                    scope_id=conversation.id,
                    key=f"extracted_{key}",
                    value=value,
                    mem_type=MemoryType.ENTITY,
                )

        # Track state transitions
        for result in transition_results:
            self.memory_store.remember(
                scope_type="conversation",
                scope_id=conversation.id,
                key=f"transition_{result.binding.process_type}",
                value={
                    "from": result.from_state,
                    "to": result.to_state,
                    "trigger": f"{intent_data.get('intent', 'unknown')}",
                },
                mem_type=MemoryType.EVENT,
            )

    @staticmethod
    def _get_active_process_type(results: list[TransitionResult]) -> str:
        """Extract the process_type from the most recent transition."""
        if results:
            return results[-1].binding.process_type
        return ""

    @staticmethod
    def _get_active_state(results: list[TransitionResult]) -> str:
        """Extract the current_state from the most recent transition."""
        if results:
            return results[-1].to_state
        return ""

    # ══════════════════════════════════════════════════════════
    #  OUTBOUND — Dispatch a follow-up via the best channel
    # ══════════════════════════════════════════════════════════

    async def dispatch_followup(self, followup: FollowUp) -> dict[str, Any]:
        """
        Execute a follow-up: select channel, create/resume conversation, send message.
        Called by the queue consumer when a job is ready.

        Also emits an "action:outreach_sent" context change so the state machine
        can track that outreach happened.
        """
        logger.info("dispatching_followup",
                     followup_id=followup.id,
                     contact_id=followup.contact_id,
                     attempt=followup.attempt_count + 1)

        # 1. Get contact
        contact = await self.context.store.get_contact(followup.contact_id)
        if not contact:
            raw_contact = await self.backend.get_contact(followup.contact_id)
            if not raw_contact:
                logger.error("contact_not_found", contact_id=followup.contact_id)
                followup.status = FollowUpStatus.FAILED
                followup.outcome = "Contact not found"
                await self.context.store.update_followup(followup)
                return {"status": "failed", "reason": "contact_not_found"}
            contact = self.backend.normalize_contact(raw_contact)
            await self.context.store.upsert_contact(contact)

        # 2. Find existing or create conversation
        context_key = followup.business_context.get("context_key", followup.id)
        conversation = await self.context.store.get_active_conversation(
            contact.id, context_key,
        )

        if not conversation:
            conversation = Conversation(
                contact_id=contact.id,
                followup_id=followup.id,
                status=ConversationStatus.PENDING,
                business_context=followup.business_context,
            )
            conversation = await self.context.store.create_conversation(conversation)
            followup.conversation_id = conversation.id

            # Auto-create state binding if process_type is specified
            process_type = followup.business_context.get("process_type", "")
            if process_type and self.state_machine.get_map(process_type):
                initial = followup.business_context.get("initial_state", "")
                binding = self.state_machine.create_binding(
                    process_type=process_type,
                    conversation_id=conversation.id,
                    contact_id=contact.id,
                    entity_id=followup.business_context.get("entity_id", ""),
                    entity_type=followup.business_context.get("entity_type", ""),
                    business_data=followup.business_context,
                    initial_state=initial,
                )
                if binding:
                    await self.context.bind_state(binding)

        # 3. Select channel
        channel = await self._select_channel(contact, followup, conversation)
        if not channel:
            logger.error("no_available_channel", contact_id=contact.id)
            followup.status = FollowUpStatus.FAILED
            followup.outcome = "No available channel"
            await self.context.store.update_followup(followup)
            return {"status": "failed", "reason": "no_channel"}

        # 4. Generate message
        conversation.active_channel = channel
        conversation.status = ConversationStatus.ACTIVE
        if channel not in conversation.channels_used:
            conversation.channels_used.append(channel)
        await self.context.store.update_conversation(conversation)

        full_context = await self.context.get_full_context(conversation.id)

        # Resolve and execute a dialogue flow for this follow-up
        template_name = followup.business_context.get("template", "")
        process_type = followup.business_context.get("process_type", "")
        current_state = followup.business_context.get("current_state", "")

        flow_result = await self._execute_dialogue_flow(
            template_name=template_name,
            process_type=process_type,
            current_state=current_state,
            context=full_context,
            channel=channel,
            conversation_id=conversation.id,
        )

        adapter = self.channels.get(channel)
        send_result = None

        if flow_result and flow_result.rendered_message:
            message_content = flow_result.rendered_message
            # Process context changes emitted by the flow
            await self._process_flow_context_changes(flow_result, conversation.id)
        else:
            # Fall back to LLM generation (no matching flow found)
            message_content = await self.engine.generate_response(full_context, channel)

        if adapter:
            send_result = await adapter.send_message(contact, message_content)

        # 5. Record and update state
        if send_result and send_result.get("status") not in ("failed",):
            await self.context.record_outbound_message(
                conversation.id, channel, message_content,
                metadata={"delivery": send_result},
            )
            followup.status = FollowUpStatus.AWAITING_REPLY
            followup.current_channel = channel
            followup.attempt_count += 1
            followup.started_at = followup.started_at or datetime.utcnow()
            followup.next_attempt_at = datetime.utcnow() + timedelta(hours=24)
            await self.context.store.update_followup(followup)

            # Emit "action:outreach_sent" context change so state machine knows
            change = await self.context.build_context_change(
                conversation_id=conversation.id,
                trigger_type="action",
                trigger_value="outreach_sent",
                extra_data={
                    "channel": channel.value,
                    "attempt": followup.attempt_count,
                    "template": template_name,
                },
                source_channel=channel,
            )
            bindings = await self.context.get_conversation_bindings(conversation.id)
            transitions = self.state_machine.apply_trigger_to_all(
                bindings, change.trigger_type, change.trigger_value, change.context_data,
            )
            for t in transitions:
                await self.context.persist_binding(t.binding)
                # Don't recurse into action execution for outreach_sent transitions
                # to avoid infinite loops. Just persist the state change.

            return {
                "status": "sent",
                "channel": channel.value,
                "conversation_id": conversation.id,
                "attempt": followup.attempt_count,
                "delivery": send_result,
            }
        else:
            return await self._handle_channel_failure(followup, contact, conversation, channel)

    # ══════════════════════════════════════════════════════════
    #  CHANNEL SELECTION
    # ══════════════════════════════════════════════════════════

    async def _select_channel(
        self,
        contact: Contact,
        followup: FollowUp,
        conversation: Conversation,
    ) -> Optional[ChannelType]:
        available = self.channels.get_available()

        # Continue on active channel if possible
        if conversation.active_channel and conversation.active_channel in available:
            if contact.get_channel(conversation.active_channel):
                return conversation.active_channel

        # Follow rule priority
        for ch in followup.channel_priority:
            if ch in available and contact.get_channel(ch):
                failed_channels = followup.metadata.get("failed_channels", [])
                if ch.value not in failed_channels:
                    return ch

        # Contact preference
        pref = contact.preferred_channel
        if pref and pref.channel in available:
            return pref.channel

        # Any available
        for ch in available:
            if contact.get_channel(ch):
                return ch

        return None

    async def _handle_channel_failure(
        self,
        followup: FollowUp,
        contact: Contact,
        conversation: Conversation,
        failed_channel: ChannelType,
    ) -> dict[str, Any]:
        failed = followup.metadata.setdefault("failed_channels", [])
        failed.append(failed_channel.value)
        await self.context.store.update_followup(followup)

        logger.warning("channel_failed_trying_next",
                        followup_id=followup.id,
                        failed=failed_channel.value)

        next_channel = await self._select_channel(contact, followup, conversation)
        if next_channel:
            followup.current_channel = next_channel
            return await self.dispatch_followup(followup)

        followup.status = FollowUpStatus.FAILED
        followup.outcome = f"All channels failed: {failed}"
        await self.context.store.update_followup(followup)

        # Emit max_attempts context change
        change = await self.context.build_context_change(
            conversation_id=conversation.id,
            trigger_type="timeout",
            trigger_value="all_channels_exhausted",
        )
        bindings = await self.context.get_conversation_bindings(conversation.id)
        transitions = self.state_machine.apply_trigger_to_all(
            bindings, change.trigger_type, change.trigger_value, change.context_data,
        )
        for t in transitions:
            await self.context.persist_binding(t.binding)
            await self._execute_actions(t.actions, conversation, contact, t)

        return {"status": "failed", "reason": "all_channels_exhausted", "failed": failed}

    # ══════════════════════════════════════════════════════════
    #  MAX ATTEMPTS
    # ══════════════════════════════════════════════════════════

    async def _handle_max_attempts(self, followup: FollowUp):
        escalation = followup.metadata.get("action", {}).get("escalation")
        if escalation:
            logger.info("escalating_followup", followup_id=followup.id)
            followup.status = FollowUpStatus.ESCALATED
        else:
            followup.status = FollowUpStatus.FAILED
            followup.outcome = f"Max attempts ({followup.max_attempts}) reached"

        await self.context.store.update_followup(followup)

        if followup.conversation_id:
            # Emit timeout trigger for state machine
            change = await self.context.build_context_change(
                conversation_id=followup.conversation_id,
                trigger_type="timeout",
                trigger_value="max_attempts_reached",
            )
            bindings = await self.context.get_conversation_bindings(followup.conversation_id)
            transitions = self.state_machine.apply_trigger_to_all(
                bindings, change.trigger_type, change.trigger_value, change.context_data,
            )
            for t in transitions:
                await self.context.persist_binding(t.binding)
                conv = await self.context.store.get_conversation(followup.conversation_id)
                contact = await self.context.store.get_contact(followup.contact_id)
                await self._execute_actions(t.actions, conv, contact, t)

    # ══════════════════════════════════════════════════════════
    #  QUEUE — Publish follow-ups for async dispatch
    # ══════════════════════════════════════════════════════════

    async def enqueue_followup(self, followup: FollowUp) -> QueueJob:
        job = QueueJob(
            followup_id=followup.id,
            rule_id=followup.rule_id,
            contact_id=followup.contact_id,
            priority=followup.priority.value,
            max_attempts=followup.max_attempts,
            scheduled_at=(
                followup.next_attempt_at.isoformat()
                if followup.next_attempt_at else datetime.utcnow().isoformat()
            ),
            business_context=followup.business_context,
            metadata=followup.metadata,
        )

        if followup.next_attempt_at and followup.next_attempt_at > datetime.utcnow():
            await self.queue.publish_delayed(job)
        else:
            await self.queue.publish(Queues.DISPATCH, job)

        return job

    async def enqueue_batch(self, followups: list[FollowUp]) -> list[QueueJob]:
        jobs = []
        for followup in followups:
            job = await self.enqueue_followup(followup)
            jobs.append(job)
        logger.info("batch_enqueued", count=len(jobs))
        return jobs
