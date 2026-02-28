"""
Dialogue Flow Registry — Loads, validates, and resolves agentic templates.

Templates are loaded from YAML config and indexed by id, tags, and
process_type so the orchestrator can find the right flow for any
given context.

Resolution order:
  1. Exact match by flow id (from TransitionAction.template)
  2. Tag match (e.g. process_type + state)
  3. Default flow for the process_type
  4. Global fallback
"""
from __future__ import annotations

import structlog
from typing import Any, Optional

from models.schemas import ChannelType, RuleCondition
from templates.models import (
    DialogueFlow, DialogueStep, StepType, ChannelVariant,
    VariableDef, BranchArm, ToolDef,
)
from utils.conditions import evaluate_conditions

logger = structlog.get_logger()


class DialogueFlowRegistry:
    """
    Central registry for all dialogue flow templates.
    Provides resolution by id, tags, and contextual guard evaluation.
    """

    def __init__(self):
        self._flows: dict[str, DialogueFlow] = {}
        self._tag_index: dict[str, list[str]] = {}       # tag → [flow_ids]
        self._process_index: dict[str, list[str]] = {}   # process_type → [flow_ids]

    # ── Registration ──────────────────────────────────

    def register(self, flow: DialogueFlow):
        """Register a single dialogue flow."""
        errors = self._validate(flow)
        if errors:
            logger.error("invalid_dialogue_flow",
                         flow_id=flow.id, errors=errors)
            raise ValueError(f"Invalid dialogue flow '{flow.id}': {'; '.join(errors)}")

        self._flows[flow.id] = flow

        for tag in flow.tags:
            self._tag_index.setdefault(tag, []).append(flow.id)

        # Index by process_type tag for quick lookup
        process_tags = [t for t in flow.tags if t.startswith("process:")]
        for pt in process_tags:
            process_type = pt.split(":", 1)[1]
            self._process_index.setdefault(process_type, []).append(flow.id)

        logger.info("dialogue_flow_registered",
                     flow_id=flow.id,
                     name=flow.name,
                     steps=len(flow.steps),
                     tags=flow.tags)

    def register_from_config(self, config: list[dict[str, Any]]):
        """Load dialogue flows from YAML config."""
        for raw in config:
            flow = self._parse_flow(raw)
            self.register(flow)
        logger.info("dialogue_flows_loaded", count=len(config))

    # ── Resolution ────────────────────────────────────

    def get(self, flow_id: str) -> Optional[DialogueFlow]:
        """Get a flow by exact id."""
        return self._flows.get(flow_id)

    def resolve(
        self,
        template_name: str = "",
        process_type: str = "",
        current_state: str = "",
        context_data: dict[str, Any] = None,
        channel: ChannelType = None,
    ) -> Optional[DialogueFlow]:
        """
        Resolve the best dialogue flow for the given context.

        Resolution order:
        1. Exact match by template_name
        2. Match by process_type + state tag with guard evaluation
        3. Match by process_type with guard evaluation
        4. Global "default" flow
        """
        context_data = context_data or {}

        # 1. Exact match
        if template_name and template_name in self._flows:
            flow = self._flows[template_name]
            if self._passes_guards(flow, context_data):
                return flow

        # 2. Process + state match
        if process_type and current_state:
            state_tag = f"state:{current_state}"
            candidates = self._get_process_candidates(process_type)
            for flow in candidates:
                if state_tag in flow.tags and self._passes_guards(flow, context_data):
                    return flow

        # 3. Process match
        if process_type:
            candidates = self._get_process_candidates(process_type)
            for flow in candidates:
                if self._passes_guards(flow, context_data):
                    return flow

        # 4. Exact template_name match (without guard check — fallback)
        if template_name and template_name in self._flows:
            return self._flows[template_name]

        # 5. Global default
        return self._flows.get("default")

    def resolve_all_matching(
        self,
        process_type: str = "",
        context_data: dict[str, Any] = None,
    ) -> list[DialogueFlow]:
        """Return all flows whose guards pass for the given context."""
        context_data = context_data or {}
        candidates = self._get_process_candidates(process_type) if process_type else self.list_all()
        return [f for f in candidates if self._passes_guards(f, context_data)]

    def list_all(self) -> list[DialogueFlow]:
        return list(self._flows.values())

    def list_by_tag(self, tag: str) -> list[DialogueFlow]:
        flow_ids = self._tag_index.get(tag, [])
        return [self._flows[fid] for fid in flow_ids if fid in self._flows]

    # ── Guard Evaluation ──────────────────────────────

    def _passes_guards(self, flow: DialogueFlow, context_data: dict[str, Any]) -> bool:
        if not flow.guards:
            return True
        return evaluate_conditions(flow.guards, context_data)

    def _get_process_candidates(self, process_type: str) -> list[DialogueFlow]:
        flow_ids = self._process_index.get(process_type, [])
        flows = [self._flows[fid] for fid in flow_ids if fid in self._flows]
        return sorted(flows, key=lambda f: f.priority, reverse=True)

    # ── Validation ────────────────────────────────────

    @staticmethod
    def _validate(flow: DialogueFlow) -> list[str]:
        errors = []
        step_ids = {s.id for s in flow.steps}

        if not flow.id:
            errors.append("flow id is required")
        if not flow.steps:
            errors.append("flow must have at least one step")

        if flow.entry_step and flow.entry_step not in step_ids:
            errors.append(f"entry_step '{flow.entry_step}' not found in steps")

        for step in flow.steps:
            if step.next and step.next not in step_ids:
                errors.append(f"step '{step.id}' references unknown next '{step.next}'")

            if step.type == StepType.BRANCH:
                for arm in step.arms:
                    if arm.goto not in step_ids:
                        errors.append(f"branch arm in '{step.id}' references unknown goto '{arm.goto}'")

            if step.type == StepType.GATE and step.fail_goto and step.fail_goto not in step_ids:
                errors.append(f"gate '{step.id}' references unknown fail_goto '{step.fail_goto}'")

            if step.timeout_goto and step.timeout_goto not in step_ids:
                errors.append(f"step '{step.id}' references unknown timeout_goto '{step.timeout_goto}'")

        return errors

    # ── Parsing ───────────────────────────────────────

    def _parse_flow(self, raw: dict[str, Any]) -> DialogueFlow:
        """Parse a raw dict (from YAML) into a DialogueFlow model."""
        steps = []
        for raw_step in raw.get("steps", []):
            step = self._parse_step(raw_step)
            steps.append(step)

        variables = [VariableDef(**v) for v in raw.get("variables", [])]
        guards = [RuleCondition(**c) for c in raw.get("guards", [])]

        channel_config = {}
        for ch_name, ch_data in raw.get("default_channel_config", {}).items():
            channel_config[ch_name] = ChannelVariant(channel=ChannelType(ch_name), **ch_data)

        return DialogueFlow(
            id=raw["id"],
            name=raw.get("name", raw["id"]),
            description=raw.get("description", ""),
            version=raw.get("version", "1.0"),
            variables=variables,
            guards=guards,
            priority=raw.get("priority", 0),
            default_channel_config=channel_config,
            steps=steps,
            entry_step=raw.get("entry_step", ""),
            tags=raw.get("tags", []),
            metadata=raw.get("metadata", {}),
        )

    def _parse_step(self, raw: dict[str, Any]) -> DialogueStep:
        """Parse a raw dict into a DialogueStep."""
        # Parse nested objects
        tool = None
        if raw.get("tool"):
            tool = ToolDef(**raw["tool"])

        arms = [BranchArm(**a) for a in raw.get("arms", [])]
        conditions = [RuleCondition(**c) for c in raw.get("conditions", [])]
        channel_variants = [
            ChannelVariant(channel=ChannelType(v.get("channel", "chat")), **{k: v2 for k, v2 in v.items() if k != "channel"})
            for v in raw.get("channel_variants", [])
        ]

        return DialogueStep(
            id=raw["id"],
            type=StepType(raw["type"]),
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
            channel_variants=channel_variants,
        )
