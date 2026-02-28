"""
Tool Registry — Declarative capability catalog for agentic dialogue flows.

The tool registry is the "menu" the agentic planner reads when deciding
what to do next. Every tool has:
  - A name and description (for the LLM to understand purpose)
  - A JSON schema for input parameters
  - A schema for outputs (what keys will be available after invocation)
  - Guard conditions (when is this tool applicable?)
  - Cost/latency hints (so the planner can make smart choices)

Tools come from three sources:
  1. Built-in tools (business hours check, attempt summary, etc.)
  2. Backend connector endpoints (registered dynamically)
  3. Custom tools registered by integration code

The planner uses the registry to:
  - Enumerate available actions when composing ad-hoc plans
  - Validate tool calls before execution
  - Build LLM prompts that describe available capabilities
"""
from __future__ import annotations

import structlog
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field
from enum import Enum

logger = structlog.get_logger()


class ToolCategory(str, Enum):
    """Categorize tools for the planner's benefit."""
    DATA_FETCH = "data_fetch"          # Read data from backend
    DATA_WRITE = "data_write"          # Write/update data
    COMMUNICATION = "communication"    # Send messages, notifications
    ANALYSIS = "analysis"              # Analyze/classify/score
    INTERNAL = "internal"              # System checks (business hours, etc.)
    FLOW_CONTROL = "flow_control"      # Affect dialogue flow (escalate, resolve)


class ToolSchema(BaseModel):
    """Describes one callable tool for the planner."""
    name: str                                             # Unique identifier
    description: str                                      # What the tool does (for LLM)
    category: ToolCategory = ToolCategory.DATA_FETCH
    endpoint: str = ""                                    # Backend endpoint or $builtin name

    # Input/output schemas (for validation + LLM understanding)
    input_schema: dict[str, Any] = {}                     # JSON Schema for parameters
    output_schema: dict[str, Any] = {}                    # JSON Schema for return value
    output_keys: list[str] = []                           # Key names available after call

    # Guards — when is this tool applicable?
    required_context_keys: list[str] = []                 # Context must have these keys
    applicable_process_types: list[str] = []              # Empty = all processes
    applicable_states: list[str] = []                     # Empty = all states

    # Execution hints
    timeout_seconds: int = 10
    is_idempotent: bool = True                            # Safe to retry?
    estimated_latency_ms: int = 100                       # For planner prioritization
    cost_tier: str = "free"                               # free | low | medium | high (LLM calls = high)

    # Runtime
    is_builtin: bool = False                              # True for $-prefixed builtins
    enabled: bool = True

    def matches_context(self, process_type: str = "", state: str = "") -> bool:
        """Check if this tool is applicable for the given context."""
        if not self.enabled:
            return False
        if self.applicable_process_types and process_type not in self.applicable_process_types:
            return False
        if self.applicable_states and state not in self.applicable_states:
            return False
        return True


class ToolRegistry:
    """
    Central catalog of all tools available to dialogue flows and the planner.

    Used by:
    - DialogueExecutor: to validate tool_call steps
    - AgenticPlanner: to enumerate capabilities when composing plans
    - Registry validation: to check flow references at load time
    """

    def __init__(self):
        self._tools: dict[str, ToolSchema] = {}
        self._handlers: dict[str, Callable] = {}        # name → async callable

    # ── Registration ──────────────────────────────────

    def register(self, schema: ToolSchema, handler: Callable = None):
        """Register a tool with optional handler."""
        self._tools[schema.name] = schema
        if handler:
            self._handlers[schema.name] = handler
        logger.info("tool_registered",
                     name=schema.name,
                     category=schema.category.value,
                     builtin=schema.is_builtin)

    def register_backend_endpoint(
        self,
        name: str,
        endpoint: str,
        description: str,
        input_schema: dict = None,
        output_keys: list[str] = None,
        category: ToolCategory = ToolCategory.DATA_FETCH,
        **kwargs,
    ):
        """Convenience: register a backend connector endpoint as a tool."""
        schema = ToolSchema(
            name=name,
            description=description,
            category=category,
            endpoint=endpoint,
            input_schema=input_schema or {},
            output_keys=output_keys or [],
            **kwargs,
        )
        self.register(schema)

    def register_builtin(
        self,
        name: str,
        description: str,
        handler: Callable,
        output_keys: list[str] = None,
        **kwargs,
    ):
        """Convenience: register a built-in async function as a tool."""
        schema = ToolSchema(
            name=name,
            description=description,
            category=ToolCategory.INTERNAL,
            endpoint=name,
            is_builtin=True,
            output_keys=output_keys or [],
            **kwargs,
        )
        self.register(schema, handler)

    # ── Lookup ────────────────────────────────────────

    def get(self, name: str) -> Optional[ToolSchema]:
        return self._tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable]:
        return self._handlers.get(name)

    def list_all(self) -> list[ToolSchema]:
        return [t for t in self._tools.values() if t.enabled]

    def list_for_context(
        self,
        process_type: str = "",
        state: str = "",
        categories: list[ToolCategory] = None,
    ) -> list[ToolSchema]:
        """Return tools applicable to the given context."""
        tools = []
        for t in self._tools.values():
            if not t.enabled:
                continue
            if not t.matches_context(process_type, state):
                continue
            if categories and t.category not in categories:
                continue
            tools.append(t)
        return tools

    def describe_for_llm(
        self,
        process_type: str = "",
        state: str = "",
        max_tools: int = 20,
    ) -> str:
        """
        Build a human-readable description of available tools
        for inclusion in LLM prompts.
        """
        tools = self.list_for_context(process_type, state)[:max_tools]
        if not tools:
            return "No tools available."

        lines = ["Available tools:"]
        for t in tools:
            params = ""
            if t.input_schema:
                props = t.input_schema.get("properties", {})
                if props:
                    param_strs = [f"{k}: {v.get('type','any')}" for k, v in props.items()]
                    params = f" ({', '.join(param_strs)})"
            outputs = ""
            if t.output_keys:
                outputs = f" → {', '.join(t.output_keys)}"
            lines.append(f"  • {t.name}{params}{outputs}")
            lines.append(f"    {t.description}")
            if t.applicable_states:
                lines.append(f"    [states: {', '.join(t.applicable_states)}]")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._tools)

    # ── Validation ────────────────────────────────────

    def validate_tool_reference(self, tool_name: str) -> Optional[str]:
        """Check if a tool reference in a flow is valid. Returns error or None."""
        # Built-in references start with $
        if tool_name.startswith("$"):
            if tool_name not in self._tools:
                return f"Unknown built-in tool: {tool_name}"
            return None
        if tool_name not in self._tools:
            return f"Unknown tool: {tool_name}"
        return None


def create_default_tool_registry() -> ToolRegistry:
    """Create a registry pre-loaded with standard built-in tools."""
    registry = ToolRegistry()

    registry.register(ToolSchema(
        name="$check_business_hours",
        description="Check if we are currently within business hours (9AM-6PM local time)",
        category=ToolCategory.INTERNAL,
        endpoint="$check_business_hours",
        is_builtin=True,
        output_keys=["is_business_hours", "local_hour", "recommendation"],
        estimated_latency_ms=1,
    ))

    registry.register(ToolSchema(
        name="$get_conversation_age",
        description="Get how long this conversation has been active (seconds, hours, days)",
        category=ToolCategory.INTERNAL,
        endpoint="$get_conversation_age",
        is_builtin=True,
        output_keys=["age_seconds", "age_hours", "age_days"],
        estimated_latency_ms=1,
    ))

    registry.register(ToolSchema(
        name="$get_attempt_summary",
        description="Get follow-up attempt count, whether this is the first or last attempt, and urgency level",
        category=ToolCategory.INTERNAL,
        endpoint="$get_attempt_summary",
        is_builtin=True,
        output_keys=["attempt_count", "max_attempts", "is_first_attempt",
                      "is_final_attempt", "remaining_attempts", "urgency"],
        estimated_latency_ms=1,
    ))

    return registry
