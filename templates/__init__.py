"""
Agentic Dialogue Template System.

Templates are executable conversation plans — not static text.
Each template is a directed graph of steps that the executor
walks through, making real-time decisions at each node.

The agentic layer adds:
  - Dynamic planning (LLM composes flows on the fly)
  - Sub-flow composition (flows invoke other flows)
  - Cross-flow memory (knowledge persists across executions)
  - Tool discovery (planner enumerates available capabilities)
  - Parallel execution (concurrent tool calls)
  - Lifecycle hooks (on_enter, on_exit, on_error)
"""
from templates.models import (
    DialogueFlow, DialogueStep, StepType, ChannelVariant,
    VariableDef, BranchArm, ToolDef,
    StepResult, FlowExecutionResult,
)
from templates.registry import DialogueFlowRegistry
from templates.executor import DialogueExecutor
from templates.tool_registry import ToolRegistry, ToolSchema, ToolCategory
from templates.memory import FlowMemoryStore, MemoryEntry, MemoryType
from templates.planner import AgenticPlanner, PlanObjective
