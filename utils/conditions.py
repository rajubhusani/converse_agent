"""
Shared condition evaluator — used by both the Rules Engine and the State Machine.

Evaluates RuleCondition objects against data dictionaries.
Supports nested dot-notation field access and type coercion.
"""
from __future__ import annotations

import re
import operator as op
from typing import Any

from models.schemas import RuleCondition


OPERATORS: dict[str, Any] = {
    "eq": op.eq,
    "neq": op.ne,
    "gt": op.gt,
    "gte": op.ge,
    "lt": op.lt,
    "lte": op.le,
    "in": lambda a, b: a in b,
    "contains": lambda a, b: b in str(a),
    "regex": lambda a, b: bool(re.search(str(b), str(a))),
    "exists": lambda a, b: a is not None,
    "not_exists": lambda a, b: a is None,
}


def get_nested_value(data: dict, field: str) -> Any:
    """Get a value from nested dict using dot notation. e.g. 'order.status'"""
    current = data
    for part in field.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def evaluate_condition(condition: RuleCondition, data: dict[str, Any]) -> bool:
    """Evaluate a single condition against data."""
    val = get_nested_value(data, condition.field)
    fn = OPERATORS.get(condition.operator)
    if fn is None:
        return False
    try:
        if isinstance(condition.value, (int, float)) and not isinstance(condition.value, bool) and isinstance(val, str):
            val = float(val)
        return fn(val, condition.value)
    except (TypeError, ValueError):
        return False


def evaluate_conditions(conditions: list[RuleCondition], data: dict[str, Any]) -> bool:
    """Evaluate all conditions (AND logic). Returns True if all pass."""
    if not conditions:
        return True
    return all(evaluate_condition(c, data) for c in conditions)
