"""Tests for the shared condition evaluator."""
import pytest
from models.schemas import RuleCondition
from utils.conditions import evaluate_condition, evaluate_conditions, get_nested_value


class TestGetNestedValue:
    def test_flat_key(self):
        assert get_nested_value({"name": "Alice"}, "name") == "Alice"

    def test_nested_key(self):
        data = {"order": {"status": "shipped", "items": 3}}
        assert get_nested_value(data, "order.status") == "shipped"
        assert get_nested_value(data, "order.items") == 3

    def test_missing_key(self):
        assert get_nested_value({"a": 1}, "b") is None

    def test_missing_nested_key(self):
        assert get_nested_value({"a": {"b": 1}}, "a.c") is None

    def test_deep_nesting(self):
        data = {"a": {"b": {"c": {"d": 42}}}}
        assert get_nested_value(data, "a.b.c.d") == 42


class TestEvaluateCondition:
    def test_eq(self):
        cond = RuleCondition(field="status", operator="eq", value="active")
        assert evaluate_condition(cond, {"status": "active"})
        assert not evaluate_condition(cond, {"status": "inactive"})

    def test_neq(self):
        cond = RuleCondition(field="status", operator="neq", value="closed")
        assert evaluate_condition(cond, {"status": "active"})
        assert not evaluate_condition(cond, {"status": "closed"})

    def test_gt(self):
        cond = RuleCondition(field="amount", operator="gt", value=100)
        assert evaluate_condition(cond, {"amount": 200})
        assert not evaluate_condition(cond, {"amount": 50})
        assert not evaluate_condition(cond, {"amount": 100})

    def test_gte(self):
        cond = RuleCondition(field="amount", operator="gte", value=100)
        assert evaluate_condition(cond, {"amount": 100})
        assert evaluate_condition(cond, {"amount": 200})
        assert not evaluate_condition(cond, {"amount": 50})

    def test_lt(self):
        cond = RuleCondition(field="score", operator="lt", value=50)
        assert evaluate_condition(cond, {"score": 30})
        assert not evaluate_condition(cond, {"score": 60})

    def test_lte(self):
        cond = RuleCondition(field="score", operator="lte", value=50)
        assert evaluate_condition(cond, {"score": 50})
        assert not evaluate_condition(cond, {"score": 51})

    def test_in(self):
        cond = RuleCondition(field="tier", operator="in", value=["gold", "platinum"])
        assert evaluate_condition(cond, {"tier": "gold"})
        assert not evaluate_condition(cond, {"tier": "silver"})

    def test_contains(self):
        cond = RuleCondition(field="name", operator="contains", value="Kumar")
        assert evaluate_condition(cond, {"name": "Rajesh Kumar"})
        assert not evaluate_condition(cond, {"name": "Priya Sharma"})

    def test_regex(self):
        cond = RuleCondition(field="email", operator="regex", value=r"@.*\.com$")
        assert evaluate_condition(cond, {"email": "test@example.com"})
        assert not evaluate_condition(cond, {"email": "test@example.org"})

    def test_exists(self):
        cond = RuleCondition(field="phone", operator="exists", value=True)
        assert evaluate_condition(cond, {"phone": "+91123"})
        assert not evaluate_condition(cond, {"email": "a@b.com"})

    def test_not_exists(self):
        cond = RuleCondition(field="phone", operator="not_exists", value=True)
        assert evaluate_condition(cond, {"email": "a@b.com"})
        assert not evaluate_condition(cond, {"phone": "+91123"})

    def test_type_coercion_string_to_float(self):
        cond = RuleCondition(field="amount", operator="gt", value=100)
        assert evaluate_condition(cond, {"amount": "200"})

    def test_nested_field(self):
        cond = RuleCondition(field="order.total", operator="gte", value=500)
        assert evaluate_condition(cond, {"order": {"total": 750}})
        assert not evaluate_condition(cond, {"order": {"total": 100}})

    def test_invalid_operator(self):
        cond = RuleCondition(field="x", operator="invalid_op", value=1)
        assert not evaluate_condition(cond, {"x": 1})

    def test_type_error_returns_false(self):
        cond = RuleCondition(field="x", operator="gt", value=10)
        assert not evaluate_condition(cond, {"x": "not_a_number"})


class TestEvaluateConditions:
    def test_empty_conditions(self):
        assert evaluate_conditions([], {"anything": True})

    def test_all_pass(self):
        conditions = [
            RuleCondition(field="amount", operator="gt", value=100),
            RuleCondition(field="status", operator="eq", value="overdue"),
        ]
        assert evaluate_conditions(conditions, {"amount": 200, "status": "overdue"})

    def test_one_fails(self):
        conditions = [
            RuleCondition(field="amount", operator="gt", value=100),
            RuleCondition(field="status", operator="eq", value="overdue"),
        ]
        assert not evaluate_conditions(conditions, {"amount": 200, "status": "paid"})

    def test_all_fail(self):
        conditions = [
            RuleCondition(field="amount", operator="gt", value=1000),
            RuleCondition(field="status", operator="eq", value="overdue"),
        ]
        assert not evaluate_conditions(conditions, {"amount": 50, "status": "paid"})
