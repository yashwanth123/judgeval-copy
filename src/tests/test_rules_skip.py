"""
Tests for the Rule Engine's ability to skip missing metrics.
"""

import pytest
from uuid import uuid4
from judgeval.rules import Rule, Condition, Operator, AlertStatus, RulesEngine


def test_skip_missing_metrics_all():
    """Test that missing metrics are skipped in 'all' combine type rules."""
    rules = {
        "quality_check": Rule(
            name="Quality Check",
            conditions=[
                Condition(metric="faithfulness", operator=Operator.GTE, threshold=0.7),
                Condition(metric="relevancy", operator=Operator.GTE, threshold=0.8)
            ],
            combine_type="all"
        )
    }
    
    engine = RulesEngine(rules)
    
    # Test when one metric is missing
    scores = {"faithfulness": 0.8}  # Missing relevancy
    results = engine.evaluate_rules(scores)
    
    assert "quality_check" in results
    assert results["quality_check"].status == AlertStatus.TRIGGERED  # Should trigger based on faithfulness only
    
    # Check condition results
    condition_results = results["quality_check"].conditions_result
    assert len(condition_results) == 2
    
    # Find the missing metric condition
    relevancy_condition = next((c for c in condition_results if c["metric"] == "relevancy"), None)
    assert relevancy_condition is not None
    assert relevancy_condition["value"] is None
    assert relevancy_condition["passed"] is None
    assert relevancy_condition["skipped"] is True
    
    # Find the existing metric condition
    faithfulness_condition = next((c for c in condition_results if c["metric"] == "faithfulness"), None)
    assert faithfulness_condition is not None
    assert faithfulness_condition["value"] == 0.8
    assert faithfulness_condition["passed"] is True
    assert faithfulness_condition["skipped"] is False


def test_skip_missing_metrics_any():
    """Test that missing metrics are skipped in 'any' combine type rules."""
    rules = {
        "quality_check": Rule(
            name="Quality Check",
            conditions=[
                Condition(metric="faithfulness", operator=Operator.GTE, threshold=0.7),
                Condition(metric="relevancy", operator=Operator.GTE, threshold=0.8)
            ],
            combine_type="any"
        )
    }
    
    engine = RulesEngine(rules)
    
    # Test when one metric is missing but other passes
    scores = {"faithfulness": 0.8}  # Missing relevancy
    results = engine.evaluate_rules(scores)
    
    assert "quality_check" in results
    assert results["quality_check"].status == AlertStatus.TRIGGERED  # Should trigger based on faithfulness only
    
    # Test when one metric is missing and other fails
    scores = {"faithfulness": 0.6}  # Missing relevancy and faithfulness fails
    results = engine.evaluate_rules(scores)
    
    assert "quality_check" in results
    assert results["quality_check"].status == AlertStatus.NOT_TRIGGERED


def test_all_metrics_missing():
    """Test behavior when all metrics are missing."""
    rules = {
        "quality_check": Rule(
            name="Quality Check",
            conditions=[
                Condition(metric="faithfulness", operator=Operator.GTE, threshold=0.7),
                Condition(metric="relevancy", operator=Operator.GTE, threshold=0.8)
            ],
            combine_type="all"
        ),
        "any_check": Rule(
            name="Any Check",
            conditions=[
                Condition(metric="faithfulness", operator=Operator.GTE, threshold=0.7),
                Condition(metric="relevancy", operator=Operator.GTE, threshold=0.8)
            ],
            combine_type="any"
        )
    }
    
    engine = RulesEngine(rules)
    
    # Test with empty scores
    scores = {}  # All metrics missing
    results = engine.evaluate_rules(scores)
    
    assert "quality_check" in results
    assert "any_check" in results
    
    # Both rules should not trigger since there are no metrics to evaluate
    assert results["quality_check"].status == AlertStatus.NOT_TRIGGERED
    assert results["any_check"].status == AlertStatus.NOT_TRIGGERED
    
    # All conditions should be marked as skipped
    for rule_id in ["quality_check", "any_check"]:
        for condition in results[rule_id].conditions_result:
            assert condition["passed"] is None
            assert condition["skipped"] is True


def test_mixed_conditions():
    """Test with a mix of missing, passing, and failing metrics."""
    rules = {
        "mixed_rule": Rule(
            name="Mixed Rule",
            conditions=[
                Condition(metric="metric1", operator=Operator.GTE, threshold=0.7),
                Condition(metric="metric2", operator=Operator.GTE, threshold=0.7),
                Condition(metric="metric3", operator=Operator.GTE, threshold=0.7)
            ],
            combine_type="all"
        )
    }
    
    engine = RulesEngine(rules)
    
    # Test with one missing, one passing, one failing
    scores = {
        "metric1": 0.8,  # Pass
        "metric3": 0.6   # Fail
        # metric2 is missing
    }
    
    results = engine.evaluate_rules(scores)
    
    assert "mixed_rule" in results
    assert results["mixed_rule"].status == AlertStatus.NOT_TRIGGERED  # Should not trigger due to metric3
    
    # Check each condition
    condition_results = results["mixed_rule"].conditions_result
    
    metric1_condition = next((c for c in condition_results if c["metric"] == "metric1"), None)
    assert metric1_condition["passed"] is True
    assert metric1_condition["skipped"] is False
    
    metric2_condition = next((c for c in condition_results if c["metric"] == "metric2"), None)
    assert metric2_condition["passed"] is None
    assert metric2_condition["skipped"] is True
    
    metric3_condition = next((c for c in condition_results if c["metric"] == "metric3"), None)
    assert metric3_condition["passed"] is False
    assert metric3_condition["skipped"] is False 