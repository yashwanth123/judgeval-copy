"""
Rules system for Judgeval that enables alerts based on metric thresholds.
"""

from typing import Dict, List, Optional, Union, Any, Set
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime

class AlertStatus(str, Enum):
    """Status of an alert evaluation."""
    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"

class Operator(str, Enum):
    """Comparison operators for conditions."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="

class Condition(BaseModel):
    """
    A single metric condition.
    
    Example:
        {
            "metric": "faithfulness",
            "operator": ">=",
            "threshold": 0.7
        }
    """
    metric: str
    operator: Operator
    threshold: float

    def evaluate(self, value: float) -> bool:
        """Evaluate this condition against a value."""
        if self.operator == Operator.GT:
            return value > self.threshold
        elif self.operator == Operator.GTE:
            return value >= self.threshold
        elif self.operator == Operator.LT:
            return value < self.threshold
        elif self.operator == Operator.LTE:
            return value <= self.threshold
        elif self.operator == Operator.EQ:
            return value == self.threshold
        elif self.operator == Operator.NEQ:
            return value != self.threshold
        return False

class Rule(BaseModel):
    """
    Configuration for a single rule.
    
    Example:
        {
            "name": "Quality Check",
            "description": "Check if quality metrics meet thresholds",
            "conditions": [
                {"metric": "faithfulness", "operator": ">=", "threshold": 0.7},
                {"metric": "relevancy", "operator": ">=", "threshold": 0.8}
            ],
            "combine_type": "all"  # "all" or "any"
        }
    """
    name: str
    description: Optional[str] = None
    conditions: List[Condition]
    combine_type: str = Field(..., pattern="^(all|any)$")  # all = AND, any = OR

    @field_validator('conditions')
    def validate_conditions_not_empty(cls, v):
        if not v:
            raise ValueError("Conditions list cannot be empty")
        return v

class AlertResult(BaseModel):
    """
    Result of evaluating a rule.
    
    Example:
        {
            "status": "triggered",
            "rule_name": "Quality Check",
            "conditions_results": [
                {"metric": "faithfulness", "value": 0.6, "threshold": 0.7, "passed": False},
                {"metric": "relevancy", "value": 0.9, "threshold": 0.8, "passed": True}
            ],
            "example_id": "example_123",
            "timestamp": "20240321_123456"
        }
    """
    status: AlertStatus
    rule_name: str
    conditions_results: List[Dict[str, Any]]
    # Essential example metadata
    example_id: Optional[str] = None
    timestamp: Optional[str] = None

class RulesEngine:
    """
    Engine for evaluating rules and managing alerts.
    
    Example usage:
        rules = {
            "quality_check": Rule(
                name="Quality Check",
                conditions=[
                    Condition(metric="faithfulness", operator=">=", threshold=0.7),
                    Condition(metric="relevancy", operator=">=", threshold=0.8)
                ],
                combine_type="all"
            )
        }
        
        engine = RulesEngine(rules)
        scores = {"faithfulness": 0.8, "relevancy": 0.9}
        alerts = engine.evaluate_rules(scores, example_metadata={
            "example_id": "example_123",
            "timestamp": "20240321_123456"
        })
    """
    
    def __init__(self, rules: Dict[str, Rule]):
        """
        Initialize the RulesEngine with rules.
        
        Args:
            rules: Dictionary mapping rule IDs to rule configurations
        """
        self.rules = rules
    
    def evaluate_rules(self, scores: Dict[str, float], example_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, AlertResult]:
        """
        Evaluate all rules against a set of scores.
        Returns mapping of rule IDs to their alert results.
        
        Args:
            scores: Dictionary of metric names to their score values
            example_metadata: Optional dictionary containing example metadata (example_id, timestamp)
        """
        results = {}
        
        for rule_id, rule in self.rules.items():
            # Evaluate each condition
            condition_results = []
            passed_conditions = []
            
            for condition in rule.conditions:
                value = scores.get(condition.metric)
                if value is None:
                    passed = False
                else:
                    passed = condition.evaluate(value)
                    
                condition_results.append({
                    "metric": condition.metric,
                    "value": value,
                    "threshold": condition.threshold,
                    "operator": condition.operator,
                    "passed": passed
                })
                passed_conditions.append(passed)
            
            # Determine if alert should trigger
            triggered = all(passed_conditions) if rule.combine_type == "all" else any(passed_conditions)
            
            # Create alert result with example metadata
            alert_result = AlertResult(
                status=AlertStatus.TRIGGERED if triggered else AlertStatus.NOT_TRIGGERED,
                rule_name=rule.name,
                conditions_results=condition_results
            )
            
            # Add example metadata if provided
            if example_metadata:
                if "example_id" in example_metadata:
                    alert_result.example_id = example_metadata["example_id"]
                if "timestamp" in example_metadata:
                    alert_result.timestamp = example_metadata["timestamp"]
            
            results[rule_id] = alert_result
            
        return results 