"""
Rules system for Judgeval that enables alerts based on metric thresholds.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import uuid  # Add import for uuid module

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
            "metric": "faithfulness" (note that metric can only be APIScorer, not JudgevalScorer)
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
            "rule_id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "Quality Check",
            "description": "Check if quality metrics meet thresholds",
            "conditions": [
                {"metric": "faithfulness", "operator": ">=", "threshold": 0.7},
                {"metric": "relevancy", "operator": ">=", "threshold": 0.8}
            ],
            "combine_type": "all"  # "all" or "any"
        }
    """
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Random UUID string as default value
    name: str
    description: Optional[str] = None
    conditions: List[Condition]
    combine_type: str = Field(..., pattern="^(all|any)$")  # all = AND, any = OR

    @field_validator('conditions')
    def validate_conditions_not_empty(cls, v):
        if not v:
            raise ValueError("Conditions list cannot be empty")
        return v

    @field_validator('combine_type')
    def validate_combine_type(cls, v):
        if v not in ["all", "any"]:
            raise ValueError(f"combine_type must be 'all' or 'any', got: {v}")
        return v


class AlertResult(BaseModel):
    """
    Result of evaluating a rule.
    
    Example:
        {
            "status": "triggered",
            "rule_name": "Quality Check",
            "conditions_result": [
                {"metric": "faithfulness", "value": 0.6, "threshold": 0.7, "passed": False},
                {"metric": "relevancy", "value": 0.9, "threshold": 0.8, "passed": True}
            ],
            "rule_id": "123e4567-e89b-12d3-a456-426614174000",
            "metadata": {
                "example_id": "example_123",
                "timestamp": "20240321_123456"
            }
        }
    """
    status: AlertStatus
    rule_id: Optional[str] = None  # The unique identifier of the rule
    rule_name: str
    conditions_result: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    
    @property
    def example_id(self) -> Optional[str]:
        """Get example_id from metadata for backward compatibility"""
        return self.metadata.get("example_id")
        
    @property
    def timestamp(self) -> Optional[str]:
        """Get timestamp from metadata for backward compatibility"""
        return self.metadata.get("timestamp")

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
                rule_id=rule.rule_id,  # Include the rule's unique identifier
                rule_name=rule.name,
                conditions_result=condition_results
            )
            
            # Add example metadata if provided
            if example_metadata:
                if "example_id" in example_metadata:
                    alert_result.metadata["example_id"] = example_metadata["example_id"]
                if "timestamp" in example_metadata:
                    alert_result.metadata["timestamp"] = example_metadata["timestamp"]
            
            results[rule_id] = alert_result
            
        return results
    
    async def evaluate_rules_parallel(self, 
                               example_scores: Dict[str, Dict[str, float]], 
                               example_metadata: Dict[str, Dict[str, Any]],
                               max_concurrent: int = 100) -> Dict[str, Dict[str, AlertResult]]:
        """
        Evaluate all rules against multiple examples in parallel.
        
        Args:
            example_scores: Dictionary mapping example_ids to their score dictionaries
            example_metadata: Dictionary mapping example_ids to their metadata
            max_concurrent: Maximum number of concurrent evaluations
            
        Returns:
            Dictionary mapping example_ids to dictionaries of rule_ids and their alert results
        """
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        tasks = []
        
        # Create a task for each example
        for example_id, scores in example_scores.items():
            metadata = example_metadata.get(example_id, {})
            task = self._evaluate_with_semaphore(
                semaphore=semaphore,
                example_id=example_id,
                scores=scores,
                metadata=metadata
            )
            tasks.append(task)
        
        # Run all tasks and collect results
        example_results = await asyncio.gather(*tasks)
        
        # Organize results by example_id
        for example_id, result in example_results:
            results[example_id] = result
            
        return results
    
    async def _evaluate_with_semaphore(self, 
                                semaphore: asyncio.Semaphore, 
                                example_id: str, 
                                scores: Dict[str, float], 
                                metadata: Dict[str, Any]) -> Tuple[str, Dict[str, AlertResult]]:
        """
        Helper method to evaluate rules for an example with semaphore control.
        
        Args:
            semaphore: Semaphore to control concurrency
            example_id: ID of the example being evaluated
            scores: Dictionary of scores for this example
            metadata: Metadata for this example
            
        Returns:
            Tuple of (example_id, rule_results)
        """
        async with semaphore:
            # Run the evaluation in a thread pool to avoid blocking the event loop
            # for CPU-bound operations
            with ThreadPoolExecutor() as executor:
                start_time = time.perf_counter()
                rule_results = await asyncio.get_event_loop().run_in_executor(
                    executor, 
                    self.evaluate_rules,
                    scores,
                    metadata
                )
                end_time = time.perf_counter()
                
                # Could log performance metrics here if needed
                # debug(f"Rule evaluation for example {example_id} took {end_time - start_time:.4f} seconds")
                
                return (example_id, rule_results) 