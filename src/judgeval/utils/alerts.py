"""
Handling alerts in Judgeval.
"""
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class AlertStatus(str, Enum):
    """Status of an alert evaluation."""
    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"

class AlertResult(BaseModel):
    """
    Result of a rule evaluation.
    
    Attributes:
        rule_name: Name of the rule that was evaluated
        status: Status of the alert (triggered or not)
        conditions_results: List of condition evaluation results
        example_id: Optional ID of the example that triggered the alert
        timestamp: Optional timestamp of when the alert was created
    """
    rule_name: str
    status: AlertStatus
    conditions_results: List[Dict[str, Any]] = []
    example_id: Optional[str] = None
    timestamp: Optional[str] = None
