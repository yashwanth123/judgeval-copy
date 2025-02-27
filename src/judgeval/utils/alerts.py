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
        conditions_result: List of condition evaluation results
        metadata: Dictionary containing example_id, timestamp, and other metadata
    """
    rule_name: str
    status: AlertStatus
    conditions_result: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    
    @property
    def example_id(self) -> Optional[str]:
        """Get example_id from metadata for backward compatibility"""
        return self.metadata.get("example_id")
        
    @property
    def timestamp(self) -> Optional[str]:
        """Get timestamp from metadata for backward compatibility"""
        return self.metadata.get("timestamp")
    
    @property
    def conditions_results(self) -> List[Dict[str, Any]]:
        """Backwards compatibility property for the conditions_result field"""
        return self.conditions_result

class AlertResultsClient:
    """
    Client for logging alerts to the Judgment server.
    
    This class is a placeholder for compatibility. The actual implementation
    is in the Judgment server.
    """
    
    @staticmethod
    def log_alerts(all_alerts: List[AlertResult], judgment_api_key: str) -> bool:
        """
        Log alerts to the Judgment server.
        
        Args:
            all_alerts: List of alert results to log
            judgment_api_key: The Judgment API key for authentication
            
        Returns:
            bool: True if successful, False otherwise
        """
        # This is just a placeholder - the actual implementation 
        # is handled in run_evaluation.py
        return True
