"""
Custom Scorer class
"""
from typing import Generic, TypeVar, Optional, Any, Dict, Union, List
from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum

from judgeval.data.example import Example


class CustomScorer():
    """
    If you want to create a scorer that does not fall under any of the ready-made Judgment scorers,
    you can create a custom scorer by extending this class. This is best used for special use cases
    where none of Judgment's scorers are suitable.
    """
    score_type: str
    threshold: float  # The threshold to pass a test while using this metric as a scorer
    score: Optional[float] = None  # The float score of the metric run on the test case
    score_breakdown: Dict = None
    reason: Optional[str] = None  # The reason for the score when evaluating the test case
    success: Optional[bool] = None  # Whether the test case passed or failed
    evaluation_model: Optional[str] = None  # The model used to evaluate the test case
    strict_mode: bool = False  # Whether to run the metric in strict mode
    async_mode: bool = True  # Whether to run the metric in async mode
    verbose_mode: bool = True  # Whether to run the metric in verbose mode
    include_reason: bool = False  # Whether to include the reason in the output
    error: Optional[str] = None  # The error message if the metric failed
    evaluation_cost: Optional[float] = None  # The cost of running the metric
    verbose_logs: Optional[str] = None  # The verbose logs of the metric
    additional_metadata: Optional[Dict] = None  # Additional metadata for the metric

    def __init__(self, score_type: str, threshold: float):
        super().__init__()
        self.score_type = score_type
        self.threshold = threshold

    @abstractmethod
    def score_example(self, example: Example, *args, **kwargs) -> float:
        """Method that must be implemented to measure test results"""
        raise NotImplementedError("You must implement the `score` method in your custom scorer")

    @abstractmethod
    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        raise NotImplementedError("You must implement the `a_score` method in your custom scorer") 
    
    @abstractmethod
    def success_check(self) -> bool:
        raise NotImplementedError("You must implement the `passes` method in your custom scorer")

    def __str__(self):
        attributes = {
            "score_type": self.score_type,
            "threshold": self.threshold,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "reason": self.reason,
            "success": self.success,
            "evaluation_model": self.evaluation_model,
            "strict_mode": self.strict_mode,
            "async_mode": self.async_mode,
            "verbose_mode": self.verbose_mode,
            "include_reason": self.include_reason,
            "error": self.error,
            "evaluation_cost": self.evaluation_cost,
            "verbose_logs": self.verbose_logs,
            "additional_metadata": self.additional_metadata,
        }
        return f"CustomScorer({attributes})"

