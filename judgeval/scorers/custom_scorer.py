"""
Custom Scorer class

Enables client to create custom scorers that do not fall under any of the ready-made Judgment scorers.
To create a custom scorer, extend this class and implement the `score_example`, `a_score_example`, and `success_check` methods.
"""

from typing import Optional, Dict
from abc import abstractmethod

from judgeval.data import Example


class CustomScorer:
    """
    If you want to create a scorer that does not fall under any of the ready-made Judgment scorers,
    you can create a custom scorer by extending this class. This is best used for special use cases
    where none of Judgment's scorers are suitable.
    """
    score_type: str  # name of your new scorer
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

    def __init__(
        self, 
        score_type: str, 
        threshold: float, 
        score: Optional[float] = None, 
        score_breakdown: Optional[Dict] = None, 
        reason: Optional[str] = None, 
        success: Optional[bool] = None, 
        evaluation_model: Optional[str] = None, 
        strict_mode: bool = False, 
        async_mode: bool = True, 
        verbose_mode: bool = True, 
        include_reason: bool = False, 
        error: Optional[str] = None, 
        evaluation_cost: Optional[float] = None, 
        verbose_logs: Optional[str] = None, 
        additional_metadata: Optional[Dict] = None
        ):
            self.score_type = score_type
            self.threshold = threshold
            self.score = score
            self.score_breakdown = score_breakdown
            self.reason = reason
            self.success = success
            self.evaluation_model = evaluation_model
            self.strict_mode = strict_mode
            self.async_mode = async_mode
            self.verbose_mode = verbose_mode
            self.include_reason = include_reason
            self.error = error
            self.evaluation_cost = evaluation_cost
            self.verbose_logs = verbose_logs
            self.additional_metadata = additional_metadata


    @abstractmethod
    def score_example(self, example: Example, *args, **kwargs) -> float:
        """
        Measures the score on a single example
        """
        raise NotImplementedError("You must implement the `score` method in your custom scorer")

    @abstractmethod
    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        """
        Asynchronously measures the score on a single example
        """
        raise NotImplementedError("You must implement the `a_score` method in your custom scorer") 
    
    @abstractmethod
    def success_check(self) -> bool:
        """
        For unit testing, determines whether the test case passes or fails
        """
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
