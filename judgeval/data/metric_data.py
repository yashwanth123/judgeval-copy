"""
Implementation of the MetricData class.
"""

from typing import List, Union, Optional, Dict
from pydantic import BaseModel, Field

from judgeval.scorers.custom_scorer import CustomScorer

class MetricData(BaseModel):
    name: str
    threshold: float
    success: bool
    score: Optional[float] = None
    reason: Optional[str] = None
    strict_mode: Optional[bool] = Field(False, alias="strictMode")
    evaluation_model: Union[List[str], str] = Field(None, alias="evaluationModel")
    error: Optional[str] = None
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    verbose_logs: Optional[str] = Field(None, alias="verboseLogs")
    additional_metadata: Optional[Dict] = Field(None, alias="additionalMetadata")


def create_metric_data(metric: CustomScorer) -> MetricData:
    """
    After a `metric` is run, it contains information about the test case that was evaluated
    using this metric. For example, after computing Faithfulness, the `metric` object will contain
    whether the test case was passed, the score, the reason for score, etc.

    This function takes an executed `metric` object and produces a MetricData object that
    contains the output of the metric evaluation run that can be exported to be logged as a part of
    the TestResult.
    """
    if metric.error is not None:  # error occurred during eval run
        return MetricData(
            name=metric.__name__,
            threshold=metric.threshold,
            score=None,
            reason=None,
            success=False,
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=metric.error,
            evaluationCost=metric.evaluation_cost,
            verboseLogs=metric.verbose_logs,
        )
    else:  # standard execution, no error
        return MetricData(
            name=metric.__name__,
            score=metric.score,
            threshold=metric.threshold,
            reason=metric.reason,
            success=metric.success_check(),
            strictMode=metric.strict_mode,
            evaluationModel=metric.evaluation_model,
            error=None,
            evaluationCost=metric.evaluation_cost,
            verboseLogs=metric.verbose_logs,
            additionalMetadata=metric.additional_metadata,
        )
