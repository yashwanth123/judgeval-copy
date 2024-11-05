"""
Judgment Scorer class.

Scores `Example`s using ready-made Judgment evaluators.
"""

from pydantic import BaseModel, field_validator

from judgeval.constants import JudgmentMetric


class JudgmentScorer(BaseModel):
    """
    Class for scorer that uses Judgment evaluators. If you would like to use one of our 
    ready-made scorers, you can use this class to score an Example.

    Args:
        score_type (JudgmentMetric): The Judgment metric to use for scoring `Example`s
    """
    threshold: float
    score_type: JudgmentMetric

    @field_validator('score_type')
    def convert_to_enum_value(cls, v):
        if isinstance(v, JudgmentMetric):
            return v.value
        elif isinstance(v, str):
            return JudgmentMetric[v.upper()].value
        raise ValueError(f"Invalid value for score_type: {v}")
    