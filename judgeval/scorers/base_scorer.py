"""
Judgment Scorer class.

Scores `Example`s using ready-made Judgment evaluators.
"""

from pydantic import BaseModel, field_validator

from judgeval.constants import JudgmentMetric


class JudgmentScorer(BaseModel):
    """
    Class for ready-made, "out-of-the-box" scorer that uses Judgment evaluators to score `Example`s.

    Args:
        score_type (JudgmentMetric): The Judgment metric to use for scoring `Example`s
    """
    threshold: float
    score_type: JudgmentMetric

    @field_validator('score_type')
    def convert_to_enum_value(cls, v):
        """
        Validates that the `score_type` is a valid `JudgmentMetric` enum value.
        Converts string values to `JudgmentMetric` enum values.
        """
        if isinstance(v, JudgmentMetric):
            return v.value
        elif isinstance(v, str):
            return JudgmentMetric[v.upper()].value
        raise ValueError(f"Invalid value for score_type: {v}")
    
    def __str__(self):
        return f"JudgmentScorer(score_type={self.score_type}, threshold={self.threshold})"
    