"""
Judgment Scorer class.

Scores `Example`s using ready-made Judgment evaluators.
"""

from pydantic import BaseModel, field_validator
from judgeval.common.logger import debug, info, warning, error

from judgeval.constants import APIScorer, UNBOUNDED_SCORERS


class APIJudgmentScorer(BaseModel):
    """
    Class for ready-made, "out-of-the-box" scorer that uses Judgment evaluators to score `Example`s.

    Args:
        score_type (APIScorer): The Judgment metric to use for scoring `Example`s
        threshold (float): A value between 0 and 1 that determines the scoring threshold
    """
    score_type: APIScorer
    threshold: float

    @field_validator('threshold')
    def validate_threshold(cls, v, info):
        """
        Validates that the threshold is between 0 and 1 inclusive.
        """
        score_type = info.data.get('score_type')
        if score_type in UNBOUNDED_SCORERS:
            if v < 0:
                error(f"Threshold for {score_type} must be greater than 0, got: {v}")
                raise ValueError(f"Threshold for {score_type} must be greater than 0, got: {v}")
        else:
            if not 0 <= v <= 1:
                error(f"Threshold for {score_type} must be between 0 and 1, got: {v}")
                raise ValueError(f"Threshold for {score_type} must be between 0 and 1, got: {v}")
        return v

    @field_validator('score_type')
    def convert_to_enum_value(cls, v):
        """
        Validates that the `score_type` is a valid `JudgmentMetric` enum value.
        Converts string values to `JudgmentMetric` enum values.
        """
        debug(f"Attempting to convert score_type value: {v}")
        if isinstance(v, APIScorer):
            info(f"Using existing JudgmentMetric: {v.value}")
            return v.value
        elif isinstance(v, str):
            debug(f"Converting string value to JudgmentMetric enum: {v}")
            return APIScorer[v.upper()].value
        error(f"Invalid score_type value: {v}")
        raise ValueError(f"Invalid value for score_type: {v}")
    
    def __str__(self):
        return f"JudgmentScorer(score_type={self.score_type.value}, threshold={self.threshold})"
    