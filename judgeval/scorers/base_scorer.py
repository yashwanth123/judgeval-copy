"""
Judgment Scorer class.

Scores `Example`s using ready-made Judgment evaluators.
"""

from pydantic import BaseModel
from enum import Enum

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
