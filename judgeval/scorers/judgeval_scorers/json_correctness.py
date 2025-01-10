"""
`judgeval` JSON correctness scorer

TODO add link to docs page for this scorer

"""


# External imports
from pydantic import BaseModel
# Internal imports
from judgeval.scorers.base_scorer import JudgmentScorer
from judgeval.constants import APIScorer


class JSONCorrectnessScorer(JudgmentScorer):
    def __init__(self, threshold: float, schema: BaseModel):
        super().__init__(threshold=threshold, score_type=APIScorer.JSON_CORRECTNESS)
        self.json_schema = schema

    @property
    def __name__(self):
        return "JSON Correctness"
