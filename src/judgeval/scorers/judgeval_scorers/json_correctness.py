"""
`judgeval` JSON correctness scorer

TODO add link to docs page for this scorer

"""


# External imports
from pydantic import BaseModel, Field
# Internal imports
from judgeval.scorers.base_scorer import JudgmentScorer
from judgeval.constants import APIScorer


class JSONCorrectnessScorer(JudgmentScorer):
    json_schema: BaseModel = Field(None, exclude=True)
    
    def __init__(self, threshold: float, json_schema: BaseModel):
        super().__init__(threshold=threshold, score_type=APIScorer.JSON_CORRECTNESS)
        object.__setattr__(self, 'json_schema', json_schema)

    def to_dict(self):
        return {
            "score_type": self.score_type,
            "threshold": self.threshold,
            "kwargs": {"json_schema": self.json_schema.model_json_schema()}
        }

    @property
    def __name__(self):
        return "JSON Correctness"
