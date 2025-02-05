"""
`judgeval` answer relevancy scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer


class AnswerCorrectnessScorer(APIJudgmentScorer):
    def __init__(self, threshold: float):
        super().__init__(threshold=threshold, score_type=APIScorer.ANSWER_CORRECTNESS)

    @property
    def __name__(self):
        return "Answer Correctness"
