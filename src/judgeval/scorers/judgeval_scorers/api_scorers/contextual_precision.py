"""
`judgeval` contextual precision scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer


class ContextualPrecisionScorer(APIJudgmentScorer):
    def __init__(self, threshold: float):
        super().__init__(threshold=threshold, score_type=APIScorer.CONTEXTUAL_PRECISION)

    @property
    def __name__(self):
        return "Contextual Precision"
