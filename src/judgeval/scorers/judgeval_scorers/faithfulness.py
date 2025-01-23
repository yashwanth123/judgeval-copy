"""
`judgeval` faithfulness scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.base_scorer import JudgmentScorer
from judgeval.constants import APIScorer


class FaithfulnessScorer(JudgmentScorer):
    def __init__(self, threshold: float):
        super().__init__(threshold=threshold, score_type=APIScorer.FAITHFULNESS)

    @property
    def __name__(self):
        return "Faithfulness"
