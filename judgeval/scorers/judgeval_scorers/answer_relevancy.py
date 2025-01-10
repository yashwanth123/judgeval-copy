"""
`judgeval` answer relevancy scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.base_scorer import JudgmentScorer
from judgeval.constants import APIScorer


class AnswerRelevancyScorer(JudgmentScorer):
    """
    Scorer that checks if the output of a model is relevant to the question
    """
    def __init__(self, threshold: float):
        super().__init__(threshold=threshold, score_type=APIScorer.ANSWER_RELEVANCY)

    @property
    def __name__(self):
        return "Answer Relevancy"
