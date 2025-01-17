"""
`judgeval` contextual relevancy scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer


class ContextualRelevancyScorer(APIJudgmentScorer):
    """
    Scorer that checks if the output of a model is relevant to the retrieval context
    """
    def __init__(self, threshold: float):
        super().__init__(threshold=threshold, score_type=APIScorer.CONTEXTUAL_RELEVANCY)

    @property
    def __name__(self):
        return "Contextual Relevancy"
