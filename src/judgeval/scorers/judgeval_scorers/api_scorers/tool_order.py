"""
`judgeval` tool order scorer
"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer

class ToolOrderScorer(APIJudgmentScorer):
    def __init__(self, threshold: float=1.0):
        super().__init__(
            threshold=threshold, 
            score_type=APIScorer.TOOL_ORDER,
        )

    @property
    def __name__(self):
        return "Tool Order"
