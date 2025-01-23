"""
`judgeval` tool correctness scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.base_scorer import JudgmentScorer
from judgeval.constants import APIScorer


class ToolCorrectnessScorer(JudgmentScorer):
    def __init__(self, threshold: float):
        super().__init__(threshold=threshold, score_type=APIScorer.TOOL_CORRECTNESS)

    @property
    def __name__(self):
        return "Tool Correctness"
