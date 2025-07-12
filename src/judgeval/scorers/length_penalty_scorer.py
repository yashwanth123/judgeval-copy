from judgeval.scorers.judgeval_scorer import JudgevalScorer
from judgeval.data import Example, ScoringResult, ScorerData

class LengthPenaltyScorer(JudgevalScorer):
    def __init__(self, threshold: float = 0.9):
        super().__init__(score_type="length_penalty", threshold=threshold)

    def score_example(self, example: Example) -> ScoringResult:
        score = 1.0 if len(example.actual_output) <= 1000 else 0.0
        reason = "Within limit." if score == 1.0 else "Too long."

        scorer_data = ScorerData(
            score=score,
            reason=reason,
            name=self.score_type,
            threshold=self.threshold,
            success=score >= self.threshold
        )

        return ScoringResult(success=scorer_data.success, scorers_data=[scorer_data], data_object=example)

    async def a_score_example(self, example: Example):
        return self.score_example(example)

    def _success_check(self) -> bool:
        return self.score >= self.threshold
