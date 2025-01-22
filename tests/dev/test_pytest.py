from judgeval.evaluation_run import EvaluationRun
from judgeval.data import Example
from judgeval.run_evaluation import assert_test
from judgeval.scorers import (
    FaithfulnessScorer,
    HallucinationScorer,
    JSONCorrectnessScorer
)
from judgeval.judges.together_judge import TogetherJudge

def test_assert_test():

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )

    scorer = FaithfulnessScorer(threshold=0.5)

    eval_run = EvaluationRun(
        eval_name="test_eval",
        examples=[example1],
        scorers=[scorer],
        model="QWEN"
    )

    assert_test(eval_run)
    
