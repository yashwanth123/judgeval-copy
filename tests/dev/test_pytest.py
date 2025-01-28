from judgeval.data import Example
from judgeval.run_evaluation import assert_test
from judgeval.scorers import (
    FaithfulnessScorer,
    AnswerRelevancyScorer
)
from judgeval.judgment_client import JudgmentClient
import os

def test_assert_test():

    client = JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))

    example = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )

    example1 = Example(
        input="How much are your croissants?",
        actual_output="Sorry, we don't accept electronic returns.",
    )

    example2 = Example(
        input="Who is the best basketball player in the world?",
        actual_output="No, the room is too small.",
    )

    scorer = FaithfulnessScorer(threshold=0.5)
    scorer1 = AnswerRelevancyScorer(threshold=0.5)

    results = client.run_evaluation(
        eval_run_name="test_eval",
        examples=[example, example1, example2],
        scorers=[scorer, scorer1],
        model="QWEN",
    )

    assert_test(results)
    
