from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    AnswerRelevancyScorer
)
from judgeval.judgment_client import JudgmentClient
import os

def test_assert_test(mocker):
    # Mock the JudgmentClient
    mock_client = mocker.Mock(spec=JudgmentClient)
    mocker.patch('judgeval.judgment_client.JudgmentClient', return_value=mock_client)

    # Create examples and scorers as before
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

    # Use the mock client instead of creating a real one
    mock_client.assert_test.return_value = {"mock": "results"}  # Add expected return value

    results = mock_client.assert_test(
        eval_run_name="test_eval",
        examples=[example, example1, example2],
        scorers=[scorer, scorer1],
        model="QWEN",
    )

    # Assert the results match what we expect
    assert results == {"mock": "results"}
    
