"""
Sanity checks for judgment client functionality
"""

import os
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import JudgmentScorer
from judgeval.constants import JudgmentMetric
from judgeval.judges import TogetherJudge
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.data.datasets.dataset import EvalDataset


def get_client():
    return JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))


def test_dataset():
    print(f"Judgment API Key: {os.getenv('JUDGMENT_API_KEY')}")
    client = get_client()
    dataset: EvalDataset = client.create_dataset()
    dataset.add_example(Example(input="input 1", actual_output="output 1"))

    client.push_dataset(alias="test_dataset_5", dataset=dataset, overwrite=False)
    
    # PULL
    dataset = client.pull_dataset(alias="test_dataset_5")
    print(dataset)
    

def test_run_eval():
    client = get_client()

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )

    example2 = Example(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        name="Password Reset",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"}
    )

    scorer = JudgmentScorer(threshold=0.5, score_type=JudgmentMetric.FAITHFULNESS)
    scorer2 = JudgmentScorer(threshold=0.5, score_type=JudgmentMetric.HALLUCINATION)
    model = TogetherJudge()
    c_scorer = CustomFaithfulnessMetric(
        threshold=0.6,
        model=model,
    )

    results = client.run_evaluation(
        examples=[example1, example2],
        scorers=[scorer, c_scorer],
        model="QWEN",
        metadata={"batch": "test"}
    )

    print(results)


def test_evaluate_dataset():
    client = get_client()

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )

    example2 = Example(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        name="Password Reset",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"}
    )

    dataset = EvalDataset(examples=[example1, example2])
    res = client.evaluate_dataset(
        dataset=dataset,
        scorers=[JudgmentScorer(threshold=0.5, score_type=JudgmentMetric.FAITHFULNESS)],
        model="QWEN",
        metadata={"batch": "test"},
    )

    print(res)

if __name__ == "__main__":
    # Test client functionality
    get_client()
    print("Client initialized successfully")

    print("Testing dataset creation, pushing, and pulling")
    test_dataset()
    print("Dataset creation, pushing, and pulling successful")
    print("Testing evaluation run")
    test_run_eval()
    print("Evaluation run successful")

    print("Testing dataset evaluation")
    test_evaluate_dataset()
    print("Dataset evaluation successful")

    print("All tests passed successfully")
