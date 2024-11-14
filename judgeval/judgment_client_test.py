import os
from judgeval.judgment_client import JudgmentClient
from judgeval.evaluation_run import EvaluationRun
from judgeval.data import Example
from judgeval.scorers import JudgmentScorer
from judgeval.constants import JudgmentMetric
from judgeval.judges import TogetherJudge
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.data.datasets.dataset import EvalDataset
def test_dataset():
    # Associate EvalDatasets with a JudgmentClient, so they don't have to pass in judgment_api_key
    client = JudgmentClient(judgment_api_key=os.getenv("TEST_JUDGMENT_API_KEY"))
    dataset: EvalDataset = client.create_dataset()
    dataset.add_example(Example(input="input 1", actual_output="output 1"))

    client.push_dataset(alias="test_dataset_5", dataset=dataset, overwrite=False)
    
    # PULL
    dataset = client.pull_dataset(alias="test_dataset_5")
    print(dataset)
    
def test_run_eval():
    client = JudgmentClient(judgment_api_key=os.getenv("TEST_JUDGMENT_API_KEY"))

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

    eval_data = EvaluationRun(
        examples=[example1, example2],
        scorers=[scorer, c_scorer],
        metadata={"batch": "test"},
        model=["QWEN", "MISTRAL_8x7B_INSTRUCT"],
        aggregator='QWEN'
    )

    results = client.run_eval(eval_data)

    print(results)

if __name__ == "__main__":
    test_dataset()
