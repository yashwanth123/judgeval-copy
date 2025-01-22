"""
base e2e tests for all default judgeval scorers
"""


from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import AnswerRelevancyScorer
from judgeval.data import Example


def test_ar_scorer():

    example_1 = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris."
    )

    example_2 = Example(
        input="What's the capital of France?",
        actual_output="There's alot to do in Marseille. Lots of bars, restaurants, and museums."
    )

    scorer = AnswerRelevancyScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run"
    res = client.run_evaluation(
        examples=[example_1, example_2],
        scorers=[scorer],
        model="QWEN",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )

    print(res)



if __name__ == "__main__":
    test_ar_scorer()
