"""
base e2e tests for all default judgeval scorers
"""


from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import (AnswerRelevancyScorer, 
                              ContextualPrecisionScorer,
                              ContextualRecallScorer)

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


def test_cp_scorer():

    example_1 = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    example_2 = Example(
        input="What's the capital of France?",
        actual_output="There's alot to do in Marseille. Lots of bars, restaurants, and museums.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Marseille is a city in southern France. It is the second largest city in France.",
            "Marseille is known for its beaches, historic port, and vibrant nightlife.",
            "Flights to Marseille are available from San Francisco starting at $500."
        ]
    )

    scorer = ContextualPrecisionScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-cp"
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


def test_cr_scorer():

    example_1 = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = ContextualRecallScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-cr"
    res = client.run_evaluation(
        examples=[example_1],
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
    # test_ar_scorer()
    # test_cp_scorer()
    test_cr_scorer()
    