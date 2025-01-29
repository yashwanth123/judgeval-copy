"""
base e2e tests for all default judgeval scorers
"""


from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import (AnswerRelevancyScorer, 
                              ContextualPrecisionScorer,
                              ContextualRecallScorer,
                              ContextualRelevancyScorer,
                              FaithfulnessScorer,
                              HallucinationScorer,
                              SummarizationScorer,)

from judgeval.data import Example


def test_ar_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris."
    )

    example_2 = Example(  # should fail
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

    assert res[0].success == True  # example_1 should pass
    assert res[1].success == False  # example_2 should fail


def test_cp_scorer():

    example_1 = Example(  # should pass
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

    assert res[0].success == True  # example_1 should pass
    assert res[1].success == False  # example_2 should fail


def test_cr_scorer():

    example_1 = Example(  # should pass
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

    assert res[0].success == True  # example_1 should pass


def test_crelevancy_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = ContextualRelevancyScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-crelevancy"
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

    assert res[0].success == True  # example_1 should pass


def test_faithfulness_scorer():

    faithful_example = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    contradictory_example = Example(  # should fail
        input="What's the capital of France?",
        actual_output="The capital of France is Lyon. It's located in southern France near the Mediterranean coast.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        retrieval_context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = FaithfulnessScorer(threshold=0.8)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-faithfulness"
    res = client.run_evaluation(
        examples=[faithful_example, contradictory_example],
        scorers=[scorer],
        model="QWEN",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )
    assert res[0].success == True  # faithful_example should pass
    assert res[1].success == False, res[1]  # contradictory_example should fail


def test_hallucination_scorer():

    example_1 = Example(  # should pass
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
        context=[
            "Paris is a city in central France. It is the capital of France.",
            "Paris is well known for its museums, architecture, and cuisine.",
            "Flights to Paris are available from San Francisco starting at $1000."
        ]
    )

    scorer = HallucinationScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-hallucination"
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
    
    assert res[0].success == True  # example_1 should pass


def test_summarization_scorer():

    example_1 = Example(  # should pass
        input="Paris is the capital city of France and one of the most populous cities in Europe. The city is known for its iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also a global center for art, fashion, gastronomy and culture. The city's romantic atmosphere, historic architecture, and world-class museums attract millions of visitors each year.",
        actual_output="Paris is France's capital and a major European city famous for landmarks like the Eiffel Tower. It's a global hub for art, fashion and culture that draws many tourists.",
    )

    scorer = SummarizationScorer(threshold=0.5)

    client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-summarization"
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
    
    assert res[0].success == True  # example_1 should pass

if __name__ == "__main__":
    test_ar_scorer()
    test_cp_scorer()
    test_cr_scorer()
    test_crelevancy_scorer()
    test_faithfulness_scorer()
    test_hallucination_scorer()
    test_summarization_scorer()