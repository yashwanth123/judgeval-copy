"""
Sanity checks for judgment client functionality
"""

import os
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import JudgmentScorer
from judgeval.constants import APIScorer
from judgeval.judges import TogetherJudge
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.data.datasets.dataset import EvalDataset
from dotenv import load_dotenv
import random
import string

from judgeval.scorers.prompt_scorer import ClassifierScorer

load_dotenv()

def get_client():
    return JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))

def get_ui_client():
    return JudgmentClient(judgment_api_key=os.getenv("UI_JUDGMENT_API_KEY"))

def test_dataset(client: JudgmentClient):
    dataset: EvalDataset = client.create_dataset()
    dataset.add_example(Example(input="input 1", actual_output="output 1"))

    client.push_dataset(alias="test_dataset_5", dataset=dataset, overwrite=False)
    
    # PULL
    dataset = client.pull_dataset(alias="test_dataset_5")
    print(dataset)

def test_run_eval(client: JudgmentClient):

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
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

    scorer = JudgmentScorer(threshold=0.5, score_type=APIScorer.FAITHFULNESS)
    scorer2 = JudgmentScorer(threshold=0.5, score_type=APIScorer.HALLUCINATION)
    c_scorer = CustomFaithfulnessMetric(threshold=0.6)

    PROJECT_NAME = "test_project_JOSEPH"
    EVAL_RUN_NAME = "yomadude"
    
    _ = client.run_evaluation(
        examples=[example1, example2],
        scorers=[scorer, c_scorer],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=True,
        override=True,
    )

    results = client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
    # print(f"Evaluation results for {EVAL_RUN_NAME} from database:", results)

def test_override_eval(client: JudgmentClient):
    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
    )
    
    scorer = JudgmentScorer(threshold=0.5, score_type=APIScorer.FAITHFULNESS)

    PROJECT_NAME = "test_eval_run_naming_collisions"
    EVAL_RUN_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    # First run should succeed
    client.run_evaluation(
        examples=[example1],
        scorers=[scorer],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=True,
        override=False,
    )
    
    # Second run with log_results=False should succeed
    client.run_evaluation(
        examples=[example1],
        scorers=[scorer],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=False,
        override=False,
    )
    
    # Third run with override=True should succeed
    try:
        client.run_evaluation(
            examples=[example1],
            scorers=[scorer],
            model="QWEN",
            metadata={"batch": "test"},
            project_name=PROJECT_NAME,
            eval_run_name=EVAL_RUN_NAME,
            log_results=True,
            override=True,
        )
    except ValueError as e:
        print(f"Unexpected error in override run: {e}")
        raise
    
    # Final non-override run should fail
    try:
        client.run_evaluation(
            examples=[example1],
            scorers=[scorer],
            model="QWEN",
            metadata={"batch": "test"},
            project_name=PROJECT_NAME,
            eval_run_name=EVAL_RUN_NAME,
            log_results=True,
            override=False,
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        if "already exists" not in str(e):
            raise
        print(f"Successfully caught expected error: {e}")
    
    

def test_evaluate_dataset(client: JudgmentClient):

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
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
        scorers=[JudgmentScorer(threshold=0.5, score_type=APIScorer.FAITHFULNESS)],
        model="QWEN",
        metadata={"batch": "test"},
    )

    print(res)
    
def test_classifier_scorer(client: JudgmentClient):
    # Modifying a classifier scorer
    # TODO: Some of the field names are not consistent between regular scorers and classifier scorers
    # Make some methods private
    classifier_scorer = client.fetch_classifier_scorer("tonescorer-72gl")
    print(f"{classifier_scorer=}")
    
    # TODO: Does ClassifierScorer actually use build_measure_prompt, enforce_prompt_format, etc.
    # TODO: Ik PromptScorer uses it, but I don't think we need to redefine it in ClassifierScorer
    
    # Creating a classifier scorer from SDK
    classifier_scorer_custom = ClassifierScorer(
        name="Test Classifier Scorer",
        threshold=0.5,
        conversation=[],
        options={}
    )
    
    classifier_scorer_custom.update_conversation(conversation=[{"role": "user", "content": "What is the capital of France?"}])
    classifier_scorer_custom.update_options(options={"yes": 1, "no": 0})
    
    slug = client.push_classifier_scorer(scorer=classifier_scorer_custom)
    
    classifier_scorer_custom = client.fetch_classifier_scorer(slug=slug)
    print(f"{classifier_scorer_custom=}")
    
    # faithfulness_scorer = JudgmentScorer(threshold=0.5, score_type=APIScorer.FAITHFULNESS)
    
    # example1 = Example(
    #     input="What if these shoes don't fit?",
    #     actual_output="We offer a 30-day full refund at no extra cost, you would have known that if you read the website stupid!",
    #     retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    # )
    
    # res = client.run_evaluation(
    #     examples=[example1],
    #     scorers=[classifier_scorer, faithfulness_scorer],
    #     model="QWEN",
    # )
    # print(res)
    
    # Pushing a classifier scorer (from SDK)

if __name__ == "__main__":
    # Test client functionality
    client = get_client()
    ui_client = get_ui_client()
    print("Client initialized successfully")
    print("*" * 40)

    # print("Testing dataset creation, pushing, and pulling")
    # test_dataset(ui_client)
    # print("Dataset creation, pushing, and pulling successful")
    # print("*" * 40)
    
    # print("Testing evaluation run")
    # test_run_eval(ui_client)
    # print("Evaluation run successful")
    # print("*" * 40)
    
    print("Testing evaluation run override")
    test_override_eval(client)
    print("Evaluation run override successful")
    print("*" * 40)
    
    print("Testing evaluation run override")
    test_override_eval(client)
    print("Evaluation run override successful")
    print("*" * 40)
    
    print("Testing dataset evaluation")
    test_evaluate_dataset(ui_client)
    print("Dataset evaluation successful")
    print("*" * 40)
    # print("*" * 40)
    
    print("Testing classifier scorer")
    test_classifier_scorer(ui_client)
    print("Classifier scorer test successful")
    print("*" * 40)

    print("All tests passed successfully")
