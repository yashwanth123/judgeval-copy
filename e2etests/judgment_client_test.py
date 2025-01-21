"""
Sanity checks for judgment client functionality
"""

import os
from pydantic import BaseModel

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    HallucinationScorer,
    JSONCorrectnessScorer
)
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
    # Single step in our workflow, an outreach Sales Agent

    example1 = Example(
        input="Generate a cold outreach email for TechCorp. Facts: They recently launched an AI-powered analytics platform. Their CEO Sarah Chen previously worked at Google. They have 50+ enterprise clients.",
        actual_output="Dear Ms. Chen,\n\nI noticed TechCorp's recent launch of your AI analytics platform and was impressed by its enterprise-focused approach. Your experience from Google clearly shines through in building scalable solutions, as evidenced by your impressive 50+ enterprise client base.\n\nWould you be open to a brief call to discuss how we could potentially collaborate?\n\nBest regards,\nAlex",
        retrieval_context=["TechCorp launched AI analytics platform in 2024", "Sarah Chen is CEO, ex-Google executive", "Current client base: 50+ enterprise customers"],
    )

    example2 = Example(
        input="Generate a cold outreach email for GreenEnergy Solutions. Facts: They're developing solar panel technology that's 30% more efficient. They're looking to expand into the European market. They won a sustainability award in 2023.",
        actual_output="Dear GreenEnergy Solutions team,\n\nCongratulations on your 2023 sustainability award! Your innovative solar panel technology with 30% higher efficiency is exactly what the European market needs right now.\n\nI'd love to discuss how we could support your European expansion plans.\n\nBest regards,\nAlex",
        expected_output="A professional cold email mentioning the sustainability award, solar technology innovation, and European expansion plans",
        context=["Business Development"],
        retrieval_context=["GreenEnergy Solutions won 2023 sustainability award", "New solar technology 30% more efficient", "Planning European market expansion"],
    )

    scorer = FaithfulnessScorer(threshold=0.5)
    scorer2 = HallucinationScorer(threshold=0.5)
    c_scorer = CustomFaithfulnessMetric(threshold=0.6)

    PROJECT_NAME = "OutreachWorkflow"
    EVAL_RUN_NAME = "ColdEmailGenerator-Improve-BasePrompt"
    
    client.run_evaluation(
        examples=[example1, example2],
        scorers=[scorer, scorer2],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=True,
        override=True,
    )

    results = client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
    print(f"Evaluation results for {EVAL_RUN_NAME} from database:", results)


def test_json_scorer(client: JudgmentClient):

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output='{"tool": "authentication"}',
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

    class SampleSchema(BaseModel):
        tool: str

    scorer = JSONCorrectnessScorer(threshold=0.5, json_schema=SampleSchema)
    PROJECT_NAME = "test_project_JOSEPH"
    EVAL_RUN_NAME = "yomadude"
    
    res = client.run_evaluation(
        examples=[example1, example2],
        scorers=[scorer],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=True,
        override=True,
    )

    print(res)


def test_override_eval(client: JudgmentClient):
    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
    )
    
    scorer = FaithfulnessScorer(threshold=0.5)

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
        scorers=[FaithfulnessScorer(threshold=0.5)],
        model="QWEN",
        metadata={"batch": "test"},
    )

    print(res)
    
def test_classifier_scorer(client: JudgmentClient):
    classifier_scorer = client.fetch_classifier_scorer("tonescorer-72gl")
    faithfulness_scorer = FaithfulnessScorer(threshold=0.5)
    
    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost, you would have known that if you read the website stupid!",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )
    
    res = client.run_evaluation(
        examples=[example1],
        scorers=[classifier_scorer, faithfulness_scorer],
        model="QWEN",
        log_results=True,
        eval_run_name="ToneScorerTest",
        project_name="ToneScorerTest",
    )

if __name__ == "__main__":
    # Test client functionality
    client = get_client()
    ui_client = get_ui_client()
    print("Client initialized successfully")
    print("*" * 40)

    print("Testing dataset creation, pushing, and pulling")
    test_dataset(ui_client)
    print("Dataset creation, pushing, and pulling successful")
    print("*" * 40)
    
    print("Testing evaluation run")
    test_run_eval(ui_client)
    print("Evaluation run successful")
    print("*" * 40)

<<<<<<< HEAD
=======
    print("Testing JSON scorer")
    test_json_scorer(ui_client)
    print("JSON scorer test successful")
    print("*" * 40)
    
    print("Testing evaluation run override")
    test_override_eval(client)
    print("Evaluation run override successful")
    print("*" * 40)
    
    print("Testing dataset evaluation")
    test_evaluate_dataset(ui_client)
    print("Dataset evaluation successful")
    print("*" * 40)
    
    print("Testing classifier scorer")
    test_classifier_scorer(ui_client)
    print("Classifier scorer test successful")
    print("*" * 40)

>>>>>>> add_json_correctness_scorer
    print("All tests passed successfully")
