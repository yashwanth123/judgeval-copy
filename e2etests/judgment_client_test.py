"""
Sanity checks for judgment client functionality
"""

import os
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    HallucinationScorer,
)
from judgeval.judges import TogetherJudge, judgevalJudge
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.data.datasets.dataset import EvalDataset
from dotenv import load_dotenv
import random
import string

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

    scorer = FaithfulnessScorer(threshold=0.5)
    scorer2 = HallucinationScorer(threshold=0.5)
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
    print(f"Evaluation results for {EVAL_RUN_NAME} from database:", results)


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
    )
    print(res)


def test_custom_judge(client: JudgmentClient):
    
    import vertexai
    from vertexai.generative_models import GenerativeModel

    PROJECT_ID = "judgment-labs"
    vertexai.init(project=PROJECT_ID, location="us-west1")
    
    class VertexAIJudge(judgevalJudge):

        def __init__(self, model_name: str = "gemini-1.5-flash-002"):
            self.model_name = model_name
            self.model = GenerativeModel(self.model_name)

        def load_model(self):
            return self.model

        def generate(self, prompt) -> str:
            # prompt is a List[dict] (conversation history)
            # For models that don't support conversation history, we need to convert to string
            # If you're using a model that supports chat history, you can just pass the prompt directly
            response = self.model.generate_content(str(prompt))
            return response.text
        
        async def a_generate(self, prompt) -> str:
            # prompt is a List[dict] (conversation history)
            # For models that don't support conversation history, we need to convert to string
            # If you're using a model that supports chat history, you can just pass the prompt directly
            response = await self.model.generate_content_async(str(prompt))
            return response.text
        
        def get_model_name(self) -> str:
            return self.model_name
        
    example = Example(
        input="What is the largest animal in the world?",
        actual_output="The blue whale is the largest known animal.",
        retrieval_context=["The blue whale is the largest known animal."],
    )

    judge = VertexAIJudge()

    res = client.run_evaluation(
        examples=[example],
        scorers=[CustomFaithfulnessMetric()],
        model=judge,
    )
    print(res)


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

    print("All tests passed successfully")
