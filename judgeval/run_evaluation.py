"""
Infra to execute evaluation runs either locally or via Judgment API
"""

import requests
import pprint
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

from judgeval.data.example import Example
from judgeval.scorers.custom_scorer import CustomScorer
from judgeval.constants import *
from judgeval.scorers.base_scorer import JudgmentScorer

class EvaluationRun(BaseModel):
    """
    Stores example and evaluation together for running
    
    Args: 
        examples (List[Example]): The examples to evaluate
        scorers (List[Union[JudgmentScorer, CustomScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
    """
    examples: List[Example]
    scorers: List[Union[JudgmentScorer, CustomScorer]]
    model: Union[str, List[str]]
    aggregator: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # TODO add parsing to make sure that all fields are valid


def execute_api_eval(evaluation_run: EvaluationRun):
    """
    Executes an evaluation of an `Example` using a `Scorer` via the Judgment API

    TODO add error handling for failed responses
    """
    # submit API request to execute test
    response = requests.post(JUDGMENT_EVAL_API_URL, json=evaluation_run.model_dump())
    assert response.status_code == 200, f"Failed to execute evaluation run."

    response_data = response.json()
    return response_data


def run_eval(evaluation_run: EvaluationRun):
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s
    """
    
    judgment_scorers = []
    custom_scorers = []
    for scorer in evaluation_run.scorers:
        if isinstance(scorer, JudgmentScorer):
            judgment_scorers.append(scorer)
        else:
            custom_scorers.append(scorer)
    
    # Execute evaluation using Judgment API
    if judgment_scorers:
        response_data = execute_api_eval(evaluation_run)
        pprint.pprint(response_data)
    
    # Run local tests
    if custom_scorers:  # TODO
        raise NotImplementedError


if __name__ == "__main__":
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

    eval_data = EvaluationRun(
        examples=[example1, example2],
        scorers=[scorer],
        metadata={"batch": "test"},
        model=["QWEN", "MISTRAL_8x7B_INSTRUCT"],
        aggregator='QWEN'
    )

    run_eval(eval_data)

