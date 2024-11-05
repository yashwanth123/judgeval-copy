"""
Infra to execute evaluation runs either locally or via Judgment API
"""

import requests
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

from judgeval.data.example import Example
from judgeval.scorers.custom_scorer import CustomScorer
from judgeval.constants import JUDGMENT_EVAL_API_URL
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
    model: str 
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
    Executes an evaluation of an `Example` using a `Scorer`
    """
    
    if isinstance(evaluation_run.scorer, JudgmentScorer):  # Use Judgment API to evaluate
        return execute_api_eval(evaluation_run)
    elif isinstance(evaluation_run.scorer, CustomScorer):  # Use custom scorer to evaluate
        # run test locally
        pass
    else:
        raise ValueError(f"Scorer type {evaluation_run.scorer} not recognized. Please use a valid scorer type, such as JudgmentScorer or CustomScorer.")
    