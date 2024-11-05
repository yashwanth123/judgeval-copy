import requests
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

from judgeval.data.example import Example
from judgeval.scorers.custom_scorer import CustomScorer
from judgeval.scorers.base_scorer import JudgmentScorer

class EvaluationRun(BaseModel):
    """Stores example and evaluation together for running"""
    example: Example
    test_evaluation: Union[JudgmentScorer, CustomScorer]


def runner(evaluation_run: EvaluationRun):
    test_case = evaluation_run.test_case
    test_evaluation = evaluation_run.test_evaluation
    
    PROPRIETARY_TESTS = ["test1", "test2", "test3"]
    
    if test_evaluation.test_type in PROPRIETARY_TESTS:
        
        response = requests.get(
            f"https://api.judgmentlabs.ai/evaluate/{test_evaluation.test_type}/",
            json=evaluation_run.model_dump()
        )
        return response.json()
        
    elif isinstance(test_evaluation, CustomScorer):
        result = test_evaluation.score(test_case.input, test_case.output)
        return {"result": result}
    else:
        raise ValueError("Invalid test evaluation type")