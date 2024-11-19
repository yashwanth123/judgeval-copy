"""
Implements the JudgmentClient to interact with the Judgment API.
"""
import os
from typing import Optional, List, Dict, Any, Union
import requests

from judgeval.constants import ROOT_API
from judgeval.data.datasets import EvalDataset
from judgeval.data import ScoringResult, Example
from judgeval.scorers import JudgmentScorer, CustomScorer
from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import run_eval
from judgeval.constants import JUDGMENT_EVAL_FETCH_API_URL
from pydantic import BaseModel

class EvalRunRequestBody(BaseModel):
    name: str
    judgment_api_key: str


class JudgmentClient:
    def __init__(self, judgment_api_key: str = os.getenv("JUDGMENT_API_KEY")):
        self.judgment_api_key = judgment_api_key
        
        # Verify API key is valid
        result, response = self._validate_api_key()
        if not result:
            # May be bad to output their invalid API key...
            raise ValueError(f"Issue with passed in Judgment API key: {response}")
        else:
            # TODO: Add logging
            print(f"Successfully initialized JudgmentClient, welcome back {response.get('detail', {}).get('user_name', 'user')}!")
            
    def run_evaluation(
        self, 
        examples: List[Example],
        scorers: List[Union[JudgmentScorer, CustomScorer]],
        model: Union[str, List[str]],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        eval_run_name: str = "",
        log_results: bool = False
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of `Example`s using one or more `Scorer`s
        """
        try:
            eval = EvaluationRun(
                name=eval_run_name,
            examples=examples,
            scorers=scorers,
            model=model,
            aggregator=aggregator,
            metadata=metadata,
            judgment_api_key=self.judgment_api_key
            )
            return run_eval(eval, name=eval_run_name, log_results=log_results)
        except ValueError as e:
            raise ValueError(f"Please check your EvaluationRun object, one or more fields are invalid: \n{str(e)}")
    
    def evaluate_dataset(
        self, 
        dataset: EvalDataset,
        scorers: List[Union[JudgmentScorer, CustomScorer]],
        model: Union[str, List[str]],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        eval_run_name: str = "",
        log_results: bool = False
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of a `EvalDataset` using one or more `Scorer`s
        """
        try:
            evaluation_run = EvaluationRun(
                name=eval_run_name,
                examples=dataset.examples,
                scorers=scorers,
                model=model,
                aggregator=aggregator,
                metadata=metadata,
            judgment_api_key=self.judgment_api_key
            )
            return run_eval(evaluation_run, name=eval_run_name, log_results=log_results)
        except ValueError as e:
            raise ValueError(f"Please check your EvaluationRun object, one or more fields are invalid: \n{str(e)}")

    def create_dataset(self) -> EvalDataset:
        return EvalDataset(judgment_api_key=self.judgment_api_key)

    def push_dataset(self, alias: str, dataset: EvalDataset, overwrite: Optional[bool] = False) -> bool:
        """
        Uploads an `EvalDataset` to the Judgment platform for storage.

        Args:
            alias (str): The name to use for the dataset
            dataset (EvalDataset): The dataset to upload to Judgment
            overwrite (Optional[bool]): Whether to overwrite the dataset if it already exists

        Returns:
            bool: Whether the dataset was successfully uploaded
        """
        # Set judgment_api_key just in case it was not set
        dataset.judgment_api_key = self.judgment_api_key
        return dataset.push(alias, overwrite)
    
    def pull_dataset(self, alias: str) -> EvalDataset:
        """
        Retrieves a saved `EvalDataset` from the Judgment platform.

        Args:
            alias (str): The name of the dataset to retrieve

        Returns:
            EvalDataset: The retrieved dataset
        """
        dataset = EvalDataset(judgment_api_key=self.judgment_api_key)
        dataset.pull(alias)
        return dataset
    
    # Maybe add option where you can pass in the EvaluationRun object and it will pull the eval results from the backend
    def pull_eval(self, eval_run_name: str) -> List[ScoringResult]:
        eval_run_request_body = EvalRunRequestBody(name=eval_run_name, judgment_api_key=self.judgment_api_key)
        eval_run = requests.post(JUDGMENT_EVAL_FETCH_API_URL, json=eval_run_request_body.model_dump())
        if eval_run.status_code != requests.codes.ok:
            raise ValueError(f"Error fetching eval results: {eval_run.json()}")
        eval_results = []
        for result in eval_run.json():
            result = result.get("result", dict())
            filtered_result = {k: v for k, v in result.items() if k in ScoringResult.__annotations__}
            eval_results.append(ScoringResult(**filtered_result))
        return eval_results
        
    def _validate_api_key(self):
        """
        Validates that the user api key is valid
        """
        response = requests.post(
            f"{ROOT_API}/validate_api_key/",
            json={"api_key": self.judgment_api_key}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("detail", "Error validating API key")
