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
            print(f"Successfully initialized JudgmentClient, welcome back {response['user_name']}!")

    def run_evaluation(
        self, 
        examples: List[Example],
        scorers: List[Union[JudgmentScorer, CustomScorer]],
        model: Union[str, List[str]],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of `Example`s using one or more `Scorer`s
        """
        eval = EvaluationRun(
            examples=examples,
            scorers=scorers,
            model=model,
            aggregator=aggregator,
            metadata=metadata,
            judgment_api_key=self.judgment_api_key
        )
        return run_eval(eval)
    
    def evaluate_dataset(
        self, 
        dataset: EvalDataset,
        scorers: List[Union[JudgmentScorer, CustomScorer]],
        model: Union[str, List[str]],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ) -> List[ScoringResult]:
        """
        Executes an evaluation of a `EvalDataset` using one or more `Scorer`s
        """
        evaluation_run = EvaluationRun(
            examples=dataset.examples,
            scorers=scorers,
            model=model,
            aggregator=aggregator,
            metadata=metadata,
            judgment_api_key=self.judgment_api_key,
        )
        return run_eval(evaluation_run)

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
            return False, response.json().get("message", "Error validating API key")
