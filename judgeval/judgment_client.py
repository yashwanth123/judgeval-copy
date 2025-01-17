"""
Implements the JudgmentClient to interact with the Judgment API.
"""
import os
from typing import Optional, List, Dict, Any, Union
import requests

from judgeval.constants import ROOT_API
from judgeval.data.datasets import EvalDataset
from judgeval.data import ScoringResult, Example
from judgeval.scorers import APIJudgmentScorer, JudgevalScorer, ClassifierScorer, ScorerWrapper
from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import run_eval
from judgeval.constants import JUDGMENT_EVAL_FETCH_API_URL
from judgeval.common.exceptions import JudgmentAPIError
from pydantic import BaseModel

class EvalRunRequestBody(BaseModel):
    eval_name: str
    project_name: str
    judgment_api_key: str


class JudgmentClient:
    def __init__(self, judgment_api_key: str = os.getenv("JUDGMENT_API_KEY")):
        self.judgment_api_key = judgment_api_key
        
        # Verify API key is valid
        result, response = self._validate_api_key()
        if not result:
            # May be bad to output their invalid API key...
            raise JudgmentAPIError(f"Issue with passed in Judgment API key: {response}")
        else:
            # TODO: Add logging
            print(f"Successfully initialized JudgmentClient, welcome back {response.get('detail', {}).get('user_name', 'user')}!")
            
    def run_evaluation(
        self, 
        examples: List[Example],
        scorers: List[Union[ScorerWrapper, JudgevalScorer]],
        model: Union[str, List[str]],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_results: bool = False,
        project_name: str = "",
        eval_run_name: str = "",
        override: bool = False,
        use_judgment: bool = True
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of `Example`s using one or more `Scorer`s
        """
        try:
            # Load appropriate implementations for all scorers
            loaded_scorers: List[Union[JudgevalScorer, APIJudgmentScorer]] = [
                scorer.load_implementation(use_judgment=use_judgment) if isinstance(scorer, ScorerWrapper) else scorer
                for scorer in scorers
            ]

            eval = EvaluationRun(
                log_results=log_results,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=examples,
                scorers=loaded_scorers,
                model=model,
                aggregator=aggregator,
                metadata=metadata,
                judgment_api_key=self.judgment_api_key
            )
            return run_eval(eval, override)
        except ValueError as e:
            raise ValueError(f"Please check your EvaluationRun object, one or more fields are invalid: \n{str(e)}")
    
    def evaluate_dataset(
        self, 
        dataset: EvalDataset,
        scorers: List[Union[APIJudgmentScorer, JudgevalScorer]],
        model: Union[str, List[str]],
        aggregator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        project_name: str = "",
        eval_run_name: str = "",
        log_results: bool = False
    ) -> List[ScoringResult]:
        """
        Executes an evaluation of a `EvalDataset` using one or more `Scorer`s
        """
        try:
            evaluation_run = EvaluationRun(
                log_results=log_results,
                project_name=project_name,
                eval_name=eval_run_name,
                examples=dataset.examples,
                scorers=scorers,
                model=model,
                aggregator=aggregator,
                metadata=metadata,
                judgment_api_key=self.judgment_api_key
            )
            return run_eval(evaluation_run)
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
    def pull_eval(self, project_name: str, eval_run_name: str) -> List[ScoringResult]:
        eval_run_request_body = EvalRunRequestBody(project_name=project_name, 
                                                   eval_name=eval_run_name, 
                                                   judgment_api_key=self.judgment_api_key)
        eval_run = requests.post(JUDGMENT_EVAL_FETCH_API_URL, 
                                 json=eval_run_request_body.model_dump())
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

    def fetch_classifier_scorer(self, slug: str) -> ClassifierScorer:
        """
        Fetches a classifier scorer configuration from the Judgment API.

        Args:
            slug (str): Slug identifier of the custom scorer to fetch

        Returns:
            ClassifierScorer: The configured classifier scorer object

        Raises:
            JudgmentAPIError: If the scorer cannot be fetched or doesn't exist
        """
        request_body = {
            "slug": slug,
            "judgment_api_key": self.judgment_api_key
        }
        
        response = requests.post(
            f"{ROOT_API}/fetch_scorer/",
            json=request_body
        )
        
        if response.status_code == 500:
            raise JudgmentAPIError(f"The server is temporarily unavailable. Please try your request again in a few moments. Error details: {response.json().get('detail', '')}")
        elif response.status_code != 200:
            raise JudgmentAPIError(f"Failed to fetch classifier scorer '{slug}': {response.json().get('detail', '')}")
            
        scorer_config = response.json()
        
        try:
            return ClassifierScorer(**scorer_config)
        except Exception as e:
            raise JudgmentAPIError(f"Failed to create classifier scorer '{slug}' with config {scorer_config}: {str(e)}")
