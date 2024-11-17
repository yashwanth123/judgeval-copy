from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import run_eval
import requests
from judgeval.constants import ROOT_API
from judgeval.data.datasets.dataset import EvalDataset
from typing import Optional, List
from judgeval.constants import JUDGMENT_EVAL_FETCH_API_URL
from judgeval.data.result import ScoringResult
from pydantic import BaseModel

class EvalRunRequestBody(BaseModel):
    name: str
    judgment_api_key: str


class JudgmentClient:
    def __init__(self, judgment_api_key: str):
        self.judgment_api_key = judgment_api_key
        
        # Verify API key is valid
        result, response = self._validate_api_key()
        if not result:
            # May be bad to output their invalid API key...
            raise ValueError(f"Issue with passed in Judgment API key: {response}")
        else:
            print(f"Successfully initialized JudgmentClient, welcome back {response['user_name']}!")

    def run_eval(self, evaluation_run: EvaluationRun, name: str = "", log_results: bool = False):
        evaluation_run.judgment_api_key = self.judgment_api_key
            
        return run_eval(evaluation_run, name = name, log_results=log_results)
    
    def create_dataset(self) -> EvalDataset:
        return EvalDataset(judgment_api_key=self.judgment_api_key)
    
    def push_dataset(self, alias: str, dataset: EvalDataset, overwrite: Optional[bool] = None):
        # Set judgment_api_key just in case it was not set
        dataset.judgment_api_key = self.judgment_api_key
        return dataset.push(alias, overwrite)
    
    def pull_dataset(self, alias: str) -> EvalDataset:
        dataset = EvalDataset(judgment_api_key=self.judgment_api_key)
        dataset.pull(alias)
        return dataset
    
    def pull_eval(self, eval_run_name: str) -> List[ScoringResult]:
        # TODO: Make a data type for eval_run_request_body
        # IDEA: Maybe add option where you can pass in the EvaluationRun object and it will pull the eval results from the backend
        eval_run_request_body = EvalRunRequestBody(name=eval_run_name, judgment_api_key=self.judgment_api_key)
        eval_run = requests.post(JUDGMENT_EVAL_FETCH_API_URL, json=eval_run_request_body.model_dump())
        if eval_run.status_code != requests.codes.ok:
            raise ValueError(f"Error fetching eval results: {eval_run.json()}")
        # Maybe do Alex's annotation filter
        return [ScoringResult(**(result["results"])) for result in eval_run.json()]
        
    def _validate_api_key(self):
        response = requests.post(
            f"{ROOT_API}/validate_api_key/",
            json={"api_key": self.judgment_api_key}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("message", "Error validating API key")
