from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import run_eval
import requests
from judgeval.constants import ROOT_API
from judgeval.data.datasets.dataset import EvalDataset
from typing import Optional
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

    def run_eval(self, evaluation_run: EvaluationRun, log_results: bool = False):
        evaluation_run.judgment_api_key = self.judgment_api_key
            
        return run_eval(evaluation_run, log_results=log_results)
    
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
        
    def _validate_api_key(self):
        response = requests.post(
            f"{ROOT_API}/validate_api_key/",
            json={"api_key": self.judgment_api_key}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("message", "Error validating API key")
