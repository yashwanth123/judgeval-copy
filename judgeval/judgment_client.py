from judgeval.evaluation_run import EvaluationRun
from judgeval.run_evaluation import run_eval
import requests
from judgeval.constants import ROOT_API

class JudgmentClient:
    def __init__(self, judgment_api_key: str):
        self.judgment_api_key = judgment_api_key
        
        # Verify API key is valid
        result, response = self.validate_api_key()
        if not result:
            # May be bad to output their invalid API key...
            raise ValueError(f"Issue with passed in Judgment API key: {response}")
        else:
            print(f"Successfully initialized JudgmentClient, welcome back {response['user_name']}!")

    def run_eval(self, evaluation_run: EvaluationRun):
        evaluation_run.judgment_api_key = self.judgment_api_key
            
        return run_eval(evaluation_run)
        
    def validate_api_key(self):
        response = requests.post(
            f"{ROOT_API}/validate_api_key/",
            json={"api_key": self.judgment_api_key}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("message", "Error validating API key")
