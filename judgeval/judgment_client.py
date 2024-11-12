from judgeval.evaluation_run import EvaluationRun
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
        # How to break into this function?
        # Best way to pass in the API key to each EvaluationRun?
        # When you run the properiary metrics (logs + cost tracking, we can charge for # of logs): When you run client.run_eval(…), the method will have internal code to make API requests to run properiarty metrics Judgment backend server, passing the user’s API key 
        if evaluation_run.judgment_api_key is None:
            evaluation_run.judgment_api_key = self.judgment_api_key
        # When you run the other metrics (just logs): client.run_eval(…) will have internal code to log outputs to Judgment backend server. Wherever the for loop is for running the custom metrics/functions, the output value I’ll put that into a request to /log/eval, then server will handle pushing to database
        
    def validate_api_key(self):
        response = requests.post(
            f"{ROOT_API}/validate_api_key/",
            json={"api_key": self.judgment_api_key}
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json().get("message", "Error validating API key")
