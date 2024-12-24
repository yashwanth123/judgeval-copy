from judgeval.judgment_client import JudgmentClient
import os

# It just a scorer, so we should be able to use it like this:

    # client.run_evaluation(
    #     examples=[example1, example2],
    #     scorers=[custom_scorer], # important line,
    #     ...
    # )
    
    # We'll need to modify run_evaluation, to handle the CustomScorers
    
if __name__ == "__main__":
    judgment_client = JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))
    classifier_scorer = judgment_client.fetch_classifier_scorer("Helpfulness Scorer")
    print(classifier_scorer)
    