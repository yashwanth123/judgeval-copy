import os
# Set up environment variables
os.environ["JUDGMENT_API_KEY"] = "your key"
os.environ["JUDGMENT_ORG_ID"] = "your id"

from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers.length_penalty_scorer import LengthPenaltyScorer


client = JudgmentClient()

example1 = Example(
    input="Short sentence?",
    actual_output="This is short."
)

example2 = Example(
    input="Write something long.",
    actual_output="word " * 120
)

scorer = LengthPenaltyScorer()
scorer.max_length = 50
scorer.threshold = 0.5


results = client.run_evaluation(
    examples=[example1, example2],
    scorers=[scorer],
    model="gpt-4o",
    project_name="length-penalty-test"
)

print(results)

