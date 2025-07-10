import os
os.environ["JUDGMENT_API_KEY"] = "your key"
os.environ["JUDGMENT_ORG_ID"] = "your id" 

from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

# Set env vars directly in the script for now:

client = JudgmentClient()

example = Example(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

scorer = FaithfulnessScorer(threshold=0.5)

results = client.run_evaluation(
    examples=[example],
    scorers=[scorer],
    model="gpt-4o",
    project_name="yashwanth-sandbox"
)

print(results)
