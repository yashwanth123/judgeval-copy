from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

client = JudgmentClient()

example = Example(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

scorer = FaithfulnessScorer(threshold=0.5)
results = client.a_run_evaluation(
    eval_run_name="test-run",
    examples=[example],
    scorers=[scorer],
    model="gpt-4o",
    override=True,
)
print(results)