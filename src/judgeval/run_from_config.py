from judgeval.scorers.length_penalty_scorer import LengthPenaltyScorer  # Add this import

# If using manual Python scripting, add the scorer like this:

from judgeval import JudgmentClient
from judgeval.data import Example

client = JudgmentClient()

example = Example(
    input="Plan a 3-day trip itinerary for New York including food and sightseeing.",
    actual_output="Day 1: Central Park... Day 2: MoMA... Day 3: Brooklyn...",
    retrieval_context=[],
)

scorers = [
    LengthPenaltyScorer(threshold=0.9),
    # Add any default ones like FaithfulnessScorer if needed
]

results = client.run_evaluation(
    examples=[example],
    scorers=scorers,
    model="gpt-4.1",
    project_name="multi_step_project"
)

print(results)
