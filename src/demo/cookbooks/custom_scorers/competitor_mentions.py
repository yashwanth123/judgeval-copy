"""
This script implements a custom scorer to evaluate customer support responses.

It checks if a support response mentions competitors (like Adidas, Reebok, etc.) in a positive way.
The scorer penalizes responses that promote competitor products, helping maintain brand focus in
customer interactions. This would be useful to score customer support responses for Nike, for example.
"""


from judgeval import JudgmentClient
from judgeval.scorers import ClassifierScorer
from judgeval.data import Example


competitor_mentions_scorer = ClassifierScorer(
    "Competitor Mentions",
    slug="competitor_mentions-487126418",
    threshold=1.0,
    conversation=[{
        "role": "system",
        "content": """Does the following customer support response discuss any of the following competitors in a positive way? (Y/N). 
        
        Competitors: Adidas, Reebok, Hoka, ON, Converse
        
        Customer Question: {{input}}
        Customer Support Response: {{actual_output}}
        """
    }],
    options={
        "Y": 0.0, 
        "N": 1.0
    }
)


if __name__ == "__main__":
    client = JudgmentClient()

    positive_example = Example(
        input="What are the best shoes for running priced under $130?",
        actual_output="You'd want to check out the newest Nike Vaporfly, it's only $120 and built for performance. "
    )

    negative_example = Example(
        input="What are the best shoes for running priced under $130?",
        actual_output="The Nike Vaporfly is a great shoe built for performance. Other great options include the Adidas Ultraboost and the Reebok Nano X which are affordable and speedy."
    )

    client.run_evaluation(
        examples=[positive_example, negative_example],
        scorers=[competitor_mentions_scorer],
        model="gpt-4o-mini",
        project_name="competitor_mentions",
        eval_run_name="competitor_mentions_test",
    )



