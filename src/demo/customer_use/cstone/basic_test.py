import csv
import os

from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer

if __name__ == "__main__":

    PATH_TO_CSV = os.path.join(os.path.dirname(__file__), "cstone_data.csv")
    client = JudgmentClient()


    with open(PATH_TO_CSV, "r") as f:
        reader = csv.reader(f)
        next(reader) # Skip the header row since we know the structure
        examples = []
        for row in reader:
            docket_id, excerpts, raw_response, quote, is_class_action, note = row

            example = Example(
                input=str(docket_id),
                actual_output=raw_response,
                retrieval_context=[excerpts]
            )
            examples.append(example)

        res = client.run_evaluation(
            examples=examples,
            scorers=[FaithfulnessScorer(threshold=1.0)],
            model="o1-preview",
            eval_run_name="cstone-basic-test",
            project_name="cornerstone_demo_new_o1",
            override=True,
            use_judgment=False,
        )

