
✅ Basic README.md Content You Can Use:

Judgeval Custom Scorer Project Setup

1. Cloning and Setup

Clone the repository.

Install dependencies:

pip install -r requirements.txt

Setup API keys:

export JUDGMENT_API_KEY=your_api_key_here
export JUDGMENT_ORG_ID=your_org_id_here


2. Custom Scorer Implementation

Added src/judgeval/scorers/length_penalty_scorer.py.

Example usage in YAML:

project_name: multi_step_project
eval_name: multi_step_eval
model: gpt-4o
examples:
  - input: "Some input text"
    actual_output: "Some actual output text"
scorers:
  - score_type: length_penalty
    threshold: 0.9


3. Running Evaluations

Use src/judgeval/run_from_config.py to trigger:

python src/judgeval/run_from_config.py evals/multi_step_eval.yaml


4. Notes

I fixed several import issues and circular dependency bugs.

Important edited files: run_evaluation.py, length_penalty_scorer.py, run_from_config.py.


5. Contribution

Fork and continue improving custom scorers.

