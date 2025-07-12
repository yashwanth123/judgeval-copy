# Judgeval Custom Evaluation Example

This project demonstrates customizing the Judgeval evaluation framework by adding:

* A new local scorer: `LengthPenaltyScorer`
* A custom YAML evaluation config: `multi_step_eval.yaml`
* Integration setup using `run_from_config.py`

## ✅ What We Edited and Added

| File Path                                       | Purpose                                                                  |
| :---------------------------------------------- | :----------------------------------------------------------------------- |
| `src/judgeval/run_from_config.py`               | Runner script that loads YAML config and triggers `run_eval`.            |
| `src/judgeval/scorers/length_penalty_scorer.py` | Custom scorer that penalizes long outputs.                               |
| `src/judgeval/run_evaluation.py`                | Patched to handle `None` values in local scorers and merge logic safely. |
| `evals/multi_step_eval.yaml`                    | Example config file using `LengthPenaltyScorer`.                         |

---

## ✅ How to Use

### 1️⃣ Install Judgeval and Dependencies

```bash
pip install judgeval
```

### 2️⃣ Set Up Environment Variables

Make sure to replace these with your own values from [https://app.judgeval.ai](https://app.judgeval.ai):

```bash
export JUDGMENT_API_KEY=your-key-here
export JUDGMENT_ORG_ID=your-org-id-here
```

Or inline:

```bash
JUDGMENT_API_KEY="your-key" JUDGMENT_ORG_ID="your-org" python src/judgeval/run_from_config.py evals/multi_step_eval.yaml
```

### 3️⃣ Run Evaluation

```bash
python src/judgeval/run_from_config.py evals/multi_step_eval.yaml
```

You’ll see console output and a UI link if everything works.

---

## ✅ Notes

* If you see errors like `'NoneType' object has no attribute 'model_copy'`, check:

  * `merge_results()` in `run_evaluation.py` should filter out None objects safely.
* For circular imports:

  * Make sure `ScorerData` and `Example` are imported from their exact file locations, not via `__init__.py`.
* Judgeval version: tested up to `0.0.52`.
