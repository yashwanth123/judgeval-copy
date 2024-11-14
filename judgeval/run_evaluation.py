import requests
import pprint
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, field_validator

from judgeval.data import Example
from judgeval.scorers import CustomScorer, JudgmentScorer
from judgeval.scorers.score import *
from judgeval.constants import *
from judgeval.litellm_model_names import LITE_LLM_MODEL_NAMES
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.judges import TogetherJudge
from judgeval.judges.mixture_of_judges import MixtureOfJudges

ACCEPTABLE_MODELS = LITE_LLM_MODEL_NAMES | set(TOGETHER_SUPPORTED_MODELS.keys())

class EvaluationRun(BaseModel):
    """
    Stores example and evaluation together for running
    
    Args: 
        examples (List[Example]): The examples to evaluate
        scorers (List[Union[JudgmentScorer, CustomScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
    """
    examples: List[Example]
    scorers: List[Union[JudgmentScorer, CustomScorer]]
    model: Union[str, List[str]]
    aggregator: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator('examples')
    def validate_examples(cls, v):
        if not v:
            raise ValueError("Examples cannot be empty.")
        for ex in v:
            if not isinstance(ex, Example):
                raise ValueError(f"Invalid type for Example: {type(ex)}")
        return v

    @field_validator('scorers')
    def validate_scorers(cls, v):
        if not v:
            raise ValueError("Scorers cannot be empty.")
        for s in v:
            if not isinstance(s, JudgmentScorer) and not isinstance(s, CustomScorer):
                raise ValueError(f"Invalid type for Scorer: {type(s)}")
        return v

    @field_validator('model')
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model cannot be empty.")
        if not isinstance(v, str) and not isinstance(v, list):
            raise ValueError("Model must be a string or a list of strings.")
        if isinstance(v, str) and v not in ACCEPTABLE_MODELS:
            raise ValueError(f"Model name {v} not recognized.")
        if isinstance(v, list):
            for m in v:
                if m not in ACCEPTABLE_MODELS:
                    raise ValueError(f"Model name {m} not recognized.")
        return v

    @field_validator('aggregator', mode='before')
    def validate_aggregator(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("Aggregator must be a string if provided.")
        if v is not None and v not in ACCEPTABLE_MODELS:
            raise ValueError(f"Model name {v} not recognized.")
        return v

    class Config:
        arbitrary_types_allowed = True


def execute_api_eval(evaluation_run: EvaluationRun) -> Any:  # TODO add return type
    """
    Executes an evaluation of a list of `Example`s using one or more `JudgmentScorer`s via the Judgment API
    """
    try:
        # submit API request to execute evals
        response = requests.post(JUDGMENT_EVAL_API_URL, json=evaluation_run.model_dump())
        response_data = response.json()
        
        # Check if the response status code is not 2XX
        if not response.ok:
            error_message = response_data.get('message', 'An unknown error occurred.')
            raise Exception(f"Error {response.status_code}: {error_message}")
        return response_data
    except requests.exceptions.RequestException as e:
        raise JudgmentAPIError(f"An internal error occurred while executing the Judgment API request: {str(e)}")
    except Exception as e:
        raise ValueError(f"An error occurred while executing the Judgment API request: {str(e)}")


def merge_results(api_results: List[ScoringResult], local_results: List[ScoringResult]) -> List[ScoringResult]:
    """
    Merges the results from the API and local evaluations

    Args:
        api_results (List[ScoringResult]): The results from the API evaluation
        local_results (List[ScoringResult]): The results from the local evaluation
    """
    if not local_results and api_results:
        return api_results
    if not api_results and local_results:
        return local_results

    # Merge ScorerData fields
    if len(api_results) != len(local_results):
        raise ValueError("The number of API and local results do not match.")
    
    # we expect that each ScoringResult in api and local have all the same fields besides metrics_data
    for api_result, local_result in zip(api_results, local_results):
        if api_result.input != local_result.input:
            raise ValueError("The API and local results are not aligned.")
        if api_result.actual_output != local_result.actual_output:
            raise ValueError("The API and local results are not aligned.")
        if api_result.expected_output != local_result.expected_output:
            raise ValueError("The API and local results are not aligned.")
        if api_result.context != local_result.context:
            raise ValueError("The API and local results are not aligned.")
        if api_result.retrieval_context != local_result.retrieval_context:
            raise ValueError("The API and local results are not aligned.")
        
        # Merge ScorerData
        api_metric_data = api_result.metrics_data
        local_metric_data = local_result.metrics_data
        if api_metric_data is None and local_metric_data is not None:
            api_result.metrics_data = local_metric_data

        if api_metric_data is not None and local_metric_data is not None:
            if len(api_metric_data) != len(local_metric_data):
                raise ValueError("The number of API and local metrics data do not match.")
            api_result.metrics_data = api_metric_data + local_metric_data
    
    return api_results


def run_eval(evaluation_run: EvaluationRun):
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s
    """
    
    # Group JudgmentScorers and CustomScorers and evaluate them in async parallel
    judgment_scorers: List[JudgmentScorer] = []
    custom_scorers: List[CustomScorer] = []
    for scorer in evaluation_run.scorers:
        if isinstance(scorer, JudgmentScorer):
            judgment_scorers.append(scorer)
        else:
            custom_scorers.append(scorer)
    
    api_results, local_results = [], []
    # Execute evaluation using Judgment API
    if judgment_scorers:
        api_evaluation_run: EvaluationRun = EvaluationRun(
            examples=evaluation_run.examples,
            scorers=judgment_scorers,
            model=evaluation_run.model,
            aggregator=evaluation_run.aggregator,
            metadata=evaluation_run.metadata,
        )
        response_data = execute_api_eval(api_evaluation_run)  # List[Dict] of converted ScoringResults
        for result in response_data["results"]:
            filtered_result = {k: v for k, v in result.items() if k in ScoringResult.__annotations__}
            api_results.append(ScoringResult(**filtered_result))

    # Run local evals
    if custom_scorers:  # List[CustomScorer]
        results: List[ScoringResult] = asyncio.run(
            a_execute_scoring(
                evaluation_run.examples,
                custom_scorers,
                ignore_errors=True,
                skip_on_missing_params=True,
                show_indicator=True,
                _use_bar_indicator=True,
                throttle_value=0,
                max_concurrent=100,
            )
        )
        local_results = results

    # Aggregate the ScorerData
    merged_results = merge_results(api_results, local_results)
    return merged_results


if __name__ == "__main__":
    # Eval using a proprietary Judgment Scorer
    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )

    example2 = Example(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        name="Password Reset",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"}
    )

    scorer = JudgmentScorer(threshold=0.5, score_type=JudgmentMetric.FAITHFULNESS)
    scorer2 = JudgmentScorer(threshold=0.5, score_type=JudgmentMetric.HALLUCINATION)
    # model = TogetherJudge()

    model = MixtureOfJudges()
    c_scorer = CustomFaithfulnessMetric(
        threshold=0.6,
        model=model,
    )

    eval_data = EvaluationRun(
        examples=[example1, example2],
        scorers=[c_scorer],
        metadata={"batch": "test"},
        model=["QWEN", "MISTRAL_8x7B_INSTRUCT"],
        aggregator='QWEN'
    )

    results = run_eval(eval_data)

    print(f"Results: {results}")
