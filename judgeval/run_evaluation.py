import requests
import litellm
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, field_validator
import os
from judgeval.data import Example
from judgeval.scorers import CustomScorer, JudgmentScorer
from judgeval.scorers.score import *
from judgeval.constants import *
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.judges import TogetherJudge

from judgeval.evaluation_run import EvaluationRun



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
            judgment_api_key=evaluation_run.judgment_api_key,
        )
        response_data = execute_api_eval(api_evaluation_run)  # List[Dict] of converted ScoringResults
        for result in response_data["results"]:  
            # filter for key-value pairs that are used to initialize ScoringResult
            # there may be some stuff in here that doesn't belong in ScoringResult
            # TODO: come back and refactor this to have ScoringResult take in **kwargs
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
        
    # TODO: Once we add logging (pushing eval results to Judgment backend server), we can charge for # of logs
    # Pass in the API key to these log requests.
    # for result in results:
    #   result["judgment_api_key"] = evaluation_run.judgment_api_key
    # requests.post(JUDGMENT_EVAL_API_URL + "/log/eval", json=results.model_dump())

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
    model = TogetherJudge()
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

    run_eval(eval_data)