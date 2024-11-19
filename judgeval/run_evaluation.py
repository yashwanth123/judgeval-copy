import asyncio
import requests
from typing import List, Dict
from datetime import datetime

from judgeval.data import Example
from judgeval.scorers import CustomScorer, JudgmentScorer
from judgeval.scorers.score import (
    ScoringResult,
    a_execute_scoring,
)
from judgeval.constants import (
    JUDGMENT_EVAL_API_URL,
    JudgmentMetric,
)
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.judges import TogetherJudge, MixtureOfJudges
from judgeval.evaluation_run import EvaluationRun
from judgeval.common.logger import enable_logging, debug, info, error, example_logging_context



def execute_api_eval(evaluation_run: EvaluationRun) -> List[Dict]:
    """
    Executes an evaluation of a list of `Example`s using one or more `JudgmentScorer`s via the Judgment API.

    Args:
        evaluation_run (EvaluationRun): The evaluation run object containing the examples, scorers, and metadata

    Returns:
        List[Dict]: The results of the evaluation. Each result is a dictionary containing the fields of a `ScoringResult`
                    object. 
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
    When executing scorers that come from both the Judgment API and custom scorers, we're left with
    results for each type of scorer. This function merges the results from the API and local evaluations,
    grouped by example. In particular, we merge the `scorers_data` field of each `ScoringResult` object.

    Args:
        api_results (List[ScoringResult]): The `ScoringResult`s from the API evaluation
        local_results (List[ScoringResult]): The `ScoringResult`s from the local evaluation

    Returns:
        List[ScoringResult]: The merged `ScoringResult`s (updated `scorers_data` field)
    """
    # No merge required
    if not local_results and api_results:
        return api_results
    if not api_results and local_results:
        return local_results

    if len(api_results) != len(local_results):
        # Results should be of same length because each ScoringResult is a 1-1 mapping to an Example
        raise ValueError(f"The number of API and local results do not match: {len(api_results)} vs {len(local_results)}")
    
    # Each ScoringResult in api and local have all the same fields besides `scorers_data`
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
        api_scorer_data = api_result.scorers_data
        local_scorer_data = local_result.scorers_data
        if api_scorer_data is None and local_scorer_data is not None:
            api_result.scorers_data = local_scorer_data

        if api_scorer_data is not None and local_scorer_data is not None:
            api_result.scorers_data = api_scorer_data + local_scorer_data
    
    return api_results


def run_eval(evaluation_run: EvaluationRun):
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s

    Args:
        evaluation_run (EvaluationRun): Stores example and evaluation together for running
    
        Args: 
            examples (List[Example]): The examples to evaluate
            scorers (List[Union[JudgmentScorer, CustomScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
            metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
            judgment_api_key (Optional[str]): The API key for running evaluations on the Judgment API

    Returns:
        List[ScoringResult]: The results of the evaluation. Each result is a dictionary containing the fields of a `ScoringResult` object.
    """
    # Set example IDs if not already set
    debug("Initializing examples with IDs and timestamps")
    for idx, example in enumerate(evaluation_run.examples):
        if example.example_id is None:
            example.example_id = idx
            debug(f"Set example ID {idx} for input: {example.input[:50]}...")
        example.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with example_logging_context(example.timestamp, example.example_id):
            debug(f"Initialized example {example.example_id}")
            debug(f"Input: {example.input}")
            debug(f"Actual output: {example.actual_output}")
            if example.expected_output:
                debug(f"Expected output: {example.expected_output}")
            if example.context:
                debug(f"Context: {example.context}")
            if example.retrieval_context:
                debug(f"Retrieval context: {example.retrieval_context}")
    
    debug(f"Starting evaluation run with {len(evaluation_run.examples)} examples")
    
    # Group JudgmentScorers and CustomScorers, then evaluate them in parallel
    debug("Grouping scorers by type")
    judgment_scorers: List[JudgmentScorer] = []
    custom_scorers: List[CustomScorer] = []
    for scorer in evaluation_run.scorers:
        if isinstance(scorer, JudgmentScorer):
            judgment_scorers.append(scorer)
            debug(f"Added judgment scorer: {type(scorer).__name__}")
        else:
            custom_scorers.append(scorer)
            debug(f"Added custom scorer: {type(scorer).__name__}")
    
    debug(f"Found {len(judgment_scorers)} judgment scorers and {len(custom_scorers)} custom scorers")
    
    api_results: List[ScoringResult] = []
    local_results: List[ScoringResult] = []
    # Execute evaluation using Judgment API
    if judgment_scorers:
        info("Starting API evaluation")
        debug(f"Creating API evaluation run with {len(judgment_scorers)} scorers")
        api_evaluation_run: EvaluationRun = EvaluationRun(
            examples=evaluation_run.examples,
            scorers=judgment_scorers,
            model=evaluation_run.model,
            aggregator=evaluation_run.aggregator,
            metadata=evaluation_run.metadata,
            judgment_api_key=evaluation_run.judgment_api_key,
        )
        debug("Sending request to Judgment API")
        response_data = execute_api_eval(api_evaluation_run)  # List[Dict] representing ScoringResults
        info(f"Received {len(response_data['results'])} results from API")
        
        # Convert the response data to `ScoringResult` objects
        debug("Processing API results")
        for idx, result in enumerate(response_data["results"]):  
            with example_logging_context(evaluation_run.examples[idx].timestamp, evaluation_run.examples[idx].example_id):
                debug(f"Processing API result for example {idx}")
                # filter for key-value pairs that are used to initialize ScoringResult
                # there may be some stuff in here that doesn't belong in ScoringResult
                # TODO: come back and refactor this to have ScoringResult take in **kwargs
                filtered_result = {k: v for k, v in result.items() if k in ScoringResult.__annotations__}
                api_results.append(ScoringResult(**filtered_result))

    # Run local evals
    if custom_scorers:  # List[CustomScorer]
        info("Starting local evaluation")
        for example in evaluation_run.examples:
            with example_logging_context(example.timestamp, example.example_id):
                debug(f"Processing example {example.example_id}: {example.input}")
        
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
        info(f"Local evaluation complete with {len(local_results)} results")
        
    # TODO: Once we add logging (pushing eval results to Judgment backend server), we can charge for # of logs
    # Pass in the API key to these log requests.
    # for result in results:
    #   result["judgment_api_key"] = evaluation_run.judgment_api_key
    # requests.post(JUDGMENT_EVAL_API_URL + "/log/eval", json=results.model_dump())

    # Aggregate the ScorerData from the API and local evaluations
    debug("Merging API and local results")
    merged_results = merge_results(api_results, local_results)
    info(f"Successfully merged {len(merged_results)} results")
    return merged_results


if __name__ == "__main__":
    from judgeval.common.logger import enable_logging, debug, info
    
    # TODO comeback and delete this, move this to a demo example
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

    # model = MixtureOfJudges()
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

    with enable_logging():
        debug("Starting evaluation")
        results = run_eval(eval_data)
        info("Evaluation complete")
        debug(f"Results: {results}")
