import asyncio
import requests
from typing import List, Dict
from datetime import datetime
from rich import print as rprint

from judgeval.data import (
    Example, 
    ScorerData, 
    ScoringResult
)
from judgeval.scorers import (
    CustomScorer, 
    JudgmentScorer,
    ClassifierScorer
)
from judgeval.scorers.score import a_execute_scoring

from judgeval.constants import (
    ROOT_API,
    JUDGMENT_EVAL_API_URL,
    JUDGMENT_EVAL_LOG_API_URL,
)
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.evaluation_run import EvaluationRun
from judgeval.common.logger import (
    enable_logging, 
    debug, 
    info, 
    error, 
    example_logging_context
)


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
        payload = evaluation_run.model_dump(warnings=False)
        response = requests.post(JUDGMENT_EVAL_API_URL, json=payload)
        response_data = response.json()
    except Exception as e:
        error(f"Error: {e}")
        details = response.json().get("detail", "No details provided")
        raise JudgmentAPIError("An error occurred while executing the Judgment API request: " + details)
    # Check if the response status code is not 2XX
    # Add check for the duplicate eval run name
    if not response.ok:
        error_message = response_data.get('detail', 'An unknown error occurred.')
        error(f"Error: {error_message=}")
        raise JudgmentAPIError(error_message)
    return response_data


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
        
        # Merge ScorerData from the API and local scorers together
        api_scorer_data = api_result.scorers_data
        local_scorer_data = local_result.scorers_data
        if api_scorer_data is None and local_scorer_data is not None:
            api_result.scorers_data = local_scorer_data

        if api_scorer_data is not None and local_scorer_data is not None:
            api_result.scorers_data = api_scorer_data + local_scorer_data
    
    return api_results


def check_missing_scorer_data(results: List[ScoringResult]) -> List[ScoringResult]:
    """
    Checks if any `ScoringResult` objects are missing `scorers_data`.

    If any are missing, logs an error and returns the results.
    """
    for i, result in enumerate(results):
        if not result.scorers_data:
            error(
                f"Scorer data is missing for example {i}. "
                "This is usually caused when the example does not contain "
                "the fields required by the scorer. "
                "Check that your example contains the fields required by the scorers. "
                "TODO add docs link here for reference."
            )
    return results

def check_eval_run_name_exists(eval_name: str, project_name: str, judgment_api_key: str) -> None:
    """
    Checks if an evaluation run name already exists for a given project.

    Args:
        eval_name (str): Name of the evaluation run
        project_name (str): Name of the project
        judgment_api_key (str): API key for authentication

    Raises:
        ValueError: If the evaluation run name already exists
        JudgmentAPIError: If there's an API error during the check
    """
    try:
        response = requests.post(
            f"{ROOT_API}/eval-run-name-exists/",
            json={
                "eval_name": eval_name,
                "project_name": project_name,
                "judgment_api_key": judgment_api_key,
            }
        )
        
        if response.status_code == 409:
            error(f"Evaluation run name '{eval_name}' already exists for this project")
            raise ValueError(f"Evaluation run name '{eval_name}' already exists for this project")
        
        if not response.ok:
            response_data = response.json()
            error_message = response_data.get('detail', 'An unknown error occurred.')
            error(f"Error checking eval run name: {error_message}")
            raise JudgmentAPIError(error_message)
            
    except requests.exceptions.RequestException as e:
        error(f"Failed to check if eval run name exists: {str(e)}")
        raise JudgmentAPIError(f"Failed to check if eval run name exists: {str(e)}")

def log_evaluation_results(merged_results: List[ScoringResult], evaluation_run: EvaluationRun) -> None:
    """
    Logs evaluation results to the Judgment API database.

    Args:
        merged_results (List[ScoringResult]): The results to log
        evaluation_run (EvaluationRun): The evaluation run containing project info and API key

    Raises:
        JudgmentAPIError: If there's an API error during logging
        ValueError: If there's a validation error with the results
    """
    try:
        res = requests.post(
            JUDGMENT_EVAL_LOG_API_URL,
            json={
                "results": [result.to_dict() for result in merged_results],
                "judgment_api_key": evaluation_run.judgment_api_key,
                "project_name": evaluation_run.project_name,
                "eval_name": evaluation_run.eval_name,
            }
        )
        
        if not res.ok:
            response_data = res.json()
            error_message = response_data.get('detail', 'An unknown error occurred.')
            error(f"Error {res.status_code}: {error_message}")
            raise JudgmentAPIError(error_message)
        
        if "ui_results_url" in res.json():
            rprint(f"\n🔍 You can view your evaluation results here: [rgb(106,0,255)]{res.json()['ui_results_url']}[/]\n")
            
    except requests.exceptions.RequestException as e:
        error(f"Request failed while saving evaluation results to DB: {str(e)}")
        raise JudgmentAPIError(f"Request failed while saving evaluation results to DB: {str(e)}")
    except Exception as e:
        error(f"Failed to save evaluation results to DB: {str(e)}")
        raise ValueError(f"Failed to save evaluation results to DB: {str(e)}")

def run_eval(evaluation_run: EvaluationRun, override: bool = False) -> List[ScoringResult]:
    """
    Executes an evaluation of `Example`s using one or more `Scorer`s

    Args:
        evaluation_run (EvaluationRun): Stores example and evaluation together for running
    
        Args: 
            project_name (str): The name of the project the evaluation results belong to
            eval_name (str): The name of the evaluation run
            examples (List[Example]): The examples to evaluate
            scorers (List[Union[JudgmentScorer, CustomScorer]]): A list of scorers to use for evaluation
            model (str): The model used as a judge when using LLM as a Judge
            aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
            metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
            judgment_api_key (Optional[str]): The API key for running evaluations on the Judgment API
            log_results (bool): Whether to log the results to the Judgment API


    Returns:
        List[ScoringResult]: The results of the evaluation. Each result is a dictionary containing the fields of a `ScoringResult` object.
    """
    
    # Call endpoint to check to see if eval run name exists (if we DON'T want to override and DO want to log results)
    if not override and evaluation_run.log_results:
        check_eval_run_name_exists(
            evaluation_run.eval_name,
            evaluation_run.project_name,
            evaluation_run.judgment_api_key
        )
    
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
        if isinstance(scorer, (JudgmentScorer, ClassifierScorer)):
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
        try:  # execute an EvaluationRun with just JudgmentScorers
            api_evaluation_run: EvaluationRun = EvaluationRun(
                eval_name=evaluation_run.eval_name,
                project_name=evaluation_run.project_name,
                examples=evaluation_run.examples,
                scorers=judgment_scorers,
                model=evaluation_run.model,
                aggregator=evaluation_run.aggregator,
                metadata=evaluation_run.metadata,
                judgment_api_key=evaluation_run.judgment_api_key,
                log_results=evaluation_run.log_results
            )
            debug("Sending request to Judgment API")    
            response_data: List[Dict] = execute_api_eval(api_evaluation_run)  # ScoringResults
            info(f"Received {len(response_data['results'])} results from API")
        except JudgmentAPIError as e:
            error(f"An error occurred while executing the Judgment API request: {str(e)}")
            raise JudgmentAPIError(f"An error occurred while executing the Judgment API request: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Please check your EvaluationRun object, one or more fields are invalid: {str(e)}")
        
        # Convert the response data to `ScoringResult` objects
        debug("Processing API results")
        for idx, result in enumerate(response_data["results"]):  
            with example_logging_context(evaluation_run.examples[idx].timestamp, evaluation_run.examples[idx].example_id):
                for scorer in judgment_scorers:
                    debug(f"Processing API result for example {idx} and scorer {scorer.score_type}")
                # filter for key-value pairs that are used to initialize ScoringResult
                # there may be some stuff in here that doesn't belong in ScoringResult
                # TODO: come back and refactor this to have ScoringResult take in **kwargs
                filtered_result = {k: v for k, v in result.items() if k in ScoringResult.__annotations__}
                
                # Convert scorers_data dicts to ScorerData objects
                if "scorers_data" in filtered_result and filtered_result["scorers_data"]:
                    filtered_result["scorers_data"] = [
                        ScorerData(**scorer_dict) 
                        for scorer_dict in filtered_result["scorers_data"]
                    ]
                
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
                model=evaluation_run.model,
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

    # Aggregate the ScorerData from the API and local evaluations
    debug("Merging API and local results")
    merged_results: List[ScoringResult] = merge_results(api_results, local_results)
    merged_results = check_missing_scorer_data(merged_results)

    info(f"Successfully merged {len(merged_results)} results")

    if evaluation_run.log_results:
        log_evaluation_results(merged_results, evaluation_run)

    for i, result in enumerate(merged_results):
        if not result.scorers_data:  # none of the scorers could be executed on this example
            info(f"None of the scorers could be executed on example {i}. This is usually because the Example is missing the fields needed by the scorers. Try checking that the Example has the necessary fields for your scorers.")
    return merged_results
