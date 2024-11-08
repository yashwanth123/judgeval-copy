"""
Infrastructure for executing evaluations of `Example`s using one or more `CustomScorer`s.
"""


import asyncio
import time 
from tqdm.asyncio import tqdm_asyncio
from typing import List, Union, Optional, Callable
from rich.progress import Progress, SpinnerColumn, TextColumn

from judgeval.common.exceptions import MissingTestCaseParamsError
from judgeval.data.example import Example
from judgeval.data.api_example import create_api_test_case
from judgeval.data.metric_data import create_metric_data
from judgeval.data.result import TestResult, create_test_result
from judgeval.scorers.custom_scorer import CustomScorer
from judgeval.scorers.utils import clone_scorers, format_metric_description
from judgeval.common.telemetry import capture_evaluation_run

from judgeval.data.example import Example 
from judgeval.scorers.custom_scorer import CustomScorer
from judgeval.common.exceptions import MissingTestCaseParamsError

async def safe_a_score_example(
    scorer: CustomScorer,
    example: Example,
    ignore_errors: bool,
    skip_on_missing_params: bool,
):
    """
    Scores an `Example` using a `CustomScorer` and handles any exceptions that may occur.

    Args:
        scorer (CustomScorer): The `CustomScorer` to use for scoring the example.
        example (Example): The `Example` to be scored.
        
        ignore_errors (bool): Whether to ignore errors during the evaluation. 
        If set to false, any error will be raised and stop the evaluation.
        If set to true, the error will be stored in the `error` attribute of the `CustomScorer` and the `success` attribute will be set to False.
        
        skip_on_missing_params (bool): Whether to skip the test case if required parameters are missing. 
    """
    try:
        await scorer.a_score_example(example, _show_indicator=False)
    except MissingTestCaseParamsError as e:
        if skip_on_missing_params:  # If the test case is missing required parameters to execute, skip it
            scorer.skipped = True
            return
        else:
            if ignore_errors:  # Gracefully handle the error, does not stop the evaluation
                scorer.error = str(e)
                scorer.success = False
            else:  # Raise the error and stop the evaluation
                raise
    except TypeError:  # in case a_score_example does not accept _show_indicator
        try:
            await scorer.a_score_example(example)
        except MissingTestCaseParamsError as e:
            if skip_on_missing_params:
                scorer.skipped = True
                return
            else:
                if ignore_errors:
                    scorer.error = str(e)
                    scorer.success = False
                else:
                    raise
    except Exception as e:
        if ignore_errors:
            scorer.error = str(e)
            scorer.success = False  # Assuming you want to set success to False
        else:
            raise


async def score_task(
    task_id: int,
    progress: Progress,
    scorer: CustomScorer,
    example: Example,
    ignore_errors: bool = True,
    skip_on_missing_params: bool = True,
):
    """
    Task function for asynchronously measuring a given example using a custom scorer.

    Args:
        task_id (int): The ID of the task being measured.
        progress (Progress): An instance of the Progress class to track task progress.
        scorer (CustomScorer): An instance of the CustomScorer class used to score the example.
        example (Example): The example to be scored.
        ignore_errors (bool, optional): Whether to ignore errors during scoring. Defaults to True.
        skip_on_missing_params (bool, optional): Whether to skip scoring if there are missing parameters. Defaults to True.

    Raises:
        MissingTestCaseParamsError: If required test case parameters are missing and skip_on_missing_params is False.
        Exception: If an unexpected error occurs and ignore_errors is False.

    Returns:
        None
    """
    while not progress.finished:
        start_time = time.perf_counter()
        metric_data = None

        if metric_data:
            ## only change metric state, not configs
            scorer.score = metric_data.score
            scorer.success = metric_data.success
            scorer.reason = metric_data.reason
            scorer.evaluation_cost = metric_data.evaluation_cost
            scorer.verbose_logs = metric_data.verbose_logs
            finish_text = "Read from Cache"
        else:
            try:
                await scorer.a_score_example(example, _show_indicator=False)
                finish_text = "Done"
            except MissingTestCaseParamsError as e:
                if skip_on_missing_params:
                    scorer.skipped = True
                    return
                else:
                    if ignore_errors:
                        scorer.error = str(e)
                        scorer.success = False  # Override metric success
                        finish_text = "Failed"
                    else:
                        raise
            except TypeError:
                try:
                    await scorer.a_score_example(example)
                    finish_text = "Done"
                except MissingTestCaseParamsError as e:
                    if skip_on_missing_params:
                        scorer.skipped = True
                        return
                    else:
                        if ignore_errors:
                            scorer.error = str(e)
                            scorer.success = False  # Override metric success
                            finish_text = "Failed"
                        else:
                            raise
            except Exception as e:
                if ignore_errors:
                    scorer.error = str(e)
                    scorer.success = False  # Override metric success
                    finish_text = "Failed"
                else:
                    raise

        end_time = time.perf_counter()
        time_taken = format(end_time - start_time, ".2f")
        progress.update(task_id, advance=100)  # Mark task as complete
        progress.update(
            task_id,
            description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]{finish_text}! ({time_taken}s)",
        )
        break


async def score_with_indicator(
    scorers: List[CustomScorer],
    example: Example,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
):
    """
    Scores an example using a list of custom scorers, optionally displaying a progress indicator.

    Args:
        scorers (List[CustomScorer]): A list of custom scorer objects to evaluate the example.
        example (Example): The example to be scored.
        ignore_errors (bool): If True, errors during scoring will be ignored.
        skip_on_missing_params (bool): If True, scoring will be skipped if required parameters are missing.
        show_indicator (bool): If True, a progress indicator will be displayed during scoring.

    Returns:
        None

    Raises:
        Any exceptions raised by the scoring functions, unless `ignore_errors` is True.
    """
    if show_indicator:
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            tasks = []
            for scorer in scorers:
                task_id = progress.add_task(
                    description=format_metric_description(
                        scorer, async_mode=True
                    ),
                    total=100,
                )  # Add task to progress bar
                tasks.append(
                    score_task(
                        task_id,
                        progress,
                        scorer,
                        example,
                        ignore_errors,
                        skip_on_missing_params,
                    )  # Create and execute task to score the example with a single scorer
                )
            await asyncio.gather(*tasks)
    else:
        tasks = []
        for scorer in scorers:
            metric_data = None

            if metric_data:
                scorer.score = metric_data.score
                scorer.threshold = metric_data.threshold
                scorer.success = metric_data.success
                scorer.reason = metric_data.reason
                scorer.strict_mode = metric_data.strict_mode
                scorer.evaluation_model = metric_data.evaluation_model
                scorer.evaluation_cost = metric_data.evaluation_cost
                scorer.verbose_logs = metric_data.verbose_logs
            else:
                tasks.append(
                    safe_a_score_example(
                        scorer, example, ignore_errors, skip_on_missing_params
                    )
                )

        await asyncio.gather(*tasks)


async def a_execute_test_cases(
    test_cases: List[Union[Example]],
    metrics: List[CustomScorer],
    ignore_errors: bool,
    skip_on_missing_params: bool,
    use_cache: bool,
    show_indicator: bool,
    throttle_value: int,
    max_concurrent: int,
    verbose_mode: Optional[bool] = None,
    _use_bar_indicator: bool = True,
) -> List[TestResult]:
    """
    Executes evaluations of `Example`s asynchronously using one or more `CustomScorer`s.
    Each `Example` will be evaluated all of the `CustomScorer`s in the `metrics` list.

    Args:
        TODO
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(func: Callable, *args, **kwargs):
        async with semaphore:
            return await func(*args, **kwargs)

    if verbose_mode is not None:
        for metric in metrics:
            metric.verbose_mode = verbose_mode

    llm_test_case_counter = -1
    test_results: List[TestResult] = [None for _ in range(len(test_cases))]
    tasks = []

    if show_indicator and _use_bar_indicator:
        with tqdm_asyncio(
            desc=f"Evaluating {len(test_cases)} test case(s) in parallel",
            unit="test case",
            total=len(test_cases),
            bar_format="{desc}: |{bar}|{percentage:3.0f}% ({n_fmt}/{total_fmt}) [Time Taken: {elapsed}, {rate_fmt}{postfix}]",
        ) as pbar:
            for i, test_case in enumerate(test_cases):
                with capture_evaluation_run("test case"):
                    if isinstance(test_case, Example):
                        if len(metrics) == 0:
                            pbar.update(1)
                            continue

                        llm_test_case_counter += 1
                        copied_llm_metrics: List[CustomScorer] = clone_scorers(
                            metrics
                        )
                        task = execute_with_semaphore(
                            func=a_execute_llm_test_cases,
                            metrics=copied_llm_metrics,
                            test_case=test_case,
                            test_results=test_results,
                            test_index=i,
                            count=llm_test_case_counter,
                            ignore_errors=ignore_errors,
                            skip_on_missing_params=skip_on_missing_params,
                            show_indicator=show_indicator,
                            _use_bar_indicator=_use_bar_indicator,
                            pbar=pbar,
                        )
                        tasks.append(asyncio.create_task(task))

                    await asyncio.sleep(throttle_value)
            await asyncio.gather(*tasks)
    else:
        for i, test_case in enumerate(test_cases):
            with capture_evaluation_run("test case"):
                if isinstance(test_case, Example):
                    if len(metrics) == 0:
                        continue
                    llm_test_case_counter += 1

                    copied_llm_metrics: List[CustomScorer] = clone_scorers(
                        metrics
                    )
                    task = execute_with_semaphore(
                        func=a_execute_llm_test_cases,
                        metrics=copied_llm_metrics,
                        test_case=test_case,
                        test_results=test_results,
                        test_index=i,
                        count=llm_test_case_counter,
                        ignore_errors=ignore_errors,
                        skip_on_missing_params=skip_on_missing_params,
                        use_cache=use_cache,
                        _use_bar_indicator=_use_bar_indicator,
                        show_indicator=show_indicator,
                    )
                    tasks.append(asyncio.create_task((task)))

                await asyncio.sleep(throttle_value)
        await asyncio.gather(*tasks)

    return test_results


async def a_execute_llm_test_cases(
    metrics: List[CustomScorer],
    test_case: Example,
    test_results: List[TestResult],
    test_index: int,
    count: int,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
    _use_bar_indicator: bool,
    pbar: Optional[tqdm_asyncio] = None,
):
    """
    Execute a single LLM test case asynchronously and evaluate it using a list of metrics.

    Args:
        metrics (List[BaseMetric]): A list of metric objects to evaluate the test cases.
        test_case (Example): The test case to be evaluated.
        test_run_manager (TestRunManager): The manager handling the test run.
        test_results (List[Union[TestResult, MLLMExample]]): A list to store the results of the test cases.
        test_index (int): The index of the test case in the test case list.
        count (int): The current count of test cases processed.
        test_run (TestRun): The test run configuration and metadata.
        ignore_errors (bool): Whether to ignore errors during the evaluation.
        skip_on_missing_params (bool): Whether to skip the test case if required parameters are missing.
        use_cache (bool): Whether to use cached results for the test case.
        show_indicator (bool): Whether to show progress indicators.
        _use_bar_indicator (bool): Whether to use a bar indicator for progress.
        pbar (Optional[tqdm_asyncio]): An optional progress bar object for tracking progress.

    Returns:
        None: The results are appended to the test_results list.
    """
    show_metrics_indicator = show_indicator and not _use_bar_indicator

    for metric in metrics:
        metric.skipped = False
        metric.error = None  # Reset metric error

    ##### Metric Calculation #####
    api_test_case = create_api_test_case(test_case, count)  # Creates API Test Case to track the progress/success
    test_start_time = time.perf_counter()
    await score_with_indicator(
        scorers=metrics,
        example=test_case,
        skip_on_missing_params=skip_on_missing_params,
        ignore_errors=ignore_errors,
        show_indicator=show_metrics_indicator,
    )  # execute the measure functions of each metric on the test case

    # Now that all the measure functions of each metric have executed, we collect 
    # the results and update the API Test Case with the metric data
    for metric in metrics:
        # At this point, the metric has been executed and already contains data.
        if metric.skipped:
            continue
        
        metric_data = create_metric_data(metric)  # Fetch metric data from completed metric evaluation
        api_test_case.update_metric_data(metric_data)  # Update API Test Case with the same metric data, including cost
          
    test_end_time = time.perf_counter()
    run_duration = test_end_time - test_start_time
    # Quick hack to check if all metrics were from cache
    if run_duration < 1:
        run_duration = 0
    api_test_case.update_run_duration(run_duration)   # Update API Test Case with execution time duration
    test_results[test_index] = create_test_result(api_test_case)  # Converts the outcomes of the executed test to a TestResult and saves it

    if pbar is not None:
        pbar.update(1)
