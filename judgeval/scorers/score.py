"""
Use Scorer to score an Example
"""


import asyncio
import time 
from tqdm.asyncio import tqdm_asyncio
from typing import List, Union, Optional, Callable, Dict, Any
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

async def safe_a_measure(
    metric: CustomScorer,
    tc: Example,
    ignore_errors: bool,
    skip_on_missing_params: bool,
):
    try:
        await metric.a_score_example(tc, _show_indicator=False)
    except MissingTestCaseParamsError as e:
        if skip_on_missing_params:
            metric.skipped = True
            return
        else:
            if ignore_errors:
                metric.error = str(e)
                metric.success = False
            else:
                raise
    except TypeError:
        try:
            await metric.a_score_example(tc)
        except MissingTestCaseParamsError as e:
            if skip_on_missing_params:
                metric.skipped = True
                return
            else:
                if ignore_errors:
                    metric.error = str(e)
                    metric.success = False
                else:
                    raise
    except Exception as e:
        if ignore_errors:
            metric.error = str(e)
            metric.success = False  # Assuming you want to set success to False
        else:
            raise


async def measure_metric_task(
    task_id,
    progress,
    metric: CustomScorer,
    test_case: Example,
    cached_test_case: None = None,
    ignore_errors: bool = True,
    skip_on_missing_params: bool = True,
):
    while not progress.finished:
        start_time = time.perf_counter()
        metric_data = None

        if metric_data:
            ## only change metric state, not configs
            metric.score = metric_data.score
            metric.success = metric_data.success
            metric.reason = metric_data.reason
            metric.evaluation_cost = metric_data.evaluation_cost
            metric.verbose_logs = metric_data.verbose_logs
            finish_text = "Read from Cache"
        else:
            try:
                await metric.a_score_example(test_case, _show_indicator=False)
                finish_text = "Done"
            except MissingTestCaseParamsError as e:
                if skip_on_missing_params:
                    metric.skipped = True
                    return
                else:
                    if ignore_errors:
                        metric.error = str(e)
                        metric.success = False  # Override metric success
                        finish_text = "Errored"
                    else:
                        raise
            except TypeError:
                try:
                    await metric.a_score_example(test_case)
                    finish_text = "Done"
                except MissingTestCaseParamsError as e:
                    if skip_on_missing_params:
                        metric.skipped = True
                        return
                    else:
                        if ignore_errors:
                            metric.error = str(e)
                            metric.success = False  # Override metric success
                            finish_text = "Errored"
                        else:
                            raise
            except Exception as e:
                if ignore_errors:
                    metric.error = str(e)
                    metric.success = False  # Override metric success
                    finish_text = "Errored"
                else:
                    raise

        end_time = time.perf_counter()
        time_taken = format(end_time - start_time, ".2f")
        progress.update(task_id, advance=100)
        progress.update(
            task_id,
            description=f"{progress.tasks[task_id].description} [rgb(25,227,160)]{finish_text}! ({time_taken}s)",
        )
        break


async def measure_metrics_with_indicator(
    metrics: List[
        CustomScorer
    ],
    test_case: Example,
    cached_test_case: None,
    ignore_errors: bool,
    skip_on_missing_params: bool,
    show_indicator: bool,
):
    if show_indicator:
        with Progress(
            SpinnerColumn(style="rgb(106,0,255)"),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            tasks = []
            for metric in metrics:
                task_id = progress.add_task(
                    description=format_metric_description(
                        metric, async_mode=True
                    ),
                    total=100,
                )
                tasks.append(
                    measure_metric_task(
                        task_id,
                        progress,
                        metric,
                        test_case,
                        cached_test_case,
                        ignore_errors,
                        skip_on_missing_params,
                    )
                )
            await asyncio.gather(*tasks)
    else:
        tasks = []
        for metric in metrics:
            metric_data = None

            if metric_data:
                ## Here we're setting the metric state from metrics metadata cache,
                ## and later using the metric state to create a new metrics metadata cache
                ## WARNING: Potential for bugs, what will happen if a metric changes state in between
                ## test cases?
                metric.score = metric_data.score
                metric.threshold = metric_data.threshold
                metric.success = metric_data.success
                metric.reason = metric_data.reason
                metric.strict_mode = metric_data.strict_mode
                metric.evaluation_model = metric_data.evaluation_model
                metric.evaluation_cost = metric_data.evaluation_cost
                metric.verbose_logs = metric_data.verbose_logs
            else:
                tasks.append(
                    safe_a_measure(
                        metric, test_case, ignore_errors, skip_on_missing_params
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
    await measure_metrics_with_indicator(
        metrics=metrics,
        test_case=test_case,
        cached_test_case=None,
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
