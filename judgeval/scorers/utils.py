"""
Util functions for Scorer objects
"""

import inspect
from typing import List, Optional

from judgeval.scorers.custom_scorer import CustomScorer


def clone_scorers(
    scorers: List[
        CustomScorer
    ]
) -> List[CustomScorer]:
    """
    Creates duplicates of the scorers passed as argument.
    """
    cloned_scorers = []
    for s in scorers:
        scorer_class = type(s)
        args = vars(s)

        signature = inspect.signature(scorer_class.__init__)
        valid_params = signature.parameters.keys()
        valid_args = {key: args[key] for key in valid_params if key in args}

        cloned_scorers.append(scorer_class(**valid_args))
    return cloned_scorers

def format_metric_description(
    metric: CustomScorer,
    async_mode: Optional[bool] = None,
):
    if async_mode is None:
        run_async = metric.async_mode
    else:
        run_async = async_mode

    return f"✨ You're running judgeval's latest [rgb(106,0,255)]{metric.__name__} Metric[/rgb(106,0,255)]! [rgb(55,65,81)](using {metric.evaluation_model}, strict={metric.strict_mode}, async_mode={run_async})...[/rgb(55,65,81)]"
