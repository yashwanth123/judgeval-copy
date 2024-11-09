"""
Util functions for Scorer objects
"""

import inspect
from typing import List, Optional

from judgeval.scorers import CustomScorer


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
    scorer: CustomScorer,
    async_mode: Optional[bool] = None,
):
    if async_mode is None:
        run_async = scorer.async_mode
    else:
        run_async = async_mode

    return f"✨ Executing Judgment's [rgb(106,0,255)]{scorer.__name__} Scorer[/rgb(106,0,255)]! [rgb(55,65,81)](using {scorer.evaluation_model}, strict={scorer.strict_mode}, async_mode={run_async})...[/rgb(55,65,81)]"
