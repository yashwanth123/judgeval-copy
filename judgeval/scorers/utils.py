"""
Util functions for Scorer objects

TODO add logging
"""

import asyncio
import nest_asyncio
import inspect
import json
import sys
import re
from contextlib import contextmanager
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from typing import List, Optional, Any

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


def scorer_console_msg(
    scorer: CustomScorer,
    async_mode: Optional[bool] = None,
):
    """
    Renders a message to be displayed to console when a scorer is being executed.
    """
    if async_mode is None:
        run_async = scorer.async_mode
    else:
        run_async = async_mode

    return f"🔨 Executing Judgment's [rgb(106,0,255)]{scorer.__name__} Scorer[/rgb(106,0,255)]! [rgb(55,65,81)](using {scorer.evaluation_model}, async_mode={run_async})...[/rgb(55,65,81)]"
