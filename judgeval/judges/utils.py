"""
This module contains utility functions for judge models.
"""
import litellm
from typing import Optional, Union, Tuple

from judgeval.common.exceptions import InvalidJudgeModelError
from judgeval.judges import judgevalJudge, LiteLLMJudge, TogetherJudge, MixtureOfJudges
from judgeval.constants import TOGETHER_SUPPORTED_MODELS

LITELLM_SUPPORTED_MODELS = set(litellm.model_list)

def create_judge(
    model: Optional[Union[str, judgevalJudge, LiteLLMJudge, TogetherJudge]] = None,
) -> Tuple[judgevalJudge, bool]:
    """
    Returns a tuple of (initialized judgevalBaseLLM, using_native_model boolean)

    If no model is provided, uses GPT4o as the default judge.
    """
    if model is None:  # default option
        return LiteLLMJudge(model="gpt-4o"), True
    # If model is already a valid judge type, return it and mark native
    if any(isinstance(model, judge_type) for judge_type in [judgevalJudge, LiteLLMJudge, TogetherJudge]):
        return model, True 
    # If model is a string, check that it corresponds to a valid model
    if model in LITELLM_SUPPORTED_MODELS:
        return LiteLLMJudge(model=model), True
    if model in TOGETHER_SUPPORTED_MODELS:
        return TogetherJudge(model=model), True
    else:
        raise InvalidJudgeModelError(f"Invalid judge model chosen: {model}")
