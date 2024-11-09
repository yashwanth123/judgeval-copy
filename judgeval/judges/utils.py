"""
This module contains utility functions for judge models.
"""

from typing import Optional, Union, Tuple

from judgeval.judges import judgevalJudge, LiteLLMJudge, TogetherJudge
from judgeval.constants import TOGETHER_SUPPORTED_MODELS
from judgeval.litellm_model_names import LITE_LLM_MODEL_NAMES as LITELLM_SUPPORTED_MODELS


def create_judge(
    model: Optional[Union[str, judgevalJudge, LiteLLMJudge, TogetherJudge]] = None,
) -> Tuple[judgevalJudge, bool]:
    """
    Returns a tuple of (initialized judgevalBaseLLM, using_native_model boolean)

    If no model is provided, uses GPT4o as the default judge.
    """
    # if isinstance(model, MixtureOfJudges):  # TODO: Implement MixtureOfJudges
    #     return model, True
    if isinstance(model, LiteLLMJudge):
        return model, False
    # If model is a judgevalBaseLLM but not a GPTModel, we can not assume it is a native model
    if isinstance(model, judgevalJudge):
        return model, False
    if model in LITELLM_SUPPORTED_MODELS:
        return LiteLLMJudge(model=model), True
    if model in TOGETHER_SUPPORTED_MODELS:
        return TogetherJudge(model=model), True
    if model is None:
        return LiteLLMJudge(model="gpt-4o"), True
