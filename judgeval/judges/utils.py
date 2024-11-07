from typing import Optional, Union, Tuple

from judgeval.judges import judgevalBaseLLM, LiteLLMModel, TogetherModel
from judgeval.constants import TOGETHER_SUPPORTED_MODELS
from judgeval.litellm_model_names import LITE_LLM_MODEL_NAMES as LITELLM_SUPPORTED_MODELS


def initialize_model(
    model: Optional[Union[str, judgevalBaseLLM, LiteLLMModel, TogetherModel]] = None,
) -> Tuple[judgevalBaseLLM, bool]:
    """
    Returns a tuple of (initialized judgevalBaseLLM, using_native_model boolean)
    """
    # if isinstance(model, MixtureOfJudges):  # TODO: Implement MixtureOfJudges
    #     return model, True
    if isinstance(model, LiteLLMModel):
        return model, False
    # If model is a judgevalBaseLLM but not a GPTModel, we can not assume it is a native model
    if isinstance(model, judgevalBaseLLM):
        return model, False
    if model in LITELLM_SUPPORTED_MODELS:
        return LiteLLMModel(model=model), True
    if model in TOGETHER_SUPPORTED_MODELS:
        return TogetherModel(model=model), True
