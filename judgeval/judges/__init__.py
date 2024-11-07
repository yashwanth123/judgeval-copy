from pydantic import BaseModel
from judgeval.judges.base_judge import judgevalBaseLLM
from judgeval.judges.litellm_judge import LiteLLMModel
from judgeval.judges.together_judge import TogetherModel


class CustomModelParameters(BaseModel):
    model_name: str
    secret_key: str
    litellm_base_url: str
    