"""
Constant variables used throughout source code
"""

from enum import Enum


class JudgmentMetric(Enum):  
    """
    Collection of proprietary metrics implemented by Judgment
    """
    FAITHFULNESS = "faithfulness"
    # TODO add the rest of the proprietary metrics here

ROOT_API = "http://127.0.0.1:8000"
# ROOT_API = "https://api.judgmentlabs.ai"  # TODO replace this with the actual API root
JUDGMENT_EVAL_API_URL = f"{ROOT_API}/evaluate/"

TOGETHER_SUPPORTED_MODELS = {
    "QWEN": "Qwen/Qwen2-72B-Instruct",
    "LLAMA3_70B_INSTRUCT_TURBO": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "LLAMA3_405B_INSTRUCT_TURBO": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "LLAMA3_8B_INSTRUCT_TURBO": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "MISTRAL_8x22B_INSTRUCT": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "MISTRAL_8x7B_INSTRUCT": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


