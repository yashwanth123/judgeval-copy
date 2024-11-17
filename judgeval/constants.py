"""
Constant variables used throughout source code
"""

from enum import Enum
import litellm

class JudgmentMetric(Enum):  
    """
    Collection of proprietary metrics implemented by Judgment.

    These are ready-made evaluation scorers/metrics that can be used to evaluate
    Examples via the Judgment API.
    """
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    HALLUCINATION = "hallucination"
    SUMMARIZATION = "summarization"
    GEVAL = "geval"
    CONTEXTUAL_RECALL = "contextual_recall"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
    CONTEXTUAL_PRECISION = "contextual_precision"
    KNOWLEDGE_RETENTION = "knowledge_retention"
    TOOL_CORRECTNESS = "tool_correctness"
    CUSTOM = "custom"

ROOT_API = "http://127.0.0.1:8000"
# ROOT_API = "https://api.judgmentlabs.ai"  # TODO replace this with the actual API root
JUDGMENT_EVAL_API_URL = f"{ROOT_API}/evaluate/"
JUDGMENT_DATASETS_PUSH_API_URL = f"{ROOT_API}/datasets/push/"
JUDGMENT_DATASETS_PULL_API_URL = f"{ROOT_API}/datasets/pull/"
JUDGMENT_EVAL_LOG_API_URL = f"{ROOT_API}/log_custom_eval_results/"
JUDGMENT_EVAL_FETCH_API_URL = f"{ROOT_API}/fetch_eval_results/"
TOGETHER_SUPPORTED_MODELS = {
    "QWEN": "Qwen/Qwen2-72B-Instruct",
    "LLAMA3_70B_INSTRUCT_TURBO": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "LLAMA3_405B_INSTRUCT_TURBO": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "LLAMA3_8B_INSTRUCT_TURBO": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "MISTRAL_8x22B_INSTRUCT": "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "MISTRAL_8x7B_INSTRUCT": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}

ACCEPTABLE_MODELS = set(litellm.model_list) | set(TOGETHER_SUPPORTED_MODELS.keys())

MAX_WORKER_THREADS = 10
