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
