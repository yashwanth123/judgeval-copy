"""
Implementation of using TogetherAI inference for judges.
"""

from pydantic import BaseModel
from typing import List, Union, Mapping
from judgeval.judges import judgevalJudge
from judgeval.common.utils import fetch_together_api_response, afetch_together_api_response

BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
]

class TogetherJudge(judgevalJudge):
    def __init__(self, model: str = "QWEN", **kwargs):
        self.model = model
        self.kwargs = kwargs
        super().__init__(model_name=model)

    # TODO: Fix cost for generate and a_generate
    def generate(self, input: Union[str, List[Mapping[str, str]]], schema: BaseModel = None) -> str:
        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            return fetch_together_api_response(self.model, convo, response_format=schema), 0
        elif isinstance(input, list):
            return fetch_together_api_response(self.model, convo, response_format=schema), 0
        else:
            raise TypeError("Input must be a string or a list of dictionaries.")

    async def a_generate(self, input: Union[str, List[dict]], schema: BaseModel = None) -> str:
        if isinstance(input, str):
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            res = await afetch_together_api_response(self.model, convo, response_format=schema)
            return res, 0
        elif isinstance(input, list):
            res = await afetch_together_api_response(self.model, input, response_format=schema)
            return res, 0
        else:
            raise TypeError("Input must be a string or a list of dictionaries.")

    def load_model(self) -> str:
        return self.model

    def get_model_name(self) -> str:
        return self.model
    