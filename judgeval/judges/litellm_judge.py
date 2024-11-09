import pydantic
from typing import List, Union, Mapping
from judgeval import *
from judgeval.judges.base_judge import judgevalJudge
from judgeval.common.utils import afetch_litellm_api_response, fetch_litellm_api_response

BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
]  # for string inputs, we need to add the user query to a base conversation, since LiteLLM only accepts a list of dictionaries as a chat history


class LiteLLMJudge(judgevalJudge):
    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        self.model = model
        self.kwargs = kwargs
        super().__init__(model_name=model)

    def generate(self, input: Union[str, List[Mapping[str, str]]], schema: pydantic.BaseModel = None) -> str:
        if type(input) == str:
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            return fetch_litellm_api_response(model=self.model, messages=convo, response_format=schema), 0  # TODO: fix the cost. Currently set to 0.
        elif type(input) == list:
            return fetch_litellm_api_response(model=self.model, messages=input, response_format=schema), 0 
        else:
            raise TypeError(f"Input must be a string or a list of dictionaries. Input type of: {type(input)}")

    async def a_generate(self, input: Union[str, List[Mapping[str, str]]], schema: pydantic.BaseModel = None) -> str:
        if type(input) == str:
            convo = BASE_CONVERSATION + [{"role": "user", "content": input}]
            response = await afetch_litellm_api_response(model=self.model, messages=convo, response_format=schema)
            return response, 0
        elif type(input) == list:
            response = await afetch_litellm_api_response(model=self.model, messages=input, response_format=schema)
            return response, 0 # TODO: fix the cost!
        else:
            raise TypeError(f"Input must be a string or a list of dictionaries. Input type of: {type(input)}")
    
    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return self.model
