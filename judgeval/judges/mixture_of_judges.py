"""
Implementation for Mixture of Judges model through Judgeval
"""
from judgeval import *
import pydantic
from typing import List, Union, Tuple, Mapping
from judgeval.judges import judgevalJudge
from judgeval.common.utils import get_completion_multiple_models, get_chat_completion, aget_completion_multiple_models, aget_chat_completion

import litellm

# Add at the start of the file or before any LiteLLM calls
litellm.set_verbose = True

def build_dynamic_mixture_prompt(judge_responses: List[str], custom_prompt: str = None) -> str:
    # Format the judge responses first
    formatted_responses = "\n".join([f"# Judge {i + 1}'s response: #\n{response}" for i, response in enumerate(judge_responses)])


    # Inject the judge responses into the mixture prompt
    mixture_prompt = langfuse.get_prompt('MIX_RESPONSES')
    compiled_mixture_prompt = mixture_prompt.compile(
        judge_responses=formatted_responses,
    )

    # The prompt is already formatted using f-strings, no need for compile
    return compiled_mixture_prompt

BASE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
]  # for string inputs, we need to add the user query to a base conversation, since LiteLLM only accepts a list of dictionaries as a chat history
class MixtureOfJudges(judgevalJudge):
    def __init__(self, 
                 models: List[str] = ['QWEN', 'LLAMA3_70B_INSTRUCT_TURBO', 'MISTRAL_8x22B_INSTRUCT'],
                 aggregator: str = 'gpt-4o', 
                 **kwargs):
        self.models = models
        self.aggregator = aggregator
        self.kwargs = kwargs
        super().__init__(model_name=models)

    def generate(self, input: Union[str, List[Mapping[str, str]]], schema: pydantic.BaseModel = None, **kwargs) -> str:
        if type(input) == str:
            input = BASE_CONVERSATION + [{"role": "user", "content": input}]
        
        try:
            responses = get_completion_multiple_models(
                models=self.models,
                messages=[input] * len(self.models),  # repeat the same input for all judges since we query them in parallel
            )
        except Exception as e:
            raise

        compiled_mixture_prompt = build_dynamic_mixture_prompt(responses, self.kwargs.get('mixture_prompt'))
        
        try:
            mixed_response = get_chat_completion(
                model_type=self.aggregator,
                messages=BASE_CONVERSATION + [{"role": "user", "content": compiled_mixture_prompt}],
            )
        except Exception as e:
            raise
            
        return mixed_response, 0

    async def a_generate(self, input: Union[str, List[Mapping[str, str]]], schema: pydantic.BaseModel = None, **kwargs) -> str:
        if type(input) == str:
            input = BASE_CONVERSATION + [{"role": "user", "content": input}]
        
        try:
            responses = await aget_completion_multiple_models(
                models=self.models,
                messages=[input] * len(self.models),  # repeat the same input for all judges since we query them in parallel
            )
        except Exception as e:
            raise

        compiled_mixture_prompt = build_dynamic_mixture_prompt(responses, self.kwargs.get('mixture_prompt'))
        
        try:
            mixed_response = await aget_chat_completion(
                model_type=self.aggregator,
                messages=BASE_CONVERSATION + [{"role": "user", "content": compiled_mixture_prompt}],
            )

            print(mixed_response)
        except Exception as e:
            raise
            
        return mixed_response, 0
    
    def load_model(self):
        return self.models

    def get_model_name(self) -> List[str]:
        return self.models

