"""
This file contains utility functions used in repo scripts

For API calling, we support:
    - parallelized model calls on the same prompt 
    - batched model calls on different prompts

NOTE: any function beginning with 'a', e.g. 'afetch_together_api_response', is an asynchronous function
"""

import concurrent.futures
from typing import List, Mapping, Dict, Union
from langfuse.decorators import observe
import asyncio
import litellm
import pydantic
import pprint
import os
from dotenv import load_dotenv 

from judgeval import async_together_client, together_client
from judgeval.constants import *
from judgeval.litellm_model_names import LITE_LLM_MODEL_NAMES as LITELLM_SUPPORTED_MODELS
from judgeval.judges import CustomModelParameters

load_dotenv()

MAX_WORKER_THREADS = 10

def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding='utf-8') as file:
        return file.read()


# @observe
def fetch_together_api_response(model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None) -> str:
    """
    Fetches a single response from the Together API for a given model and messages.
    """
    if model not in TOGETHER_SUPPORTED_MODELS:
        raise ValueError(f"Model {model} is not in the list of supported TogetherAI models: {TOGETHER_SUPPORTED_MODELS}.")
    
    if response_format is not None:
        response = together_client.chat.completions.create(
            model=TOGETHER_SUPPORTED_MODELS.get(model),
            messages=messages,
            response_format=response_format
        )
    else:
        response = together_client.chat.completions.create(
            model=TOGETHER_SUPPORTED_MODELS.get(model),
            messages=messages,
        )
    return response.choices[0].message.content

# @observe
async def afetch_together_api_response(model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None) -> str:
    """
    ASYNCHRONOUSLY Fetches a single response from the Together API for a given model and messages.
    """
    if model not in TOGETHER_SUPPORTED_MODELS:
        raise ValueError(f"Model {model} is not in the list of supported TogetherAI models: {TOGETHER_SUPPORTED_MODELS}.")
    
    if response_format is not None:
        response = await async_together_client.chat.completions.create(
            model=TOGETHER_SUPPORTED_MODELS.get(model),
            messages=messages,
            response_format=response_format
        )
    else:
        response = await async_together_client.chat.completions.create(
            model=TOGETHER_SUPPORTED_MODELS.get(model),
            messages=messages,
        )
    return response.choices[0].message.content


@observe
def query_together_api_multiple_calls(models: List[str], messages: List[List[Mapping]], response_formats: List[pydantic.BaseModel] = None) -> List[str]:
    """
    Queries the Together API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: TogetherAI responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    num_workers = int(os.getenv('NUM_WORKER_THREADS', MAX_WORKER_THREADS))  
    # Initialize results to maintain ordered outputs
    out = [None] * len(messages)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all queries to together API with index, gets back the response content
        futures = {executor.submit(fetch_together_api_response, model, message, response_format): idx \
                   for idx, (model, message, response_format) in enumerate(zip(models, messages, response_formats))}
        
        # Collect results as they complete -- result is response content
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                out[idx] = future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
                out[idx] = None 
    return out


@observe
async def aquery_together_api_multiple_calls(models: List[str], messages: List[List[Mapping]], response_formats: List[pydantic.BaseModel] = None) -> List[str]:
    """
    Queries the Together API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: TogetherAI responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    # Initialize results to maintain ordered outputs
    out = [None] * len(messages)
    
    async def fetch_and_store(idx, model, message, response_format):
        try:
            out[idx] = await afetch_together_api_response(model, message, response_format)
        except Exception as e:
            print(f"An error occurred: {e}")
            out[idx] = None

    tasks = [
        fetch_and_store(idx, model, message, response_format)
        for idx, (model, message, response_format) in enumerate(zip(models, messages, response_formats))
    ]
    
    await asyncio.gather(*tasks)
    return out

@observe
def fetch_litellm_api_response(model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None) -> str:
    """
    Fetches a single response from the Litellm API for a given model and messages.
    """
    if model not in LITELLM_SUPPORTED_MODELS:
        raise ValueError(f"Model {model} is not in the list of supported Litellm models: {LITELLM_SUPPORTED_MODELS}.")
    
    if response_format is not None:
        response = litellm.completion(
            model=model,
            messages=messages,
            response_format=response_format
        )
    else:
        response = litellm.completion(
            model=model,
            messages=messages,  
        )
    return response.choices[0].message.content

def fetch_custom_litellm_api_response(custom_model_parameters: CustomModelParameters, messages: List[Mapping], response_format: pydantic.BaseModel = None) -> str:
    if response_format is not None:
        response = litellm.completion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
            response_format=response_format
        )
    else:
        response = litellm.completion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
        )
    return response.choices[0].message.content

@observe
async def afetch_litellm_api_response(model: str, messages: List[Mapping], response_format: pydantic.BaseModel = None) -> str:
    """
    ASYNCHRONOUSLY Fetches a single response from the Litellm API for a given model and messages.
    """
    # TODO: Uncomment code once we resolve Alma API key issues
    # if response_format is not None:
    #     response = await litellm.acompletion(
    #         model=MODEL_NAME,
    #         messages=messages,
    #         api_key=ALMA_FT_SECRET_KEY,
    #         base_url=ALMA_LITELLM_BASE_URL,
    #         response_format=response_format
    #     )
    # else:
    #     response = await litellm.acompletion(
    #         model=MODEL_NAME,
    #         messages=messages,
    #         api_key=ALMA_FT_SECRET_KEY,
    #         base_url=ALMA_LITELLM_BASE_URL,
    #     )
    # return response.choices[0].message.content
    if model not in LITELLM_SUPPORTED_MODELS:
        raise ValueError(f"Model {model} is not in the list of supported Litellm models: {LITELLM_SUPPORTED_MODELS}.")
    
    if response_format is not None:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            response_format=response_format
        )
    else:
        response = await litellm.acompletion(
            model=model,
            messages=messages,  
        )
    return response.choices[0].message.content

async def afetch_custom_litellm_api_response(custom_model_parameters: CustomModelParameters, messages: List[Mapping], response_format: pydantic.BaseModel = None) -> str:
    """
    ASYNCHRONOUSLY Fetches a single response from the Litellm API for a given model and messages.
    """
    if response_format is not None:
        response = await litellm.acompletion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
            response_format=response_format
        )
    else:
        response = await litellm.acompletion(
            model=custom_model_parameters.model_name,
            messages=messages,
            api_key=custom_model_parameters.secret_key,
            base_url=custom_model_parameters.litellm_base_url,
        )
    return response.choices[0].message.content

@observe
def query_litellm_api_multiple_calls(models: List[str], messages: List[Mapping], response_formats: List[pydantic.BaseModel] = None) -> List[str]:
    """
    Queries the Litellm API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[Mapping]): List of messages to query
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: Litellm responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    num_workers = int(os.getenv('NUM_WORKER_THREADS', MAX_WORKER_THREADS))  
    # Initialize results to maintain ordered outputs
    out = [None] * len(messages)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all queries to Litellm API with index, gets back the response content
        futures = {executor.submit(fetch_litellm_api_response, model, message, response_format): idx \
                    for idx, (model, message, response_format) in enumerate(zip(models, messages, response_formats))}
        
        # Collect results as they complete -- result is response content
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                out[idx] = future.result()
            except Exception as e:
                print(f"An error occurred: {e}")
                out[idx] = None 
    return out


@observe
async def aquery_litellm_api_multiple_calls(models: List[str], messages: List[Mapping], response_formats: List[pydantic.BaseModel] = None) -> List[str]:
    """
    Queries the Litellm API for multiple calls in parallel

    Args:
        models (List[str]): List of models to query
        messages (List[Mapping]): List of messages to query
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.

    Returns:
        List[str]: Litellm responses for each model and message pair in order. Any exceptions in the thread call result in a None.
    """
    # Initialize results to maintain ordered outputs
    out = [None] * len(messages)
    
    async def fetch_and_store(idx, model, message, response_format):
        try:
            out[idx] = await afetch_litellm_api_response(model, message, response_format)
        except Exception as e:
            print(f"An error occurred: {e}")
            out[idx] = None

    tasks = [
        fetch_and_store(idx, model, message, response_format)
        for idx, (model, message, response_format) in enumerate(zip(models, messages, response_formats))
    ]
    
    await asyncio.gather(*tasks)
    return out


@observe
def get_chat_completion(model_type: str, 
                        messages : Union[List[Mapping], List[List[Mapping]]], 
                        response_format: pydantic.BaseModel = None, 
                        batched: bool = False
                        ) -> Union[str, List[str]]:
    """
    Generates chat completions using a single model and potentially several messages. Supports closed-source and OSS models.

    Parameters:
        - model_type (str): The type of model to use for generating completions.
        - messages (Union[List[Mapping], List[List[Mapping]]]): The messages to be used for generating completions. 
            If batched is True, this should be a list of lists of mappings.
        - response_format (pydantic.BaseModel, optional): The format of the response. Defaults to None.
        - batched (bool, optional): Whether to process messages in batch mode. Defaults to False.
    Returns:
        - str: The generated chat completion(s). If batched is True, returns a list of strings.
    Raises:
        - ValueError: If requested model is not supported by Litellm or TogetherAI.
    """
    if batched and model_type in TOGETHER_SUPPORTED_MODELS:
        return query_together_api_multiple_calls(models=[model_type] * len(messages), 
                                                 messages=messages, 
                                                 response_formats=[response_format] * len(messages))
    elif batched and model_type in LITELLM_SUPPORTED_MODELS:
        return query_litellm_api_multiple_calls(models=[model_type] * len(messages), 
                                                messages=messages, 
                                                response_format=response_format)
    elif not batched and model_type in TOGETHER_SUPPORTED_MODELS:
        return fetch_together_api_response(model=model_type, 
                                           messages=messages, 
                                           response_format=response_format)
    elif not batched and model_type in LITELLM_SUPPORTED_MODELS:
        return fetch_litellm_api_response(model=model_type, 
                                          messages=messages, 
                                          response_format=response_format)
    
    raise ValueError(f"Model {model_type} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again.")


@observe
async def aget_chat_completion(model_type: str, 
                               messages : Union[List[Mapping], List[List[Mapping]]], 
                               response_format: pydantic.BaseModel = None, 
                               batched: bool = False
                               ) -> Union[str, List[str]]:
    """
    ASYNCHRONOUSLY generates chat completions using a single model and potentially several messages. Supports closed-source and OSS models.

    Parameters:
        - model_type (str): The type of model to use for generating completions.
        - messages (Union[List[Mapping], List[List[Mapping]]]): The messages to be used for generating completions. 
            If batched is True, this should be a list of lists of mappings.
        - response_format (pydantic.BaseModel, optional): The format of the response. Defaults to None.
        - batched (bool, optional): Whether to process messages in batch mode. Defaults to False.
    Returns:
        - str: The generated chat completion(s). If batched is True, returns a list of strings.
    Raises:
        - ValueError: If requested model is not supported by Litellm or TogetherAI.
    """
    if batched and model_type in TOGETHER_SUPPORTED_MODELS:
        return await aquery_together_api_multiple_calls(models=[model_type] * len(messages), 
                                                        messages=messages, 
                                                        response_formats=[response_format] * len(messages))
    elif batched and model_type in LITELLM_SUPPORTED_MODELS:
        return await aquery_litellm_api_multiple_calls(models=[model_type] * len(messages), 
                                                       messages=messages, 
                                                       response_formats=[response_format] * len(messages))
    elif not batched and model_type in TOGETHER_SUPPORTED_MODELS:
        return await afetch_together_api_response(model=model_type, 
                                                  messages=messages, 
                                                  response_format=response_format)
    elif not batched and model_type in LITELLM_SUPPORTED_MODELS:
        return await afetch_litellm_api_response(model=model_type, 
                                                 messages=messages, 
                                                 response_format=response_format)
    
    raise ValueError(f"Model {model_type} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again.")


@observe
def get_completion_multiple_models(models: List[str], messages: List[List[Mapping]], response_formats: List[pydantic.BaseModel] = None) -> List[str]:
    """
    Retrieves completions for a single prompt from multiple models in parallel. Supports closed-source and OSS models.

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.
    
    Returns:
        List[str]: List of completions from the models in the order of the input models
    Raises:
        ValueError: If a model is not supported by Litellm or Together
    """
    if len(models) != len(messages):
        raise ValueError(f"Number of models and messages must be the same: {len(models)} != {len(messages)}")
    if response_formats is None:
        response_formats = [None] * len(models)
    # Partition the model requests into TogetherAI and Litellm models, but keep the ordering saved
    together_calls, litellm_calls = {}, {}  # index -> model, message, response_format
    together_responses, litellm_responses = [], []
    for idx, (model, message, r_format) in enumerate(zip(models, messages, response_formats)):
        if model in TOGETHER_SUPPORTED_MODELS:
            together_calls[idx] = (model, message, r_format)
        elif model in LITELLM_SUPPORTED_MODELS:
            litellm_calls[idx] = (model, message, r_format)
        else:
            raise ValueError(f"Model {model} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again.")
    
    # Get the responses from the TogetherAI models
    # List of responses from the TogetherAI models in order of the together_calls dict
    if together_calls:
        together_responses = query_together_api_multiple_calls(models=[model for model, _, _ in together_calls.values()], 
                                                           messages=[message for _, message, _ in together_calls.values()], 
                                                           response_formats=[format for _, _, format in together_calls.values()])  
    
    # Get the responses from the Litellm models
    if litellm_calls:
        litellm_responses = query_litellm_api_multiple_calls(models=[model for model, _, _ in litellm_calls.values()],
                                                        messages=[message for _, message, _ in litellm_calls.values()],
                                                        response_formats=[format for _, _, format in litellm_calls.values()])

    # Merge the responses in the order of the original models
    out = [None] * len(models)
    for idx, (model, message, r_format) in together_calls.items():
        out[idx] = together_responses.pop(0)
    for idx, (model, message, r_format) in litellm_calls.items():
        out[idx] = litellm_responses.pop(0)
    return out 


@observe
async def aget_completion_multiple_models(models: List[str], messages: List[List[Mapping]], response_formats: List[pydantic.BaseModel] = None) -> List[str]:
    """
    ASYNCHRONOUSLY retrieves completions for a single prompt from multiple models in parallel. Supports closed-source and OSS models.

    Args:
        models (List[str]): List of models to query
        messages (List[List[Mapping]]): List of messages to query. Each inner object corresponds to a single prompt.
        response_formats (List[pydantic.BaseModel], optional): A list of the format of the response if JSON forcing. Defaults to None.
    
    Returns:
        List[str]: List of completions from the models in the order of the input models
    Raises:
        ValueError: If a model is not supported by Litellm or Together
    """
    if len(models) != len(messages):
        raise ValueError(f"Number of models and messages must be the same: {len(models)} != {len(messages)}")
    if response_formats is None:
        response_formats = [None] * len(models)
    
    # Partition the model requests into TogetherAI and Litellm models, but keep the ordering saved
    together_calls, litellm_calls = {}, {}  # index -> model, message, response_format
    together_responses, litellm_responses = [], []
    for idx, (model, message, r_format) in enumerate(zip(models, messages, response_formats)):
        if model in TOGETHER_SUPPORTED_MODELS:
            together_calls[idx] = (model, message, r_format)
        elif model in LITELLM_SUPPORTED_MODELS:
            litellm_calls[idx] = (model, message, r_format)
        else:
            raise ValueError(f"Model {model} is not supported by Litellm or TogetherAI for chat completions. Please check the model name and try again.")
    
    # Get the responses from the TogetherAI models
    # List of responses from the TogetherAI models in order of the together_calls dict
    if together_calls:
        together_responses = await aquery_together_api_multiple_calls(
            models=[model for model, _, _ in together_calls.values()], 
            messages=[message for _, message, _ in together_calls.values()], 
            response_formats=[format for _, _, format in together_calls.values()]
        )
    
    # Get the responses from the Litellm models
    if litellm_calls:
        litellm_responses = await aquery_litellm_api_multiple_calls(
            models=[model for model, _, _ in litellm_calls.values()],
            messages=[message for _, message, _ in litellm_calls.values()],
            response_formats=[format for _, _, format in litellm_calls.values()]
        )

    # Merge the responses in the order of the original models
    out = [None] * len(models)
    for idx, (model, message, r_format) in together_calls.items():
        out[idx] = together_responses.pop(0)
    for idx, (model, message, r_format) in litellm_calls.items():
        out[idx] = litellm_responses.pop(0)
    return out


if __name__ == "__main__":
    
    # Batched
    pprint.pprint(get_chat_completion(
        model_type="LLAMA3_405B_INSTRUCT_TURBO",
        messages=[
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Japan?"},
            ]
        ],
        batched=True
    ))

    # Non batched
    pprint.pprint(get_chat_completion(
        model_type="LLAMA3_8B_INSTRUCT_TURBO",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        batched=False
    ))

    # Batched single completion to multiple models
    pprint.pprint(get_completion_multiple_models(
        models=[
            "LLAMA3_70B_INSTRUCT_TURBO", "LLAMA3_405B_INSTRUCT_TURBO", "gpt-4o-mini"
        ],
        messages=[
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of China?"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of Japan?"},
            ]
        ]
    ))
