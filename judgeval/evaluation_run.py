import requests
import pprint
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, field_validator

from judgeval.data import Example
from judgeval.scorers import CustomScorer, JudgmentScorer
from judgeval.scorers.score import *
from judgeval.constants import ACCEPTABLE_MODELS
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.playground import CustomFaithfulnessMetric
from judgeval.judges import TogetherJudge
import litellm


class EvaluationRun(BaseModel):
    """
    Stores example and evaluation together for running
    
    Args: 
        name (str): A name for this evaluation run
        examples (List[Example]): The examples to evaluate
        scorers (List[Union[JudgmentScorer, CustomScorer]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
        aggregator (Optional[str]): The aggregator to use for evaluation if using Mixture of Judges
        metadata (Optional[Dict[str, Any]]): Additional metadata to include for this evaluation run, e.g. comments, dataset name, purpose, etc.
    """
    name: Optional[str] = ""
    examples: List[Example]
    scorers: List[Union[JudgmentScorer, CustomScorer]]
    model: Union[str, List[str]]
    aggregator: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # API Key will be "" until user calls client.run_eval(), then API Key will be set
    judgment_api_key: Optional[str] = ""
    # The user will specify whether they want log_results when they call run_eval
    log_results: bool = False
    
    @field_validator('examples')
    def validate_examples(cls, v):
        if not v:
            raise ValueError("Examples cannot be empty.")
        for ex in v:
            if not isinstance(ex, Example):
                raise ValueError(f"Invalid type for Example: {type(ex)}")
        return v

    @field_validator('scorers')
    def validate_scorers(cls, v):
        if not v:
            raise ValueError("Scorers cannot be empty.")
        for s in v:
            if not isinstance(s, JudgmentScorer) and not isinstance(s, CustomScorer):
                raise ValueError(f"Invalid type for Scorer: {type(s)}")
        return v

    @field_validator('model')
    def validate_model(cls, v):
        if not v:
            raise ValueError("Model cannot be empty.")
        if not isinstance(v, str) and not isinstance(v, list):
            raise ValueError("Model must be a string or a list of strings.")
        if isinstance(v, str) and v not in ACCEPTABLE_MODELS:
            raise ValueError(f"Model name {v} not recognized.")
        if isinstance(v, list):
            for m in v:
                if m not in ACCEPTABLE_MODELS:
                    raise ValueError(f"Model name {m} not recognized.")
        return v

    @field_validator('aggregator', mode='before')
    def validate_aggregator(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("Aggregator must be a string if provided.")
        if v is not None and v not in ACCEPTABLE_MODELS:
            raise ValueError(f"Model name {v} not recognized.")
        return v

    class Config:
        arbitrary_types_allowed = True