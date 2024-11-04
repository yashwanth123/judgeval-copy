import asyncio
import os 
import logging
logging.basicConfig(level=logging.INFO)
from fastapi import FastAPI
from pydantic import BaseModel

# from some_module import load_metric, run_execution  # THESE COME FROM THE PROPRIETARY PACKAGE

from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests
from typing import Generic, TypeVar, Optional, Any, Dict
from enum import Enum

Input = TypeVar('Input')
Output = TypeVar('Output')

    
class TestCase(BaseModel, Generic[Input, Output]):
    """Base class for holding test case data"""
    input: Input
    output: Output
    name: Optional[str] = "unnamed_test"
    metadata: Optional[Dict[str, Any]] = {}

class TestEvaluation(BaseModel):
    """Stores information about the type of test evaluation to run"""
    test_type: str
    temperature: float

class EvaluationRun(BaseModel):
    """Stores test case and evaluation together for running"""
    test_case: TestCase
    test_evaluation: TestEvaluation

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Judgment Labs API",
        "description": "An API for evaluating and testing AI models",
        "endpoints": {
            "/": "This welcome page",
            "/evaluate": "POST endpoint for running model evaluations"
        },
        "documentation": "For full documentation, visit our docs at https://docs.judgmentlabs.ai"
    }

@app.post("/evaluate")
def runner(evaluation_run: EvaluationRun):
    """
    Endpoint to run evaluation using provided test case and evaluation
    """
    test_case = evaluation_run.test_case
    test_evaluation = evaluation_run.test_evaluation
    
    # Get the response and convert it to a dict
    response = requests.get(f"https://api.judgmentlabs.ai/evaluate/{test_evaluation.test_type}/", json=evaluation_run.model_dump())
    response_data = response.json()
    
    # Return a properly structured response
    return response_data