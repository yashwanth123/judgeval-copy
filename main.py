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
from typing import Generic, TypeVar, Optional, Any, Dict, Union
from enum import Enum

Input = TypeVar('Input')
Output = TypeVar('Output')

    
class TestCase(BaseModel, Generic[Input, Output]):
    """Base class for holding test case data"""
    # TODO: Add additional parameters based on backend server configuration
    input: Input
    output: Output
    name: Optional[str] = "unnamed_test"
    metadata: Optional[Dict[str, Any]] = {}

class BaseTestEvaluation(BaseModel):
    """Base class for test evaluations that don't require measure implementation"""
    # TODO: Add additional parameters based on backend server configuration
    test_type: str
    temperature: float

class CustomTestEvaluation(BaseModel, ABC):
    """Test evaluation that requires a measure implementation"""
    # TODO: Add additional parameters based on backend server configuration
    test_type: str
    temperature: float
    
    @abstractmethod
    def measure(self, input: Any, output: Any) -> float:
        """Method that must be implemented to measure test results"""
        pass

class EvaluationRun(BaseModel):
    """Stores test case and evaluation together for running"""
    test_case: TestCase
    test_evaluation: Union[BaseTestEvaluation, CustomTestEvaluation]

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
    test_case = evaluation_run.test_case
    test_evaluation = evaluation_run.test_evaluation
    
    PROPRIETARY_TESTS = ["test1", "test2", "test3"]
    
    if test_evaluation.test_type in PROPRIETARY_TESTS:
        
        response = requests.get(
            f"https://api.judgmentlabs.ai/evaluate/{test_evaluation.test_type}/",
            json=evaluation_run.model_dump()
        )
        return response.json()
        
    elif isinstance(test_evaluation, CustomTestEvaluation):
        result = test_evaluation.measure(test_case.input, test_case.output)
        return {"result": result}
    else:
        raise ValueError("Invalid test evaluation type")