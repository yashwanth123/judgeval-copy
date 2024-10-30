"""
Example server setup.

NOTE this server would be set up inside of the proprietary (private) repo and never seen by public.

You would call this server from the public repo by executing API requests to the routes inside of this server.

e.g. (from the public repo)
requests.post('{private_endpoint_url}/evaluate/', json={"testcase": {"input": "input", "expected": "expected"}, 
                                     "metric": "metric"})
"""

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

class TestStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL" 
    ERROR = "ERROR"

class TestResult(BaseModel):
    """Stores the result of a test evaluation"""
    status: TestStatus
    score: float
    message: str
    metadata: Optional[Dict[str, Any]] = None

class TestCase(BaseModel, Generic[Input, Output]):
    """Base class for holding test case data"""
    input: Input
    expected: Output
    name: Optional[str] = "unnamed_test"
    metadata: Optional[Dict[str, Any]] = {}
    output: Optional[Output] = None

class TestEvaluation(BaseModel, Generic[Input, Output], ABC):
    """Base class for implementing test evaluations"""
    
    @abstractmethod
    def evaluate(self, test_case: TestCase[Input, Output]) -> TestResult:
        """
        Evaluate the quality of the test case output against its expected result.
        Must be implemented by subclasses.
        
        Args:
            test_case: The test case containing input, output and expected values
            
        Returns:
            TestResult comparing the output against the expected
        """
        raise NotImplementedError

class EvaluationRun(BaseModel):
    """Stores test case and evaluation together for running"""
    test_case: TestCase
    evaluation: TestEvaluation

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/result/")
def result_evaluation(experiment_id: int):
    # fetch result with id
    fake_db = None 
    exp_result = fake_db.get(experiment_id)
    return {"result": exp_result}

@app.post("/evaluate/")  # this post req gets hit with a json payload.
def evaluate(evaluation_run: EvaluationRun):
    """
    Endpoint to run evaluation using provided test case and evaluation
    """
    test_case = evaluation_run.test_case
    evaluation = evaluation_run.evaluation
    
    result = evaluation.evaluate(test_case)

    return {
        "status": result.status.value,
        "score": result.score,
        "message": result.message,
        "metadata": result.metadata
    }