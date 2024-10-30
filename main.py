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

# TODO: I don't need both JudgevalRequestClass and CustomModelParameters
class JudgevalRequestClass(BaseModel):
    judge_model: str = "gpt-4o"
    # Either pass in a model string, or pass in dictionary
    custom_model: bool = False


class EvaluationRun(BaseModel):
    testcase: dict
    metric: str



app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/result/")
def result_evaluation(experiment_id: int):
    # fetch result with id
    fake_db = None 
    exp_result=  fake_db.get(experiment_id)
    return {"result": exp_result}


@app.post("/evaluate/")  # this post req gets hit with a json payload.
def evaluate(inp: EvaluationRun):
    """
    Ideal input payload:
    """
    # Params: testcase object and the metric (proprietary metric)
    metric_name = inp.metric 
    testcase = inp.testcase

    judgment_metric = load_metric(metric_name)
    result = run_execution(judgment_metric, testcase)

    return {
        "score": result.score
        # ...
    }


