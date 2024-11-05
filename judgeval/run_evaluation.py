import requests

from judgeval.core_classes import EvaluationRun, CustomTestEvaluation


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