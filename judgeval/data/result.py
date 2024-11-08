from dataclasses import dataclass 
from typing import List, Union, Optional

from judgeval.data.metric_data import MetricData
from judgeval.data.api_example import LLMApiTestCase

@dataclass
class TestResult:
    """Returned from run_test"""

    success: bool
    metrics_data: Union[List[MetricData], None]
    input: Optional[str] = None
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert the TestResult instance to a dictionary, properly serializing metrics_data."""
        return {
            "success": self.success,
            "metrics_data": [metric.model_dump() for metric in self.metrics_data] if self.metrics_data else None,
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context
        }


def create_test_result(
    api_test_case: LLMApiTestCase,
) -> TestResult:
    """
    Creates a final TestResult object for an evaluation run based on the results from a completed LLMApiTestCase.

    When an LLMTestCase is executed, it turns into an LLMApiTestCase and the progress of the evaluation run is tracked.
    At the end of the evaluation run, we create a TestResult object out of the completed LLMApiTestCase.
    """
    return TestResult(
        success=api_test_case.success,
        metrics_data=api_test_case.metrics_data,
        input=api_test_case.input,
        actual_output=api_test_case.actual_output,
        expected_output=api_test_case.expected_output,
        context=api_test_case.context,
        retrieval_context=api_test_case.retrieval_context,
    )
