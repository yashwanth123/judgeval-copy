from dataclasses import dataclass, fields
from typing import List, Union, Optional

from judgeval.data import ScorerData, processExample

@dataclass
class ScoringResult:
    """
    A ScoringResult contains the output of one or more scorers applied to a single example.

    Args:
        success (bool): Whether the evaluation was successful. 
                        This means that all scorers applied to this example returned a success.
        metrics_data (List[MetricData]): The metrics data for the evaluated example
        input (Optional[str]): The input to the example
        actual_output (Optional[str]): The actual output of the example
        expected_output (Optional[str]): The expected output of the example
        context (Optional[List[str]]): The context of the example
        retrieval_context (Optional[List[str]]): The retrieval context of the example
    """
    ### Fields for scoring outputs ### 
    success: bool  # used for unit testing
    metrics_data: Union[List[ScorerData], None]

    ### Inputs from the original example ### 
    input: Optional[str] = None
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert the ScoringResult instance to a dictionary, properly serializing metrics_data."""
        return {
            "success": self.success,
            "metrics_data": [metric.model_dump() for metric in self.metrics_data] if self.metrics_data else None,
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context
        }
    
    def __str__(self) -> str:
        return f"ScoringResult(\
            success={self.success}, \
            metrics_data={self.metrics_data}, \
            input={self.input}, \
            actual_output={self.actual_output}, \
            expected_output={self.expected_output}, \
            context={self.context}, \
            retrieval_context={self.retrieval_context})"


def generate_scoring_result(
    api_test_case: processExample,
) -> ScoringResult:
    """
    Creates a final ScoringResult object for an evaluation run based on the results from a completed LLMApiTestCase.

    When an LLMTestCase is executed, it turns into an LLMApiTestCase and the progress of the evaluation run is tracked.
    At the end of the evaluation run, we create a TestResult object out of the completed LLMApiTestCase.
    """
    return ScoringResult(
        success=api_test_case.success,
        metrics_data=api_test_case.metrics_data,
        input=api_test_case.input,
        actual_output=api_test_case.actual_output,
        expected_output=api_test_case.expected_output,
        context=api_test_case.context,
        retrieval_context=api_test_case.retrieval_context,
    )
