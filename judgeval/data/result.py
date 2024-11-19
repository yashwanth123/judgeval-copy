from dataclasses import dataclass
from typing import List, Union, Optional

from judgeval.data import ScorerData, ProcessExample

@dataclass
class ScoringResult:
    """
    A ScoringResult contains the output of one or more scorers applied to a single example.

    Args:
        success (bool): Whether the evaluation was successful. 
                        This means that all scorers applied to this example returned a success.
        scorer_data (List[ScorerData]): The scorers data for the evaluated example
        input (Optional[str]): The input to the example
        actual_output (Optional[str]): The actual output of the example
        expected_output (Optional[str]): The expected output of the example
        context (Optional[List[str]]): The context of the example
        retrieval_context (Optional[List[str]]): The retrieval context of the example
        
    """
    # Fields for scoring outputs 
    success: bool  # used for unit testing
    scorers_data: Union[List[ScorerData], None]

    # Inputs from the original example
    input: Optional[str] = None
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert the ScoringResult instance to a dictionary, properly serializing scorer_data."""
        return {
            "success": self.success,
            "scorers_data": [scorer.model_dump() for scorer in self.scorers_data] if self.scorers_data else None,
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context
        }
    
    def __str__(self) -> str:
        return f"ScoringResult(\
            success={self.success}, \
            scorer_data={self.scorers_data}, \
            input={self.input}, \
            actual_output={self.actual_output}, \
            expected_output={self.expected_output}, \
            context={self.context}, \
            retrieval_context={self.retrieval_context})"


def generate_scoring_result(
    process_example: ProcessExample,
) -> ScoringResult:
    """
    Creates a final ScoringResult object for an evaluation run based on the results from a completed LLMApiTestCase.

    When an LLMTestCase is executed, it turns into an LLMApiTestCase and the progress of the evaluation run is tracked.
    At the end of the evaluation run, we create a TestResult object out of the completed LLMApiTestCase.
    """
    return ScoringResult(
        success=process_example.success,
        scorers_data=process_example.scorers_data,
        input=process_example.input,
        actual_output=process_example.actual_output,
        expected_output=process_example.expected_output,
        context=process_example.context,
        retrieval_context=process_example.retrieval_context,
    )
