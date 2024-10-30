from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests
from typing import Generic, TypeVar, Optional, Any, Dict
from enum import Enum

Input = TypeVar('Input')
Output = TypeVar('Output')

class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"

@dataclass
class TestResult:
    """Stores the result of a test evaluation"""
    status: TestStatus
    score: float
    message: str
    metadata: Dict[str, Any] = None

class TestCase(Generic[Input, Output], ABC):
    """Abstract base class for holding test case data"""
    
    @abstractmethod
    def __init__(
        self,
        input: Input,
        expected: Output,
        name: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.input = input
        self.expected = expected
        self.name = name or "unnamed_test"
        self.metadata = metadata or {}
        self.output: Optional[Output] = None
    
    def __str__(self):
        return f"TestCase(name={self.name}, input={self.input}, expected={self.expected})"
    
    def __dict__(self) -> Dict[str, Any]:
        """
        Serialize the test case into a dictionary format.
        
        Returns:
            Dictionary containing the test case data
        """
        return {
            "name": self.name,
            "input": self.input,
            "expected": self.expected,
            "output": self.output,
            "metadata": self.metadata
        }

class TestEvaluation(Generic[Input, Output], ABC):
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
        pass

class TestRunner:
    """Handles running tests and calling APIs"""
    
    def __init__(self, local: bool = False):
        self.local = local
    
    def run(self, test_case: TestCase, evaluation: TestEvaluation) -> TestResult:
        """
        Run a single test case with the given evaluation
        
        Args:
            test_case: The test case to run
            evaluation: The evaluation to assess the result
        """
        try:
            # Call API and get response
            response = self._call_api(test_case.input)
            
            # Store the output
            test_case.output = response
            
            # Evaluate the result using provided evaluation
            result = evaluation.evaluate(test_case)
            
            return result
            
        except Exception as e:
            return TestResult(
                status=TestStatus.ERROR,
                score=0.0,
                message=f"Error running test: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _call_api(self, input: Any) -> Any:
        """Make the actual API call"""
        data = {
            "testcase": TestCase.__dict__,
            "testevaluation": TestEvaluation.__dict__,
        }

        if self.local:
            response = requests.post("http://localhost:8000/evaluation", json=data)
        else:
            response = requests.post("http://my-fargate-alb-334054785.us-west-1.elb.amazonaws.com/evaluation", json=data)

        if response.status_code == 200:
            return response.json()
        response.raise_for_status()

# Example implementation for text comparison
class TextComparisonEvaluation(TestEvaluation[str, str]):
    """A evaluation class for comparing text outputs"""
    
    def evaluate(self, test_case: TestCase[str, str]) -> TestResult:
        if not test_case.output:
            return TestResult(
                status=TestStatus.ERROR,
                score=0.0,
                message="No output to evaluate",
                metadata={"error": "No output available"}
            )
        
        # Simple exact match for now
        matches = test_case.output.strip() == test_case.expected.strip()
        
        return TestResult(
            status=TestStatus.PASS if matches else TestStatus.FAIL,
            score=1.0 if matches else 0.0,
            message="Exact match" if matches else "Output does not match expected",
            metadata={
                "output": test_case.output,
                "expected": test_case.expected
            }
        )
