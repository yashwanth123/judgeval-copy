from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    """Base class for creating test cases"""
    
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
    
    @abstractmethod
    def measure(self) -> TestResult:
        """
        Measure the quality of the output against the expected result.
        Must be implemented by subclasses.
        
        Returns:
            TestResult comparing self.output against self.expected
        """
        pass
    
    def __str__(self):
        return f"TestCase(name={self.name}, input={self.input}, expected={self.expected})"

class TestRunner:
    """Handles running tests and calling APIs"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    async def run(self, test_case: TestCase) -> TestResult:
        """Run a single test case"""
        try:
            # Call API and get response
            response = await self._call_api(test_case.input)
            
            # Store the output
            test_case.output = response
            
            # Measure the result
            result = test_case.measure()
            
            return result
            
        except Exception as e:
            return TestResult(
                status=TestStatus.ERROR,
                score=0.0,
                message=f"Error running test: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _call_api(self, input: Any) -> Any:
        """Make the actual API call"""
        # TODO: Implement actual API call
        # For now, just return a dummy response
        return f"Dummy response for: {input}"

# Example implementation for text comparison
class TextComparisonTest(TestCase[str, str]):
    """A simple test case for comparing text outputs"""
    
    def __init__(
        self,
        input: str,
        expected: str,
        threshold: float = 0.8,
        **kwargs
    ):
        super().__init__(input, expected, **kwargs)
        self.threshold = threshold
    
    def measure(self) -> TestResult:
        if not self.output:
            return TestResult(
                status=TestStatus.ERROR,
                score=0.0,
                message="No output to measure",
                metadata={"error": "No output available"}
            )
        
        # Simple exact match for now
        matches = self.output.strip() == self.expected.strip()
        
        return TestResult(
            status=TestStatus.PASS if matches else TestStatus.FAIL,
            score=1.0 if matches else 0.0,
            message="Exact match" if matches else "Output does not match expected",
            metadata={
                "output": self.output,
                "expected": self.expected
            }
        )

