from typing import Generic, TypeVar, Optional, Any, Dict, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel

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


