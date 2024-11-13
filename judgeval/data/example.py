"""
Classes for representing examples in a dataset.
"""


from typing import Generic, TypeVar, Optional, Any, Dict, Union, List
from pydantic import BaseModel, Field, ConfigDict, model_validator
from enum import Enum


Input = TypeVar('Input')
Output = TypeVar('Output')

class ExampleParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    REASONING = "reasoning"


class Example(BaseModel):
    input: Input
    actual_output: Output
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    name: Optional[str] = None

    def __post_init__(self):
        # Ensure `context` is None or a list of strings
        if self.context is not None:
            if not isinstance(self.context, list) or not all(
                isinstance(item, str) for item in self.context
            ):
                raise TypeError("'context' must be None or a list of strings")

        # Ensure `retrieval_context` is None or a list of strings
        if self.retrieval_context is not None:
            if not isinstance(self.retrieval_context, list) or not all(
                isinstance(item, str) for item in self.retrieval_context
            ):
                raise TypeError(
                    "'retrieval_context' must be None or a list of strings"
                )

        # Ensure `tools_called` is None or a list of strings
        if self.tools_called is not None:
            if not isinstance(self.tools_called, list) or not all(
                isinstance(item, str) for item in self.tools_called
            ):
                raise TypeError(
                    "'tools_called' must be None or a list of strings"
                )

        # Ensure `expected_tools` is None or a list of strings
        if self.expected_tools is not None:
            if not isinstance(self.expected_tools, list) or not all(
                isinstance(item, str) for item in self.expected_tools
            ):
                raise TypeError(
                    "'expected_tools' must be None or a list of strings"
                )
