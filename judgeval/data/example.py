"""
Classes for representing examples in a dataset.
"""


from typing import TypeVar, Optional, Any, Dict, List
from pydantic import BaseModel
from enum import Enum
from datetime import datetime


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
    example_id: Optional[int] = None
    timestamp: Optional[str] = None

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

    def __init__(self, **data):
        super().__init__(**data)
        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self):
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "additional_metadata": self.additional_metadata,
            "tools_called": self.tools_called,
            "expected_tools": self.expected_tools,
            "name": self.name,
            "example_id": self.example_id,
            "timestamp": self.timestamp,
        }

    def __str__(self):
        return (
            f"Example(input={self.input}, "
            f"actual_output={self.actual_output}, "
            f"expected_output={self.expected_output}, "
            f"context={self.context}, "
            f"retrieval_context={self.retrieval_context}, "
            f"additional_metadata={self.additional_metadata}, "
            f"tools_called={self.tools_called}, "
            f"expected_tools={self.expected_tools}, "
            f"name={self.name}, "
            f"example_id={self.example_id}, "
            f"timestamp={self.timestamp})"
        )
