from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class GroundTruthExample(BaseModel):
    """
    GroundTruthExample is the atomic unit of a `Dataset`. It is essentially the same
    as an `Example`, but the `actual_output` field is optional to enable users to 
    run their workflow on the `input` field at test-time to evaluate their current 
    workflow's performance.
    """
    input: str
    actual_output: Optional[str] = Field(
        default=None, serialization_alias="actualOutput"
    )
    expected_output: Optional[str] = Field(
        default=None, serialization_alias="expectedOutput"
    )
    context: Optional[List[str]] = Field(default=None)
    retrieval_context: Optional[List[str]] = Field(
        default=None, serialization_alias="retrievalContext"
    )
    additional_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="additionalMetadata"
    )
    comments: Optional[str] = Field(default=None)
    tools_called: Optional[List[str]] = Field(
        default=None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[str]] = Field(
        default=None, serialization_alias="expectedTools"
    )
    source_file: Optional[str] = Field(
        default=None, serialization_alias="sourceFile"
    )

    def to_dict(self):
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "context": self.context,
            "retrieval_context": self.retrieval_context,
            "additional_metadata": self.additional_metadata,
            "comments": self.comments,
            "tools_called": self.tools_called,
            "expected_tools": self.expected_tools,
            "source_file": self.source_file,
        }
