from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator

from judgeval.data.example import Example
from judgeval.data.scorer_data import ScorerData
from judgeval.common.logger import debug, error

class ProcessExample(BaseModel):
    """
    ProcessExample is an `Example` object that contains intermediate information 
    about an undergoing evaluation on the original `Example`. It is used purely for
    internal operations and keeping track of the evaluation process.
    """
    name: str
    input: Optional[str] = None
    actual_output: Optional[str] = Field(None, alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    tools_called: Optional[list] = Field(None, alias="toolsCalled")
    expected_tools: Optional[list] = Field(None, alias="expectedTools")

    # make these optional, not all test cases in a conversation will be evaluated
    success: Union[bool, None] = Field(None)
    scorers_data: Union[List[ScorerData], None] = Field(
        None, alias="scorersData"
    )
    run_duration: Union[float, None] = Field(None, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")

    order: Union[int, None] = Field(None)
    # These should map 1 to 1 from golden
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    comments: Optional[str] = Field(None)
    trace_id: Optional[str] = Field(None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_scorer_data(self, scorer_data: ScorerData):
        """
        Updates scorer data field of test case after the scorers have been
        evaluated on this test case.
        """
        debug(f"Updating scorer data for example '{self.name}' with scorer: {scorer_data}")
        # self.scorers_data is a list of ScorerData objects that contain the 
        # evaluation results of each scorer on this test case
        if self.scorers_data is None:
            self.scorers_data = [scorer_data]
        else:
            self.scorers_data.append(scorer_data)

        if self.success is None:
            # self.success will be None when it is a message
            # in that case we will be setting success for the first time
            self.success = scorer_data.success
        else:
            if scorer_data.success is False:
                debug(f"Example '{self.name}' marked as failed due to scorer: {scorer_data}")
                self.success = False

    def update_run_duration(self, run_duration: float):
        self.run_duration = run_duration

    @model_validator(mode="before")
    def check_input(cls, values: Dict[str, Any]):
        input = values.get("input")
        actual_output = values.get("actualOutput")

        if (input is None or actual_output is None):
            error(f"Validation error: Required fields missing. input={input}, actualOutput={actual_output}")
            raise ValueError(
                "'input' and 'actualOutput' must be provided."
            )

        return values
    

def create_process_example(
    example: Example,
) -> ProcessExample:
    """
    When an LLM Test Case is executed, we track its progress using an ProcessExample.

    This will track things like the success of the test case, as well as the metadata (such as verdicts and claims in Faithfulness).
    """
    success = True
    if example.name is not None:
        name = example.name
    else:
        name = "Test Case Placeholder"
        debug(f"No name provided for example, using default name: {name}")
    order = None
    scorers_data = []

    debug(f"Creating ProcessExample for: {name}")
    process_ex = ProcessExample(
        name=name,
        input=example.input,
        actualOutput=example.actual_output,
        expectedOutput=example.expected_output,
        context=example.context,
        retrievalContext=example.retrieval_context,
        toolsCalled=example.tools_called,
        expectedTools=example.expected_tools,
        success=success,
        scorersData=scorers_data,
        runDuration=None,
        evaluationCost=None,
        order=order,
        additionalMetadata=example.additional_metadata,
        trace_id=example.trace_id
    )
    return process_ex

