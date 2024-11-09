from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator

from judgeval.data.metric_data import MetricData
from judgeval.data.example import Example


class processExample(BaseModel):
    """
    processExample is an `Example` object that contains intermediate information 
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
    metrics_data: Union[List[MetricData], None] = Field(
        None, alias="metricsData"
    )
    run_duration: Union[float, None] = Field(None, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")

    order: Union[int, None] = Field(None)
    # These should map 1 to 1 from golden
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    comments: Optional[str] = Field(None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update_metric_data(self, metric_data: MetricData):
        """
        Updates metric data field of test case after the metrics have been
        evaluated on this test case.
        """
        # self.metrics_data is a list of MetricData objects that contain the 
        # evaluation results of each metric on this test case
        if self.metrics_data is None:
            self.metrics_data = [metric_data]
        else:
            self.metrics_data.append(metric_data)

        if self.success is None:
            # self.success will be None when it is a message
            # in that case we will be setting success for the first time
            self.success = metric_data.success
        else:
            if metric_data.success is False:
                self.success = False

        # Track evaluation costs
        evaluationCost = metric_data.evaluation_cost
        if evaluationCost is None:
            return

        if self.evaluation_cost is None:
            self.evaluation_cost = evaluationCost
        else:
            self.evaluation_cost += evaluationCost

    def update_run_duration(self, run_duration: float):
        self.run_duration = run_duration

    @model_validator(mode="before")
    def check_input(cls, values: Dict[str, Any]):
        input = values.get("input")
        actual_output = values.get("actualOutput")

        if (input is None or actual_output is None):
            raise ValueError(
                "'input' and 'actualOutput' must be provided."
            )

        return values
    

def create_process_example(
    example: Example,
) -> processExample:
    """
    When an LLM Test Case is executed, we track its progress using an LLMAPITestCase.

    This will track things like the success of the test case, as well as the metadata (such as verdicts and claims in Faithfulness).
    """
    success = True
    if example.name is not None:
        name = example.name
    else:
        # raise ValueError(f"Test case name must be provided. Name: {test_case.name}")
        name = "Test Case Placeholder"
    # order = test_case._dataset_rank
    order = None
    metrics_data = []


    api_test_case = processExample(
        name=name,
        input=example.input,
        actualOutput=example.actual_output,
        expectedOutput=example.expected_output,
        context=example.context,
        retrievalContext=example.retrieval_context,
        toolsCalled=example.tools_called,
        expectedTools=example.expected_tools,
        success=success,
        metricsData=metrics_data,
        runDuration=None,
        evaluationCost=None,
        order=order,
        additionalMetadata=example.additional_metadata,
    )
    return api_test_case

