from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator

from judgeval.data.metric_data import MetricData
from judgeval.data.example import Example


class LLMApiTestCase(BaseModel):
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
    conversational_instance_id: Optional[int] = Field(None)

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
    
def create_api_test_case(
    test_case: Example,
    conversational_instance_id: Optional[int] = None,
) -> LLMApiTestCase:
    """
    When an LLM Test Case is executed, we track its progress using an LLMAPITestCase.

    This will track things like the success of the test case, as well as the metadata (such as verdicts and claims in Faithfulness).
    """
    success = True
    if test_case.name is not None:
        name = test_case.name
    else:
        # raise ValueError(f"Test case name must be provided. Name: {test_case.name}")
        name = "Test Case Placeholder"
    # order = test_case._dataset_rank
    order = None
    metrics_data = []


    api_test_case = LLMApiTestCase(
        name=name,
        input=test_case.input,
        actualOutput=test_case.actual_output,
        expectedOutput=test_case.expected_output,
        context=test_case.context,
        retrievalContext=test_case.retrieval_context,
        toolsCalled=test_case.tools_called,
        expectedTools=test_case.expected_tools,
        success=success,
        metricsData=metrics_data,
        runDuration=None,
        evaluationCost=None,
        order=order,
        additionalMetadata=test_case.additional_metadata,
        conversational_instance_id=conversational_instance_id,
    )
    return api_test_case

