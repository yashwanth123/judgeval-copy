"""
Code that implements a baseline customizable metric.

The CustomizableMetric class is a base class that can be used to create custom metrics.
In order to implement a subclass of CustomizableMetric, you need to implement the following methods:
- build_measure_prompt: builds the prompt that is sent to the model inside of the `evaluate()` method
- is_successful: returns a boolean indicating whether the metric was successful

The core idea of the CustomizableMetric is to provide a flexible way to create 
custom metrics that simply require a prompt to be sent to a model, a JSON response to be parsed,
and a boolean indicating whether the metric was successful. 

For example, if you wanted to use the CustomizableMetric class to create a metric for RCACoPilot accuracy,
you'd simply tweak the build_measure_prompt method to construct the prompt for evaluating the accuracy of an
RCA CoPilot response (given a Testcase), and the is_successful method to return a boolean indicating whether the
response was accurate.

NOTE: There are some things to keep in mind when creating the prompt in the build_measure_prompt method:
- The prompt should instruct the LLM to generate an output JSON that contains a "score" and "reason" field.
- The "score" field can be anything that you determine to be the result of the evaluation
- The "reason" field should contain a string that explains why the score was calculated as it was

In the context of the RCACoPilot example, the score could be whether or not the hypothesis was matching the gold hypothesis,
and the reason could be why the hypothesis was or was not matching the gold hypothesis.
"""

from abc import abstractmethod
from typing import List, Optional, Union, Tuple, Any
from pydantic import BaseModel

from judgeval.data import Example
from judgeval.scorers import CustomScorer
from judgeval.judges import judgevalJudge
from judgeval.scorers.utils import (scorer_progress_meter, 
                                    parse_response_json,
                                    get_or_create_event_loop,
                                    create_verbose_logs)
from judgeval.judges.utils import create_judge


class ReasonScore(BaseModel):
    reason: str
    score: float


class PromptScorer(CustomScorer):
    def __init__(
        self,
        name: str, 
        threshold: float = 0.5,
        model: Optional[Union[str, judgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        self.name = name
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = create_judge(model)
        self.using_native_model = True  # NOTE: SETTING THIS FOR LITELLM and TOGETHER usage
        self.evaluation_model = self.model.get_model_name()
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        
    def score_example(
            self, 
            example: Example, 
            _show_indicator: bool = True
            ) -> float:
        """
        Synchronous method for scoring an example using the prompt criteria.
        """
        with scorer_progress_meter(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(example, _show_indicator=False)
                )
            else:
                result, reason = self.evaluate(example)
                self.reason = reason
                self.result = result
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Results: {self.result}\nReason: {self.reason}",
                    ],
                )
                return result

    async def a_score_example(
            self,
            example: Example,
            _show_indicator: bool = True,
            ) -> float: 
        """
        Async method for scoring an example using the prompt criteria.
        """
        with scorer_progress_meter(self, display_meter=_show_indicator):
            result, reason = await self.a_evaluate(example)
            self.reason = reason
            self.result = result
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Results: {self.result}\nReason: {self.reason}",
                ],
            )
            return result
    
    def evaluate(self, example: Example) -> Tuple[Any, str]:
        """
        Synchronous helper method for evaluating an example using the prompt criteria.

        Builds a custom prompt using `build_measure_prompt` and sends it to the judge model 
        for evaluation. The result is then parsed as JSON and returned.

        NOTE: It is assumed that the model response will be JSON and contain a "score" and "reason" field.
        """
        prompt = self.build_measure_prompt(example)
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["score"], data["reason"]
        else:
            try:
                res: ReasonScore = self.model.generate(
                    prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["score"], data["reason"]

    async def a_evaluate(self, example: Example) -> Tuple[Any, str]:
        """
        Asynchronous helper method for evaluating an example using the prompt criteria.

        Builds a custom prompt using `build_measure_prompt` and sends it to the judge model 
        for evaluation. The result is then parsed as JSON and returned.

        NOTE: It is assumed that the model response will be JSON and contain a "score" and "reason" field.
        """
        prompt = self.build_measure_prompt(example)
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)

            self.score = data["score"]
            self.reason = data["reason"]
            self.response = data
            return data["score"], data["reason"]
        else:
            try:
                res: ReasonScore = await self.model.a_generate(
                    prompt, schema=ReasonScore
                )
                return res.score, res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["score"], data["reason"]

    @abstractmethod
    def build_measure_prompt(self, example: Example, *args, **kwargs) -> Union[str, List[dict]]:
        # builds the prompt that is sent to the model inside of the `score_example()` method
        # returns either a string prompt or a conversation prompt of the form [{"role": "system", "content": "..."}, ...]
        pass

    @abstractmethod
    def success_check(self, **kwargs):
        pass
    
    @property
    def __name__(self):
        return "Prompt Scorer"