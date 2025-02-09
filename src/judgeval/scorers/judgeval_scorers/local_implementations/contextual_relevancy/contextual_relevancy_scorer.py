from typing import Optional, List, Union
import asyncio

from judgeval.constants import APIScorer
from judgeval.scorers.utils import (get_or_create_event_loop,
                                    scorer_progress_meter,
                                    create_verbose_logs,
                                    parse_response_json,
                                    check_example_params
                                    )
from judgeval.scorers import JudgevalScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers.judgeval_scorers.local_implementations.contextual_relevancy.prompts import *


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.RETRIEVAL_CONTEXT,
]


class ContextualRelevancyScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        user: Optional[str] = None
    ):
        super().__init__(
            score_type=APIScorer.CONTEXTUAL_RELEVANCY,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.user = user
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()

    def score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)
        
        with scorer_progress_meter(self, display_meter=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_score_example(example, _show_indicator=False)
                )
            else:
                self.verdicts_list: List[ContextualRelevancyVerdicts] = [
                    (self._generate_verdicts(example.input, context))
                    for context in example.retrieval_context
                ]
                self.score = self._calculate_score()
                self.reason = self._generate_reason(example.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{[v.model_dump() for v in self.verdicts_list]}",
                        f"Score: {self.score}\nReason: {self.reason}",
                    ],
                )

                return self.score

    async def a_score_example(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)
        
        with scorer_progress_meter(
            self,
            async_mode=True,
            display_meter=_show_indicator,
        ):
            self.verdicts_list: List[ContextualRelevancyVerdicts] = (
                await asyncio.gather(
                    *[
                        self._a_generate_verdicts(example.input, context)
                        for context in example.retrieval_context
                    ]
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(example.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{[v.model_dump() for v in self.verdicts_list]}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        irrelevancies = []
        relevant_statements = []
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                if verdict.verdict.lower() == "no":
                    irrelevancies.append(verdict.model_dump())
                else:
                    relevant_statements.append(verdict.model_dump())
        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            relevant_statements=relevant_statements,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return data["reason"]

    def _generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        irrelevancies = []
        relevant_statements = []
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                if verdict.verdict.lower() == "no":
                    irrelevancies.append(verdict.reason)
                else:
                    relevant_statements.append(verdict.statement)

        prompt: dict = ContextualRelevancyTemplate.generate_reason(
            input=input,
            irrelevancies=irrelevancies,
            relevant_statements=relevant_statements,
            score=format(self.score, ".2f"),
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return data["reason"]

    def _calculate_score(self):
        total_verdicts = 0
        relevant_statements = 0
        for verdicts in self.verdicts_list:
            for verdict in verdicts.verdicts:
                total_verdicts += 1
                if verdict.verdict.lower() == "yes":
                    relevant_statements += 1

        if total_verdicts == 0:
            return 0

        score = relevant_statements / total_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_verdicts(
        self, input: str, context: List[str]
    ) -> ContextualRelevancyVerdicts:
        prompt = ContextualRelevancyTemplate.generate_verdicts(
            input=input, context=context
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            return ContextualRelevancyVerdicts(**data)
        else:
            try:
                res = await self.model.a_generate(
                    prompt, schema=ContextualRelevancyVerdicts
                )
                return res
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                return ContextualRelevancyVerdicts(**data)

    def _generate_verdicts(
        self, input: str, context: str
    ) -> ContextualRelevancyVerdicts:
        prompt = ContextualRelevancyTemplate.generate_verdicts(
            input=input, context=context
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            return ContextualRelevancyVerdicts(**data)
        else:
            try:
                res = self.model.generate(
                    prompt, schema=ContextualRelevancyVerdicts
                )
                return res
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                return ContextualRelevancyVerdicts(**data)

    def _success_check(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Contextual Relevancy"
    