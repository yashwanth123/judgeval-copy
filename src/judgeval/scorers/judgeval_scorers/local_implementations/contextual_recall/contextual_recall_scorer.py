from typing import Optional, List, Union

from judgeval.constants import APIScorer
from judgeval.scorers.utils import (
    get_or_create_event_loop, 
    parse_response_json, 
    scorer_progress_meter, 
    create_verbose_logs,
    check_example_params
)
from judgeval.judges.utils import create_judge
from judgeval.scorers import JudgevalScorer
from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers.judgeval_scorers.local_implementations.contextual_recall.prompts import *

required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.EXPECTED_OUTPUT,
    ExampleParams.RETRIEVAL_CONTEXT,
]

class ContextualRecallScorer(JudgevalScorer):
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
            score_type=APIScorer.CONTEXTUAL_RECALL,
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
                self.verdicts: List[ContextualRecallVerdict] = (
                    self._generate_verdicts(
                        example.expected_output, example.retrieval_context
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(example.expected_output)
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
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
            self.verdicts: List[ContextualRecallVerdict] = (
                await self._a_generate_verdicts(
                    example.expected_output, example.retrieval_context
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(
                example.expected_output
            )
            self.success = self.score >= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, expected_output: str):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for idx, verdict in enumerate(self.verdicts):
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(f"Sentence {idx + 1}: {verdict.reason}")
            else:
                unsupportive_reasons.append(f"Sentence {idx + 1}: {verdict.reason}")

        prompt = ContextualRecallTemplate.generate_reason(
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
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

    def _generate_reason(self, expected_output: str):
        if self.include_reason is False:
            return None

        supportive_reasons = []
        unsupportive_reasons = []
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                supportive_reasons.append(verdict.reason)
            else:
                unsupportive_reasons.append(verdict.reason)

        prompt = ContextualRecallTemplate.generate_reason(
            expected_output=expected_output,
            supportive_reasons=supportive_reasons,
            unsupportive_reasons=unsupportive_reasons,
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
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        justified_sentences = 0
        for verdict in self.verdicts:
            if verdict.verdict.lower() == "yes":
                justified_sentences += 1

        score = justified_sentences / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_verdicts(
        self, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualRecallVerdict]:
        prompt = ContextualRecallTemplate.generate_verdicts(
            expected_output=expected_output, retrieval_context=retrieval_context
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                ContextualRecallVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts: Verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                verdicts = [
                    ContextualRecallVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(
        self, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualRecallVerdict]:
        prompt = ContextualRecallTemplate.generate_verdicts(
            expected_output=expected_output, retrieval_context=retrieval_context
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                ContextualRecallVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts: Verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                verdicts = [
                    ContextualRecallVerdict(**item) for item in data["verdicts"]
                ]
                return verdicts

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
        return "Contextual Recall"