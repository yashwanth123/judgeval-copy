from typing import Optional, List, Union

from judgeval.judges import JudgevalJudge
from judgeval.judges.utils import create_judge
from judgeval.data import Example, ExampleParams
from judgeval.scorers import JudgevalScorer
from judgeval.constants import APIScorer
from judgeval.scorers.utils import (
    get_or_create_event_loop,
    parse_response_json,
    scorer_progress_meter,
    create_verbose_logs,
    check_example_params,
)
from judgeval.scorers.judgeval_scorers.local_implementations.contextual_precision.prompts import *

required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.RETRIEVAL_CONTEXT,
    ExampleParams.EXPECTED_OUTPUT,
]

class ContextualPrecisionScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        model: Optional[Union[str, JudgevalJudge]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
    ):
        super().__init__(
            score_type=APIScorer.CONTEXTUAL_PRECISION,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
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
                self.verdicts: List[ContextualPrecisionVerdict] = (
                    self._generate_verdicts(
                        example.input,
                        example.expected_output,
                        example.retrieval_context,
                    )
                )
                self.score = self._calculate_score()
                self.reason = self._generate_reason(example.input)
                self.success = self.score >= self.threshold
                self.verbose_logs = create_verbose_logs(
                    self,
                    steps=[
                        # Convert to dict for serialization purposes
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
            self.verdicts: List[ContextualPrecisionVerdict] = (
                await self._a_generate_verdicts(
                    example.input,
                    example.expected_output,
                    example.retrieval_context,
                )
            )
            self.score = self._calculate_score()
            self.reason = await self._a_generate_reason(example.input)
            self.success = self.score >= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    # Convert to dict for serialization purposes
                    f"Verdicts:\n{[v.model_dump() for v in self.verdicts]}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def _a_generate_reason(self, input: str):
        if self.include_reason is False:
            return None

        retrieval_contexts_verdicts = [
            {"verdict": verdict.verdict, "reasons": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = ContextualPrecisionTemplate.generate_reason(
            input=input,
            verdicts=retrieval_contexts_verdicts,
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

        retrieval_contexts_verdicts = [
            {"verdict": verdict.verdict, "reasons": verdict.reason}
            for verdict in self.verdicts
        ]
        prompt = ContextualPrecisionTemplate.generate_reason(
            input=input,
            verdicts=retrieval_contexts_verdicts,
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

    async def _a_generate_verdicts(
        self, input: str, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        prompt = ContextualPrecisionTemplate.generate_verdicts(
            input=input,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        if self.using_native_model:
            res = await self.model.a_generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                ContextualPrecisionVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = parse_response_json(res, self)
                verdicts = [
                    ContextualPrecisionVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _generate_verdicts(
        self, input: str, expected_output: str, retrieval_context: List[str]
    ) -> List[ContextualPrecisionVerdict]:
        prompt = ContextualPrecisionTemplate.generate_verdicts(
            input=input,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        if self.using_native_model:
            res = self.model.generate(prompt)
            data = parse_response_json(res, self)
            verdicts = [
                ContextualPrecisionVerdict(**item) for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = parse_response_json(res, self)
                verdicts = [
                    ContextualPrecisionVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _calculate_score(self):
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 0

        # Convert verdicts to a binary list where 'yes' is 1 and others are 0
        node_verdicts = [
            1 if v.verdict.strip().lower() == "yes" else 0
            for v in self.verdicts
        ]

        sum_weighted_precision_at_k = 0.0
        relevant_nodes_count = 0
        for k, is_relevant in enumerate(node_verdicts, start=1):
            # If the item is relevant, update the counter and add the weighted precision at k to the sum
            if is_relevant:
                relevant_nodes_count += 1
                precision_at_k = relevant_nodes_count / k
                sum_weighted_precision_at_k += precision_at_k * is_relevant

        if relevant_nodes_count == 0:
            return 0
        # Calculate weighted cumulative precision
        score = sum_weighted_precision_at_k / relevant_nodes_count
        return 0 if self.strict_mode and score < self.threshold else score

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
        return "Contextual Precision"
